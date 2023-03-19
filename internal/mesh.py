from importlib.resources import path
import os
import pdb
import torch
import skfmm
import mcubes
import trimesh
import numpy as np
import pymeshlab as ml
from .utils import prCyan, prYellow, prBlue, prRed
from os.path import exists, join
from types import SimpleNamespace
from torch import Tensor
import trimesh.ray.ray_pyembree as RayMeshIntersector

@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def extract_fields(bound_min, bound_max, resolution, query_func, fn):
    '''Creating point grid cube to extract density
    Args:
      bound_min: the minimun bound that the scene can reach (e.g. [-1,-1,-1])
      bound_max: the maximun bound that the scene can reach (e.g. [1,1,1])
      resolution:  is the number of distinct points in each dimension (x,y,z) 
        that the point grid cube is compose.
      query_func: function used for passing queries to network_fn.
      fn: function. Model for predicting RGB and density at each point
        in space.
    Returns:
      u: Estimated density per each point of the point grid cube. 
    '''
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([
                        xx.reshape(-1, 1),
                        yy.reshape(-1, 1),
                        zz.reshape(-1, 1)
                    ],
                                    dim=-1)
                    val = query_func(pts[:, None, :], torch.zeros_like(pts),
                                     fn)[..., -1].reshape(
                                         len(xs), len(ys),
                                         len(zs)).detach().cpu().numpy()
                    u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys),
                      zi * N:zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func,
                     fn, level_set, from_file):
    prYellow(f'threshold: {threshold}, level_set: {level_set}')

    # Extracting the estimated density of a point grid cube within scene bounds
    if from_file is None:
        u = extract_fields(bound_min, bound_max, resolution, query_func, fn)
        np.save('nose.npy',u)
        pdb.set_trace()
    else:
        u = np.load(from_file)
    if torch.is_tensor(bound_max):
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()
    else:
        b_max_np = np.array(bound_max)
        b_min_np = np.array(bound_min)
    # print(np.histogram(u, bins=10)[0]) # Where is the density?
    # print(np.histogram(u, bins=10)[1])
    # Extracting mesh
    if level_set != 0:
        # Extracting the inflated mesh according a distance by using SDF
        u = skfmm.distance(u - threshold,
                           dx=((b_max_np - b_min_np) / resolution))
        vertices, triangles = mcubes.marching_cubes(u, level_set)
    else:
        # Extracting directly the mesh according to a threshold
        vertices, triangles = mcubes.marching_cubes(u, threshold)

    # Scaling vertices according scene bounds and resolution.
    vertices = vertices / (resolution - 1.0) * \
        (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class Mesh:
    def __init__(self,
                 mesh_path,
                 do_tri2quad=False,
                 do_simplify=False,
                 targetlen=0.1,
                 **kwargs):
        args = SimpleNamespace(**kwargs)
        self.mesh_path = mesh_path

        # Create mesh 
        if 'model' in kwargs:
            self._create_mesh(args.model, args.resolution, args.threshold,
                              args.level_set, args.num_comp, args.from_file)

        # Load the mesh 
        self.mesh = self._load(mesh_path)

        if do_tri2quad:
            # From triangles to quads
            self._vertices, self._faces = self._tri2quad(self.mesh)
        else:
            self._vertices, self._faces = self.mesh.vertices, self.mesh.faces

    def get_data(self):
        return self._vertices, self._faces

    def _load(self, path):
        return trimesh.load(path, process=False, maintain_order=True)

    def _save(self, mesh, meshlab=False):
        os.makedirs(join(*self.mesh_path.split('/')[:-1]), exist_ok=True)
        if meshlab:
            mesh.save_current_mesh(self.mesh_path)
        else:    
            mesh.export(self.mesh_path)
        
    def _trimesh2Pymeshlab(self, mesh):
        # import mesh to pymeshlab
        mesh_meshlab = ml.Mesh(vertex_matrix=mesh.vertices,
                               face_matrix=mesh.faces)
        meshset = ml.MeshSet()
        meshset.add_mesh(mesh_meshlab)
        return meshset


    def _tri2quad(self, mesh):
         # import mesh to pymeshlab
        meshset = self._trimesh2Pymeshlab(mesh)

        # triangles to quads
        meshset.apply_filter("tri_to_quad_by_4_8_subdivision")
        # meshset.apply_filter("tri_to_quad_by_smart_triangle_pairing")
        # meshset.apply_filter("meshing_tri_to_quad_by_smart_triangle_pairing")

        # vertices: Nx3, faces: Mx4
        vertices = meshset.current_mesh().vertex_matrix()
        faces = np.stack(meshset[0].polygonal_face_list())
        return vertices, faces
    
    def intersected_points(self, rays_o, rays_d):
        rayMesh = RayMeshIntersector.RayMeshIntersector(self.mesh)
        pts, idx_ray, idx_tri = rayMesh.intersects_location(rays_o,
                                                            rays_d,
                                                            multiple_hits=False)
        # Corresponding vertices from faces intersected
        vtx = self.mesh.vertices[self.mesh.faces[idx_tri]]
        return pts, idx_ray, vtx, idx_tri  

    def _create_mesh(self, model, resolution, threshold, level_set, num_comp, from_file):
        if model.dataset_type == 'blender':
            poses =  model.dataset.poses
            hwk =  model.dataset.hwk
            near =  model.dataset.near
            far =  model.dataset.far

        # Getting bounds of the scene
        
        b_min = model.b_min[0].split(',')
        b_min = [float(b_min[0]), float(b_min[1]), float(b_min[2])]
        b_max = model.b_max[0].split(',')
        b_max = [float(b_max[0]), float(b_max[1]), float(b_max[2])]

        prCyan(f'b_max: {b_max}, b_min: {b_min}')
        # 1st stage model declaration
        net = model.test_model(args=model)

        # Extracting vertices and faces from a point grid cube
        v, t = extract_geometry(b_min,
                                b_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=net['network_query_fn'],
                                fn=net['network_fine'],
                                level_set=level_set,
                                from_file=from_file)

        # Creating a mesh with v and f
        mesh = trimesh.Trimesh(v, t)
        print('Geometry extraction finish')
        # Post_processing: keep largest component
        if num_comp > 0:
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components])

            # Keep the Nth components of the mesh
            mesh = trimesh.util.concatenate( \
                components[np.argpartition(areas, -num_comp)[-num_comp:]])
        # Save mesh
        self._save(mesh)

    def _postprocess(self, 
                    mesh,  
                    targetlen=0.1):
         # import mesh to pymeshlab
        meshset = self._trimesh2Pymeshlab(mesh)

        # MC mesh has low quality redundant triangles, simplification is required
        # Do decimation: less faces and vertices
        meshset = self._decimation(meshset)
        
        # simplify mesh
        # meshset.apply_filter(
        #     "simplification_edge_collapse_for_marching_cube_meshes")
            
        # Do decimation: less faces and vertices
        # meshset = self._decimation(meshset)

        # make triangles similar size
        # targetlen = ml.AbsoluteValue(targetlen)
        # meshset.apply_filter("remeshing_isotropic_explicit_remeshing",
        #                         targetlen=targetlen)

        self._save(meshset, meshlab=True)

    def _decimation(self, mesh):
        ms = mesh.current_mesh()
        prYellow(f'Input mesh: {ms.vertex_number()} V and {ms.face_number()} F')

        # Target number of vertex
        TARGET = 75000
        # Estimate number of faces to have 100+10000 vertex using Euler
        numFaces = 100 + 2 * TARGET

        # Simplify the mesh. Only first simplification will be agressive
        while (ms.vertex_number() > TARGET):
            # mesh.apply_filter('simplification_quadric_edge_collapse_decimation',
            #                 targetfacenum=numFaces,
            #                 preservenormal=True)
            mesh.apply_filter('meshing_decimation_quadric_edge_collapse',
                            targetfacenum=numFaces,
                            preservenormal=True)

            vertex = mesh.current_mesh().vertex_number()
            print(f'Decimated to {numFaces} F and {vertex} V')

            # Refine our estimation to slowly converge to TARGET vertex number
            numFaces = numFaces - (mesh.current_mesh().vertex_number() - TARGET)

            ms = mesh.current_mesh()
        prYellow(f'Output mesh: {ms.vertex_number()}  V and {ms.face_number()} F')

        return mesh
        
    @staticmethod
    def export_mesh(save_path, vertices, faces, texture_vertices=None):
        # Exporting mesh with textures: features
        vertex_to_str = lambda v: "v " + str(v)[1:-1] + "\n"
        texture_to_str = lambda vt: "vt " + str(vt)[1:-1] + "\n"
        face3_to_str = lambda v, vt: f"f {v[0]+1}/{vt[0]+1} {v[1]+1}/{vt[1]+1} {v[2]+1}/{vt[2]+1}\n"
        faceN_to_str = lambda v: "f " + str(v)[1:-1] + "\n"
        # import pdb; pdb.set_trace()
        with open(save_path, 'w') as file:
            # export material
            if texture_vertices is not None:
                file.write("mtllib ./test.obj.mtl\n\n")

            # export vertices
            for v in vertices:
                file.write(vertex_to_str(v))
            # import pdb; pdb.set_trace()

            if texture_vertices is None:
                # export faces without texture
                for f in faces:
                    file.write(faceN_to_str(Geometry.bot_triangle(f + 1)))
                    file.write(faceN_to_str(Geometry.top_triangle(f + 1)))
            else:
                # export faces with texture coords
                file.write("\nusemtl material_0\n")
                for f_id, f in enumerate(faces):

                    # export texture coords for the given face
                    face_vt = texture_vertices[f_id]
                    for vt in face_vt:
                        file.write(texture_to_str(vt))

                    # export face
                    vt_ids = range(f_id * 4, (f_id + 1) * 4)
                    file.write(
                        face3_to_str(Geometry.bot_triangle(f),
                                     Geometry.bot_triangle(vt_ids)))
                    file.write(
                        face3_to_str(Geometry.top_triangle(f),
                                     Geometry.top_triangle(vt_ids)))
    @staticmethod
    def export_mesh_quad(save_path, vertices, faces, texture_vertices=None):
        # Exporting mesh with textures: features
        vertex_to_str = lambda v: "v " + str(v)[1:-1] + "\n"
        texture_to_str = lambda vt: "vt " + str(vt)[1:-1] + "\n"
        face3_to_str = lambda v, vt: f"f {v[0]+1}/{vt[0]+1} {v[1]+1}/{vt[1]+1} {v[2]+1}/{vt[2]+1} {v[3]+1}/{vt[3]+1}\n"
        faceN_to_str = lambda v: "f " + str(v)[1:-1] + "\n"
        # import pdb; pdb.set_trace()
        with open(save_path, 'w') as file:
            # export material
            if texture_vertices is not None:
                file.write("mtllib ./test.obj.mtl\n\n")

            # export vertices
            for v in vertices:
                file.write(vertex_to_str(v))
            # import pdb; pdb.set_trace()

            if texture_vertices is None:
                # export faces without texture
                for f in faces:
                    # file.write(faceN_to_str(Geometry.quad(f + 1)))
                    # file.write(faceN_to_str(Geometry.top_triangle(f + 1)))
                    print('miaw')
            else:
                # export faces with texture coords
                file.write("\nusemtl material_0\n")
                for f_id, f in enumerate(faces):

                    # export texture coords for the given face
                    face_vt = texture_vertices[f_id]
                    for vt in face_vt:
                        file.write(texture_to_str(vt))

                    # export face
                    vt_ids = range(f_id * 4, (f_id + 1) * 4)
                    file.write(
                        face3_to_str(Geometry.quad(f),
                                     Geometry.quad(vt_ids)))

    @staticmethod
    def export_mesh_tri(save_path, vertices, faces, texture_vertices=None):
        # Exporting mesh with textures: features
        vertex_to_str = lambda v: "v " + str(v)[1:-1] + "\n"
        texture_to_str = lambda vt: "vt " + str(vt)[1:-1] + "\n"
        face3_to_str = lambda v, vt: f"f {v[0]+1}/{vt[0]+1} {v[1]+1}/{vt[1]+1} {v[2]+1}/{vt[2]+1}\n"
        faceN_to_str = lambda v: "f " + str(v)[1:-1] + "\n"
        # import pdb; pdb.set_trace()
        with open(save_path, 'w') as file:
            # export material
            if texture_vertices is not None:
                file.write("mtllib ./test.obj.mtl\n\n")

            # export vertices
            for v in vertices:
                file.write(vertex_to_str(v))
            # import pdb; pdb.set_trace()

            if texture_vertices is None:
                # export faces without texture
                print('tranquilo jesus')
                # for f in faces:
                #     file.write(faceN_to_str(Geometry.bot_triangle(f + 1)))
                #     file.write(faceN_to_str(Geometry.top_triangle(f + 1)))
            else:
                # export faces with texture coords
                file.write("\nusemtl material_0\n")
                for f_id, f in enumerate(faces):

                    # export texture coords for the given face
                    face_vt = texture_vertices[f_id]
                    for vt in face_vt:
                        file.write(texture_to_str(vt))

                    # export face
                    vt_ids = range(f_id * 3, (f_id + 1) * 3)
                    file.write(
                        face3_to_str(f, vt_ids))
                    
                   

class Geometry:
    # v
    # 3 - 2 j
    # | / |
    # 0 - 1 u
    # i

    @staticmethod
    def triangle_2d_to_3d(tri_2d, tri_3d, pts_2d):
        # pdb.set_trace()
        # pts_2d to homogeneous coordinates
        pts_2d_h = np.vstack((pts_2d.T, np.ones((1, pts_2d.shape[0]))))  # 3xN

        # splits triangles per components (3 points)
        q0, q1, q2 = tri_2d
        p0, p1, p2 = tri_3d

        # compute barycentric coordinates bar of pts_2d within tri_2d
        M_2d = np.array([[(q1[0] - q0[0]), (q2[0] - q0[0]), q0[0]],
                         [(q1[1] - q0[1]), (q2[1] - q0[1]), q0[1]],
                         [0., 0., 1.]])
        bar_coords = np.linalg.inv(M_2d) @ pts_2d_h  # 3x3 x 3xN[vt1]

        # convert bar_coords to cartessian using triangle tri_3d
        M_3d = np.array([[(p1[0] - p0[0]), (p2[0] - p0[0]), p0[0]],
                         [(p1[1] - p0[1]), (p2[1] - p0[1]), p0[1]],
                         [(p1[2] - p0[2]), (p2[2] - p0[2]), p0[2]]])
        pts_3d = M_3d @ bar_coords

        return pts_3d

    @staticmethod
    def triangle_2d_to_3d_bz(tri_2d, tri_3d, pts_2d):
        # pts_2d to homogeneous coordinates
        pts_2d_h = torch.cat([pts_2d.transpose(1,2), torch.ones((pts_2d.shape[0], 1, pts_2d.shape[1]))], axis=1)  # 3xN
        # splits triangles per components (3 points)
        q0, q1, q2 = tri_2d[:,0], tri_2d[:,1], tri_2d[:,2]
        p0, p1, p2 = tri_3d[:,0], tri_3d[:,1], tri_3d[:,2]
        # compute barycentric coordinates bar of pts_2d within tri_2d
        M_2d = torch.zeros_like(tri_3d)
        M_2d[:,0,0] = q1[:,0] - q0[:,0]
        M_2d[:,0,1] = q2[:,0] - q0[:,0]
        M_2d[:,0,2] = q0[:,0]
        M_2d[:,1,0] = q1[:,1] - q0[:,1]
        M_2d[:,1,1] = q2[:,1] - q0[:,1]
        M_2d[:,1,2] = q0[:,1]
        M_2d[:,2,2] = 1
        bar_coords = torch.bmm(torch.inverse(M_2d), pts_2d_h)  # 3x3 x 3xN[vt1]
        # convert bar_coords to cartessian using triangle tri_3d
        M_3d = torch.zeros_like(tri_3d)
        M_3d[:,0,0] = p1[:,0] - p0[:,0]
        M_3d[:,0,1] = p2[:,0] - p0[:,0]
        M_3d[:,0,2] = p0[:,0]
        M_3d[:,1,0] = p1[:,1] - p0[:,1]
        M_3d[:,1,1] = p2[:,1] - p0[:,1]
        M_3d[:,1,2] = p0[:,1]
        M_3d[:,2,0] = p1[:,2] - p0[:,2]
        M_3d[:,2,1] = p2[:,2] - p0[:,2]
        M_3d[:,2,2] = p0[:,2]
        pts_3d = torch.bmm(M_3d, bar_coords)
        return pts_3d

    @staticmethod
    def triangle_3d_to_2d_bz(tri_2d, tri_3d, pts_3d):
        # # pts_2d to homogeneous coordinates
        # pts_2d_h = torch.cat([pts_2d.transpose(1,2), torch.ones((pts_2d.shape[0], 1, pts_2d.shape[1]))], axis=1)  # 3xN
        # splits triangles per components (3 points)
        q0, q1, q2 = tri_2d[:,0], tri_2d[:,1], tri_2d[:,2]
        p0, p1, p2 = tri_3d[:,0], tri_3d[:,1], tri_3d[:,2]
        # compute barycentric coordinates bar of pts_2d within tri_2d
        M_2d = torch.zeros_like(tri_3d)
        M_2d[:,0,0] = q1[:,0] - q0[:,0]
        M_2d[:,0,1] = q2[:,0] - q0[:,0]
        M_2d[:,0,2] = q0[:,0]
        M_2d[:,1,0] = q1[:,1] - q0[:,1]
        M_2d[:,1,1] = q2[:,1] - q0[:,1]
        M_2d[:,1,2] = q0[:,1]
        M_2d[:,2,2] = 1
        # convert bar_coords to cartessian using triangle tri_3d
        M_3d = torch.zeros_like(tri_3d)
        M_3d[:,0,0] = p1[:,0] - p0[:,0]
        M_3d[:,0,1] = p2[:,0] - p0[:,0]
        M_3d[:,0,2] = p0[:,0]
        M_3d[:,1,0] = p1[:,1] - p0[:,1]
        M_3d[:,1,1] = p2[:,1] - p0[:,1]
        M_3d[:,1,2] = p0[:,1]
        M_3d[:,2,0] = p1[:,2] - p0[:,2]
        M_3d[:,2,1] = p2[:,2] - p0[:,2]
        M_3d[:,2,2] = p0[:,2]
        # pdb.set_trace()
        bar_coords = torch.bmm(torch.inverse(M_3d),  pts_3d[...,None]) # 3x3 x 3xN
        pts_2d = torch.bmm(M_2d, bar_coords)  # 3x3 x 3xN[vt1]
        # pdb.set_trace()
        return pts_2d
    @staticmethod
    def quad_2d_to_3d(quad_3d, pts_2d, vt, v):
        qs = int(np.sqrt(pts_2d.shape[0]))
        _quad_3d = np.stack(quad_3d)
        _quad_3d = _quad_3d[[0,3,1,2]]
        quad_3d = torch.Tensor(_quad_3d.T).reshape(1,3,2,2)
        pts_3d =  torch.nn.functional.interpolate(quad_3d, size=(qs,qs), mode='bilinear', align_corners=True)
        min_x = pts_2d[:,0].min()
        min_y = pts_2d[:,1].min()
        return pts_3d[:,:,pts_2d[:,0]-min_x,pts_2d[:,1]-min_y].reshape(3,-1).t()

    @staticmethod
    def top_triangle(quad):
        # returns data corresponding to the bottom triangle of the quad
        return quad[0], quad[2], quad[3]

    @staticmethod
    def bot_triangle(quad):
        # returns data corresponding to the top triangle of the quad
        return quad[0], quad[1], quad[2]

    @staticmethod
    def quad(quad):
        # returns data corresponding to the bottom triangle of the quad
        return quad[0], quad[1], quad[2], quad[3]

    @staticmethod
    def sample_pts_2d(vt):
        # pixels inside quad
        u_min, v_min, u_max, v_max = vt[0, 0], vt[0, 1], vt[2, 0], vt[2, 1]
        grid = np.mgrid[u_min:u_max, v_min:v_max].reshape(2, -1).T

        # mask per triangle
        mask_bot = np.tri(u_max - u_min, v_max - v_min).reshape(-1)  # N
        mask_top = 1 - np.tri(u_max - u_min, v_max - v_min, -1).reshape(-1)

        # points per triangle
        pts_bot = grid[mask_bot == 1, :]
        pts_top = grid[mask_top == 1, :]
        return pts_bot, pts_top  # (Nx2, Nx2)

    @staticmethod
    def sample_pts_2d_tri(vt, face_id):
        u_min, v_min, u_max, v_max = vt[0, 0], vt[0, 1], vt[1, 0], vt[2, 1]
        # pixels inside quad
        grid = np.mgrid[u_min:u_max, v_min:v_max].reshape(2, -1).T

        # mask per triangle
        mask_bot = np.tri(u_max - u_min, v_max - v_min).reshape(-1)  # N
        mask_top = 1 - np.tri(u_max - u_min, v_max - v_min).reshape(-1)
        # mask_top = 1 - np.tri(u_max - u_min, v_max - v_min, -1).reshape(-1)
        # points per triangle
        pts_bot = grid[mask_bot == 1, :]
        pts_top = grid[mask_top == 1, :]
        return pts_bot, pts_top  # (Nx2, Nx2)

    @staticmethod
    def sample_pts_2d_tri_bz(vt, face_id):
        vt_np = vt.cpu().numpy()
        u_min, v_min, u_max, v_max = vt_np[:, 0, 0], vt_np[:, 0, 1], vt_np[:, 1, 0], vt_np[:, 2, 1]
        # pixels inside quad
        
        grid = [np.mgrid[u_min:u_max, v_min:v_max].reshape(2, -1).T for u_min, v_min, u_max, v_max in zip(u_min, v_min, u_max, v_max)]
        grid = torch.Tensor(np.stack(grid))
        # mask per triangle
        mask_bot = np.tri(u_max[0] - u_min[0], v_max[0] - v_min[0]).reshape(-1)  # N
        mask_top = 1 - np.tri(u_max[0] - u_min[0], v_max[0] - v_min[0]).reshape(-1)
        # mask_top = 1 - np.tri(u_max - u_min, v_max - v_min, -1).reshape(-1)
        # points per triangle
        pts_bot = grid[:,torch.Tensor(mask_bot).bool(),:]
        pts_top = grid[:,torch.Tensor(mask_top).bool(),:]
        return pts_bot, pts_top  # (Nx2, Nx2)
    
    @staticmethod
    def sample_pts_2d_grid(vt):
        # pixels inside quad
        u_min, v_min, u_max, v_max = vt[0, 0], vt[0, 1], vt[2, 0], vt[2, 1]
        grid = np.mgrid[u_min:u_max, v_min:v_max].reshape(2, -1).T
        return grid

    @staticmethod
    def sample_triangle(v, vt, triangle_f, pts_2d):
        # sample points inside the triangles
        tri_2d = triangle_f(vt)
        tri_3d = triangle_f(v)
        pts_3d = Geometry.triangle_2d_to_3d(tri_2d, tri_3d, pts_2d)
        return pts_3d
    
    @staticmethod
    def sample_quad(v, vt, quad_f, pts_2d):
        # sample points inside the quads
        quad_3d = quad_f(v)
        pts_3d = Geometry.quad_2d_to_3d(quad_3d, pts_2d, vt, v)
        return pts_3d
    

