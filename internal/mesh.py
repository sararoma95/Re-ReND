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


class Mesh:
    def __init__(self,
                 mesh_path,
                 **kwargs):
        args = SimpleNamespace(**kwargs)
        self.mesh_path = mesh_path

        # Create mesh
        if 'model' in kwargs:
            self._create_mesh(args.model, args.resolution, args.threshold,
                              args.level_set, args.num_comp, args.from_file)

        # Load the mesh
        self.mesh = self._load(mesh_path)


    def _load(self, path):
        return trimesh.load(path, process=False, maintain_order=True)

    def _save(self, mesh, meshlab=False):
        os.makedirs(join(*self.mesh_path.split('/')[:-1]), exist_ok=True)
        if meshlab:
            mesh.save_current_mesh(self.mesh_path)
        else:
            mesh.export(self.mesh_path)


    def intersected_points(self, rays_o, rays_d):
        rayMesh = RayMeshIntersector.RayMeshIntersector(self.mesh)
        pts, idx_ray, idx_tri = rayMesh.intersects_location(rays_o,
                                                            rays_d,
                                                            multiple_hits=False)
        # Corresponding vertices from faces intersected
        vtx = self.mesh.vertices[self.mesh.faces[idx_tri]]
        return pts, idx_ray, vtx, idx_tri

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, level_set, from_file):
        prYellow(f'threshold: {threshold}, level_set: {level_set}')

        # Extracting the estimated density of a point grid cube within scene bounds
        if from_file is None:
            assert 'no grid of densities'
        else:
            u = np.load(from_file)

        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

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

    def _create_mesh(self, model, resolution, threshold, level_set, num_comp, from_file):
        # Getting bounds of the scene

        b_min = model.b_min[0].split(',')
        b_min = [float(b_min[0]), float(b_min[1]), float(b_min[2])]
        b_max = model.b_max[0].split(',')
        b_max = [float(b_max[0]), float(b_max[1]), float(b_max[2])]
        prCyan(f'b_max: {b_max}, b_min: {b_min}')
    
        # Extracting vertices and faces from a point grid cube
        v, t = self.extract_geometry(b_min,
                                     b_max,
                                     resolution=resolution,
                                     threshold=threshold,
                                     level_set=level_set,
                                     from_file=from_file)

        # Creating a mesh with v and f
        mesh = trimesh.Trimesh(v, t)
        prYellow('Geometry extraction finish')

        # Post_processing: keep largest component
        if num_comp > 0:
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components])

            # Keep the Nth components of the mesh
            mesh = trimesh.util.concatenate(
                components[np.argpartition(areas, -num_comp)[-num_comp:]])
        # Save mesh
        self._save(mesh)

    @staticmethod
    def export_mesh_tri(save_path, vertices, faces, texture_vertices=None):
        # Exporting mesh with textures: features
        def vertex_to_str(v): return "v " + str(v)[1:-1] + "\n"
        def texture_to_str(vt): return "vt " + str(vt)[1:-1] + "\n"
        def face3_to_str(
            v, vt): return f"f {v[0]+1}/{vt[0]+1} {v[1]+1}/{vt[1]+1} {v[2]+1}/{vt[2]+1}\n"

        def faceN_to_str(v): return "f " + str(v)[1:-1] + "\n"
        with open(save_path, 'w') as file:
            # export material
            if texture_vertices is not None:
                file.write("mtllib ./test.obj.mtl\n\n")

            # export vertices
            for v in vertices:
                file.write(vertex_to_str(v))

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
        pts_2d_h = torch.cat([pts_2d.transpose(1, 2), torch.ones(
            (pts_2d.shape[0], 1, pts_2d.shape[1]))], axis=1)  # 3xN
        # splits triangles per components (3 points)
        q0, q1, q2 = tri_2d[:, 0], tri_2d[:, 1], tri_2d[:, 2]
        p0, p1, p2 = tri_3d[:, 0], tri_3d[:, 1], tri_3d[:, 2]
        # compute barycentric coordinates bar of pts_2d within tri_2d
        M_2d = torch.zeros_like(tri_3d)
        M_2d[:, 0, 0] = q1[:, 0] - q0[:, 0]
        M_2d[:, 0, 1] = q2[:, 0] - q0[:, 0]
        M_2d[:, 0, 2] = q0[:, 0]
        M_2d[:, 1, 0] = q1[:, 1] - q0[:, 1]
        M_2d[:, 1, 1] = q2[:, 1] - q0[:, 1]
        M_2d[:, 1, 2] = q0[:, 1]
        M_2d[:, 2, 2] = 1
        bar_coords = torch.bmm(torch.inverse(M_2d), pts_2d_h)  # 3x3 x 3xN[vt1]
        # convert bar_coords to cartessian using triangle tri_3d
        M_3d = torch.zeros_like(tri_3d)
        M_3d[:, 0, 0] = p1[:, 0] - p0[:, 0]
        M_3d[:, 0, 1] = p2[:, 0] - p0[:, 0]
        M_3d[:, 0, 2] = p0[:, 0]
        M_3d[:, 1, 0] = p1[:, 1] - p0[:, 1]
        M_3d[:, 1, 1] = p2[:, 1] - p0[:, 1]
        M_3d[:, 1, 2] = p0[:, 1]
        M_3d[:, 2, 0] = p1[:, 2] - p0[:, 2]
        M_3d[:, 2, 1] = p2[:, 2] - p0[:, 2]
        M_3d[:, 2, 2] = p0[:, 2]
        pts_3d = torch.bmm(M_3d, bar_coords)
        return pts_3d

    @staticmethod
    def triangle_3d_to_2d_bz(tri_2d, tri_3d, pts_3d):
        # # pts_2d to homogeneous coordinates
        # pts_2d_h = torch.cat([pts_2d.transpose(1,2), torch.ones((pts_2d.shape[0], 1, pts_2d.shape[1]))], axis=1)  # 3xN
        # splits triangles per components (3 points)
        q0, q1, q2 = tri_2d[:, 0], tri_2d[:, 1], tri_2d[:, 2]
        p0, p1, p2 = tri_3d[:, 0], tri_3d[:, 1], tri_3d[:, 2]
        # compute barycentric coordinates bar of pts_2d within tri_2d
        M_2d = torch.zeros_like(tri_3d)
        M_2d[:, 0, 0] = q1[:, 0] - q0[:, 0]
        M_2d[:, 0, 1] = q2[:, 0] - q0[:, 0]
        M_2d[:, 0, 2] = q0[:, 0]
        M_2d[:, 1, 0] = q1[:, 1] - q0[:, 1]
        M_2d[:, 1, 1] = q2[:, 1] - q0[:, 1]
        M_2d[:, 1, 2] = q0[:, 1]
        M_2d[:, 2, 2] = 1
        # convert bar_coords to cartessian using triangle tri_3d
        M_3d = torch.zeros_like(tri_3d)
        M_3d[:, 0, 0] = p1[:, 0] - p0[:, 0]
        M_3d[:, 0, 1] = p2[:, 0] - p0[:, 0]
        M_3d[:, 0, 2] = p0[:, 0]
        M_3d[:, 1, 0] = p1[:, 1] - p0[:, 1]
        M_3d[:, 1, 1] = p2[:, 1] - p0[:, 1]
        M_3d[:, 1, 2] = p0[:, 1]
        M_3d[:, 2, 0] = p1[:, 2] - p0[:, 2]
        M_3d[:, 2, 1] = p2[:, 2] - p0[:, 2]
        M_3d[:, 2, 2] = p0[:, 2]
        bar_coords = torch.bmm(torch.inverse(
            M_3d),  pts_3d[..., None])  # 3x3 x 3xN
        pts_2d = torch.bmm(M_2d, bar_coords)  # 3x3 x 3xN[vt1]
        return pts_2d

    @staticmethod
    def quad_2d_to_3d(quad_3d, pts_2d, vt, v):
        qs = int(np.sqrt(pts_2d.shape[0]))
        _quad_3d = np.stack(quad_3d)
        _quad_3d = _quad_3d[[0, 3, 1, 2]]
        quad_3d = torch.Tensor(_quad_3d.T).reshape(1, 3, 2, 2)
        pts_3d = torch.nn.functional.interpolate(
            quad_3d, size=(qs, qs), mode='bilinear', align_corners=True)
        min_x = pts_2d[:, 0].min()
        min_y = pts_2d[:, 1].min()
        return pts_3d[:, :, pts_2d[:, 0]-min_x, pts_2d[:, 1]-min_y].reshape(3, -1).t()

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

        # points per triangle
        pts_bot = grid[mask_bot == 1, :]
        pts_top = grid[mask_top == 1, :]
        return pts_bot, pts_top  # (Nx2, Nx2)

    @staticmethod
    def sample_pts_2d_tri_bz(vt, face_id):
        vt_np = vt.cpu().numpy()
        u_min, v_min, u_max, v_max = vt_np[:, 0,
                                           0], vt_np[:, 0, 1], vt_np[:, 1, 0], vt_np[:, 2, 1]
        # pixels inside quad
        grid = [np.mgrid[u_min:u_max, v_min:v_max].reshape(
            2, -1).T for u_min, v_min, u_max, v_max in zip(u_min, v_min, u_max, v_max)]
        grid = torch.Tensor(np.stack(grid))

        # mask per triangle
        mask_bot = np.tri(u_max[0] - u_min[0],
                          v_max[0] - v_min[0]).reshape(-1)  # N
        mask_top = 1 - np.tri(u_max[0] - u_min[0],
                              v_max[0] - v_min[0]).reshape(-1)
        
        # points per triangle
        pts_bot = grid[:, torch.Tensor(mask_bot).bool(), :]
        pts_top = grid[:, torch.Tensor(mask_top).bool(), :]
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
