import math
import torch
import numpy as np
from tqdm import tqdm
from os.path import join

from internal.utils import *
from internal.mesh import Mesh, Geometry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def direction_textures(model, path, num_sample_elev, num_sample_azim):
    chunk = 2000000

    # Get grid of directions in Cartesian Coords
    xyz = grid_direction(num_sample_elev, num_sample_azim)
    dir = (xyz).to(device)
    pts = torch.zeros_like(dir)

    # Query the Direction-dependent network L_dir for features in chunks
    dir_ft = []
    with torch.no_grad():
        for i in range(0, len(dir), chunk):
            dir_ft.append(
                model(pts[i:i+chunk], dir[i:i+chunk], features='True')[1])

    dir_ft = torch.cat(dir_ft, 0)
    out = dir_ft[..., 0].reshape(num_sample_elev, num_sample_azim, -1)

    # Quantize features
    out, min, max = quantize_dir(out)

    # Transforming image in a mosaic
    out = mosaic_dir(out)

    # Save features
    save_img(out.cpu().numpy(), path, 'feat_b')
    return min, max


class Texture:

    def __init__(self, num_tri, quad_size, features_size, uvw=3):
        self._quad_size = quad_size
        self._features_size = features_size
        self._h, self._w = self.size_tablet(num_tri // 2 + 1)
        self.uvw = uvw
        self._data = self._init_data()

    def size_tablet(self, num_quads):
        # returns texture size with max 2048 pixels in height (tablets and phones)
        max_height = 2048
        h = max_height // self._quad_size
        w = math.ceil(num_quads / h)
        return h, w

    @staticmethod
    def size_min_perimeter(num_quads):
        # returns texture size with minimum perimeter for the given area num_quads
        max_h = int(math.sqrt(num_quads))
        for h in range(max_h, 0, -1):
            if num_quads % h == 0:
                w = num_quads // h
                return h, w

    @staticmethod
    def size_square(num_quads):
        # returns texture size with square shape
        size = math.ceil(math.sqrt(num_quads))
        return size, size

    def _init_data(self):
        # initialize texture
        h = self._h * self._quad_size
        w = self._w * self._quad_size
        c = self._features_size
        return np.ones([h, w, c, self.uvw], np.float32)

    def _get_face_txt_coords_tri(self, face_id, offset=0.0):
        # get tri position
        tri_id = face_id % 2  # bottom or top triangle
        face_id = face_id // 2  # id in texture
        v_id = face_id // self._w  # row in texture
        u_id = face_id % self._w  # column in texture

        # get tri coordinates
        u_min = u_id * self._quad_size
        v_min = v_id * self._quad_size
        u_max = u_min + self._quad_size
        v_max = v_min + self._quad_size

        # the three vt coordinates
        if tri_id == 0:
            vts = np.array([[u_min + offset, v_min + offset],
                            [u_max - offset, v_min + offset],
                            [u_max - offset, v_max - offset]])
        else:
            # We multiply by 3 to remove the pixels that are in
            # the diagonal of the quad in the texture
            vts = np.array([[u_min + offset, v_min + offset*3],
                            [u_max - offset*3, v_max - offset],
                            [u_min + offset, v_max - offset],])

        return vts

    def _get_face_txt_coords_tri_bz(self, face_id, offset=0.0):
        # get tri position
        tri_id = face_id % 2
        face_id = face_id // 2
        v_id = face_id // self._w
        u_id = face_id % self._w

        # get tri coordinates
        u_min = u_id * self._quad_size
        v_min = v_id * self._quad_size
        u_max = u_min + self._quad_size
        v_max = v_min + self._quad_size
        # the three vt coordinates
        vts_bot = torch.Tensor(np.array([[u_min + offset, v_min + offset],
                                         [u_max - offset, v_min + offset],
                                         [u_max - offset, v_max - offset]]))
        # We multiply by 3 to remove the pixels that are in
        # the diagonal of the quad in the texture
        vts_top = torch.Tensor(np.array([[u_min + offset, v_min + offset*3],
                                         [u_max - offset*3, v_max - offset],
                                         [u_min + offset, v_max - offset],]))
        idx_bot = tri_id == 0
        idx_top = tri_id == 1
        vts = torch.zeros_like(vts_bot)
        vts[..., idx_bot] = vts_bot[..., idx_bot]
        vts[..., idx_top] = vts_top[..., idx_top]
        return vts.permute((2, 0, 1))

    def get_vertices_tri(self, faces, offset=0.0):
        num_faces = faces.shape[0]
        vertices = np.zeros((num_faces, 3, 2))

        # texture size in pixel space
        texture_size = np.array([self._w, self._h]) * self._quad_size

        # store for each face its 4 vts
        for face_id, face in enumerate(faces):

            # get face coordinates
            vt = self._get_face_txt_coords_tri(face_id, offset=offset)
            vertices[face_id, :, :] = vt / texture_size

        return vertices

    def _uv_to_ij(self, uv):
        # map uv coordinates to ij coordinates
        u, v = uv[:, 0], uv[:, 1]
        i = self._h * self._quad_size - v - 1
        j = u
        x = np.stack([i, j], 1)
        return x

    def _update(self, vt, new_data):
        # update texture data given positions and new data
        def to_ij_coords(x): return np.hsplit(self._uv_to_ij(x).astype(int), 2)
        ij = to_ij_coords(vt)
        self._data[ij[0], ij[1], :] = new_data[:, None, :]

    def fill_tri(self, model, vertices, faces):
        # fill all texture
        for face_id, face in enumerate(tqdm(faces)):
            # get points 2d in the triangle
            v = vertices[face]
            vt = self._get_face_txt_coords_tri(face_id)

            pts2d_bot, pts2d_top = Geometry.sample_pts_2d_tri(
                vt, face_id)  # without offset

            if face_id % 2 == 0:
                pts2d = pts2d_bot  # Triangle even in the bottom
            else:
                pts2d = pts2d_top  # Triangle odd in the top

            # map the texture 2d points to the mesh 3d points
            pts3d = Geometry.triangle_2d_to_3d(vt, v, pts2d)  # offset

            # Query the Position-dependent network L_pos for features
            pts3d_tri = torch.Tensor(pts3d).t().to(device)
            dir = torch.zeros_like(pts3d_tri)
            with torch.no_grad():
                pts_ft = model(pts3d_tri, dir, features=True)[0]

            # store in texture
            self._update(pts2d, pts_ft.cpu().numpy())  # [..., None]

    def quantize(self):
        min = np.amin(self._data, axis=(0, 1))
        self._data = self._data - min
        max = np.amax(self._data, axis=(0, 1))
        self._data = self._data / max
        return min, max

    def mosaic(self):
        # rearrange texture data to be in the right order
        # (h, w, components, uvw) -> (-1, w*4, 4, uvw)

        self._data = torch.Tensor(self._data).to(torch.device('cpu'))
        self._data = torch.stack(
            self._data.split(4, -2)).permute(1, 0, 2, 3, 4)
        self._data = self._data.reshape(
            self._h * self._quad_size, -1, 4, self.uvw)

        self._data = torch.cat(self._data.split(
            self._w * self._quad_size*4, dim=1)).cpu().numpy()

    def export(self, out_dir, name_pre):
        # export texture into multiple png images
        self.mosaic()
        feat_uvw = ['u', 'v', 'w']
        for uvw in range(self.uvw):
            img = self._data[..., uvw]
            save_img(img, out_dir, f'{name_pre}{feat_uvw[uvw]}')


def position_textures(model, mesh, out_path, tri_size, features_size):
    # Get mesh
    vertices, faces = mesh.mesh.vertices, mesh.mesh.faces

    # Create texture
    num_tri = faces.shape[0]
    texture = Texture(num_tri, tri_size, features_size, uvw=3)

    # Fill texture
    texture.fill_tri(model, vertices, faces)

    # Quantize texture
    min, max = texture.quantize()

    # Export texture
    out_dir = join(*out_path.split('/')[:-1])
    texture.export(out_dir, name_pre='feat_')

    # Export mesh
    texture_vertices = texture.get_vertices_tri(faces, offset=0.5)
    Mesh.export_mesh_tri(out_path, vertices, faces, texture_vertices)
    return min, max


def export_textures(args, fn_test, out_path, mesh):
    path = join(*out_path.split('/')[:-1])

    # Storing features of dir branch (beta)
    min_b, max_b = direction_textures(
        fn_test,
        path,
        args.num_sample_elev,
        args.num_sample_azim)
    prYellow(f'Saved direction features at {path}')

    # Generate textures
    min_uvw, max_uvw = position_textures(model=fn_test,
                                         mesh=mesh,
                                         out_path=out_path,
                                         tri_size=args.tri_size,
                                         features_size=args.components)

    save_min_max(min_uvw, max_uvw, min_b, max_b, out_path)
