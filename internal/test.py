import imageio
from tqdm import trange
import torch.nn.functional as F

from .utils import *
from .mesh import Geometry
from .textures import Texture
from sklearn.neighbors import NearestNeighbors


def pts3d2ij(texture, face_id, vertices, pts):
    vertices = torch.Tensor(np.array(vertices))
    vt = texture._get_face_txt_coords_tri_bz(face_id, offset=0.5)
    pts_2d = Geometry.triangle_3d_to_2d_bz(vt, vertices, pts).long()
    pts_2d[:, 0, 0] = torch.clip(
        pts_2d[:, 0, 0], min=0, max=texture._w * texture._quad_size - 1)
    pts_2d[:, 1, 0] = torch.clip(
        pts_2d[:, 1, 0], min=0, max=texture._h * texture._quad_size - 1)
    ij = texture._uv_to_ij(pts_2d[..., 0].cpu().numpy())
    return torch.Tensor(ij)


def compute_uvw(args, texture, face_id, vertices, pts, min, max):
    uvw = read_png(args, args.basedir, min, max, ['u', 'v', 'w'])
    uvw = torch.Tensor(uvw)
    ij = pts3d2ij(texture, face_id, vertices, torch.Tensor(pts))
    return uvw[ij[:, 0].long(), ij[:, 1].long()]


def compute_beta(args, dir_ray, min, max):
    beta = read_png(args, args.basedir, min, max,['b'])
    beta = beta.reshape(-1, args.components)
    xyz = grid_direction(args.num_sample_elev, args.num_sample_azim)
    nbrs = NearestNeighbors(n_neighbors=1,
                            algorithm='ball_tree').fit(xyz.cpu().numpy())
    _, indices = nbrs.kneighbors(dir_ray.cpu().numpy())
    return beta[indices[:, 0]]


def test_quantization(args, dataset, mesh, testsavedir):
    # Get mesh
    faces =  mesh.mesh.faces

    # create texture
    num_tri = faces.shape[0]
    texture = Texture(num_tri, args.tri_size, args.components, uvw=3)

    # get min and max
    min_uvw, max_uvw, min_b, max_b = read_minmax(args)

    # Loop fro each test image
    psnr_m, ssim_m, lpips_m = [], [], []
    for l in trange(0, len(dataset)):
        H, W = dataset[l].H, dataset[l].W
        rays = dataset[l].get_all()
        gt_img = torch.Tensor(dataset[l].get_img()).to(device)
        rays_o, rays_d = rays['ray_o'], rays['ray_d']

        # Get vertices and face id from poses and intrinsics
        pts, idx_ray, vtx, faceid  = mesh.intersected_points(rays_o, rays_d)

        # Get quantized uvw and beta
        uvw = compute_uvw(args, texture, faceid,
                          vtx, pts, min_uvw, max_uvw)
        dir_ray = rays_d[idx_ray]
        beta = compute_beta(args, dir_ray, min_b, max_b)

        # Dot product and sigmoid
        rgb = (torch.Tensor(beta[..., None]) * uvw).sum(axis=1)
        rgb = torch.sigmoid(rgb)

        # RGB predicted image
        mask = torch.ones((H, W, 3)).reshape((-1, 3))
        mask[idx_ray] = rgb
        img = mask.reshape((H, W, 3))

        # Save image
        rgb8 = to8b(img.cpu().numpy())
        filename = join(testsavedir, f'{l:03d}_q_{args.tri_size}.png')
        imageio.imwrite(filename, rgb8)

        # Compute metrics
        loss_print = F.mse_loss(img, gt_img)
        psnr_m.append(mse2psnr(loss_print))
        with torch.no_grad():
            ssim_m.append(ssim_fn(img, gt_img))
            lpips_m.append(lpips_fn(img, gt_img))
    
    psnr_m = torch.stack(psnr_m).mean().item() 
    ssim_m = torch.stack(ssim_m).mean().item() 
    lpips_m = torch.stack(lpips_m).mean().item()
    return psnr_m, ssim_m, lpips_m
