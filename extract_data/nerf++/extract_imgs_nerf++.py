from collections import OrderedDict
import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf, intersect_sphere
import logging
import pdb
from nerf_sample_ray_split import get_rays_single_image
import trimesh.ray.ray_pyembree as RayMeshIntersector
import trimesh
from scipy.spatial.transform import Rotation as R
logger = logging.getLogger(__package__)

def prCyan(s):
    print("\033[96m\033[1m {}\033[00m".format(s))


def prYellow(s):
    print("\033[93m {}\033[00m".format(s))


def prBlue(s):
    print("\033[94m {}\033[00m".format(s))


def prRed(s):
    print("\033[91m {}\033[00m".format(s))

def verify_rays(ray_o, ray_d, min_depth, name='bien.ply'):
    N_samples = 10
    fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
    fg_near_depth = min_depth # [..., ]
    step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
    fg_z_vals = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                
    # ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
    # viewdirs = ray_d / ray_d_norm      # [..., 3]
    dots_sh = list(ray_d.shape[:-1])

    ######### render foreground
    N_samples = fg_z_vals.shape[-1]
    fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    # fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
    # pdb.set_trace()
    save_pc(fg_pts.reshape(-1,3).cpu().numpy(), torch.ones(fg_pts.reshape(-1,3).shape).cpu().numpy(), name)

def save_pc(PC, PC_color, filename):
    # Saving a point cloud with PC_color and format '.ply'
    from plyfile import PlyElement, PlyData
    PC = np.concatenate((PC, PC_color), axis=1)
    PC = [tuple(element) for element in PC]
    el = PlyElement.describe(
        np.array(PC,
                 dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                        ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([el]).write(filename)

def export_data(id, expname, gt, path):
    name = f'{path}/{expname}_{id}.pt'
    torch.save(torch.cat(gt), name)
    prYellow(f'Saved at {name} and size: {torch.cat(gt).shape}')

def get_random_pose_dist(ray_samplers, idx_list):
    c2w_mat = [ray_samplers[idx].c2w_mat for idx in idx_list]
    c2w_mat = np.stack(c2w_mat)

    # Translation random
    t_max = c2w_mat[:,:3,3].max(0)
    t_min = c2w_mat[:,:3,3].min(0)
    t = c2w_mat[:,:3,3].mean(0)
    noise_t = np.random.uniform(0, 1, len(t)) * (t_max - t_min)
    t +=  noise_t 

    # Rotation random 
    r = c2w_mat[0,:3,:3]
    r = R.from_matrix(r)
    r_euler = r.as_euler('xzy', degrees=False)
    noise_r = np.random.uniform(-1, 1, len(r_euler)) * np.deg2rad(np.array([5, 5, 10]))
    r *= R.from_euler('xzy', noise_r.tolist(), degrees=False)
    c2w_mat = np.concatenate((np.concatenate((np.array(r.as_matrix()),t[:,None]), axis=-1),
                                             np.array([0,0,0,1])[None,:]), axis=0)
    return  c2w_mat

def preparing_data(rays_o, rays_d, depth, H, W):
    data_rays = OrderedDict([
                ('ray_o', rays_o),
                ('ray_d', rays_d),
                ('depth', depth),
                ('min_depth', 1e-4 * np.ones_like(rays_d[..., 0])),
                ('img_path', None),
                ('H', H),
                ('W', W),

            ])
    for k in data_rays:
        if isinstance(data_rays[k], np.ndarray):
            data_rays[k] = torch.from_numpy(data_rays[k]).to(torch.float32)
    return data_rays

def get_random_pose(ray_samplers):
    # Select on random pose
    rand_idx = np.random.randint(low=0, high=len(ray_samplers) - 1, size=1)[0]
    idx_list = [rand_idx, rand_idx + 1] 

    # Get random c2w_mat
    c2w_mat = get_random_pose_dist(ray_samplers, idx_list)

    # Sample rays with the new pose
    rays_o, rays_d, depth = get_rays_single_image(ray_samplers[0].H, ray_samplers[0].W, 
                                                            ray_samplers[rand_idx].intrinsics, c2w_mat)
    return rays_o, rays_d, depth 

def intersected_points(rays_o, rays_d, mesh_path):
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)          
    rayMesh = RayMeshIntersector.RayMeshIntersector(mesh)
    loc, idx_ray, idx_tri = rayMesh.intersects_location(rays_o,
                                                        rays_d,
                                                        multiple_hits=False)
    return loc, idx_ray, idx_tri

def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -np.sum(ray_d * ray_o, axis=-1) / np.sum(ray_d * ray_d, axis=-1)
    p = ray_o + d1[...,0] * ray_d
    # consider the case where the ray does not intersect the sphere
    p_norm_sq = np.sum(p * p, axis=-1)
    return p_norm_sq
    

def ddp_test_nerf(rank, args):
    # Set up logger
    logger = logging.getLogger(__package__)
    setup_logger()
    np.random.seed(args.id)
      
    # Decide chunk size according to gpu memory
    args.N_rand = 1024 
    args.chunk_size = 8192 
    logger.info(f'setting batch size according to 24G gpu {args.N_rand}, {args.chunk_size}')

    # Create network and wrap in ddp; each process should do this
    _, models = create_nerf(rank, args)

    # Creating output dataset
    out_dir = os.path.join(os.path.dirname(args.basedir),'pdata' ) #+ '_' + args.folder
    logger.info(f'Saving at {out_dir}/{args.expname}_{args.id}.pt')
    os.makedirs(out_dir, exist_ok=True)
    # pdb.set_trace()
    
    # Load data and create ray samplers; each process should do this
    ray_samplers = load_data_split(args.datadir, args.scene, 'train', try_load_min_depth=args.load_min_depth)

    # Start Generating
    gt = []
    for i in range(500):
        logger.info(f'IMAGE: {i}')
        # Get random pose
        rays_o, rays_d, depth = get_random_pose(ray_samplers)

        # Confirm that is a correct pose otherwise descart it 
        p_norm_sq = intersect_sphere(rays_o, rays_d)
        if (p_norm_sq >= 1.).any():
            continue
        
        # Get data from the random pose
        data = preparing_data(rays_o, rays_d, depth, ray_samplers[0].H, ray_samplers[0].W)

        # Render image from the random pose
        time0 = time.time()
        ret = render_single_image(rank, args.world_size, models, data, args.chunk_size, pseudo=True)
        dt = time.time() - time0

        # Appending information 
        gt.append(torch.cat((
                            torch.Tensor(rays_o),
                            torch.Tensor(rays_d),
                            ret[-1]['rgb'].reshape(-1,3)),
                            dim=-1))

        logger.info('Rendered {} in {} seconds'.format(i, dt))
        torch.cuda.empty_cache()

    # Export pt with generated data
    export_data(args.id, args.expname, gt, out_dir)


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    if args.gcp:
        ddp_test_nerf(args.num_gcp, args)
    else:
        ddp_test_nerf(0, args)


if __name__ == '__main__':
    setup_logger()
    test()
