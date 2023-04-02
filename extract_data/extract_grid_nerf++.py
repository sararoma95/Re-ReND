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
from ddp_model import depth2pts_outside

logger = logging.getLogger(__package__)

def get_bounds(ray_o, ray_d, N_samples, min_depth):
    fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
    fg_near_depth = min_depth  # [..., ]
    step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
    fg_z_vals = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)
    dots_sh = list(ray_d.shape[:-1])

    fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

    b_max = fg_pts.amax(dim=(0, 1))
    b_min = fg_pts.amin(dim=(0, 1))
    return b_max, b_min

def all_bounds(ray_samplers):
    # Getting bounds of the fg scene
    b_max_rays, b_min_rays = [], []
    N_samples = 3
    for idx in range(len(ray_samplers)):
        ray_o = ray_samplers[idx].get_all()['ray_o']
        ray_d = ray_samplers[idx].get_all()['ray_d']
        min_depth = ray_samplers[idx].get_all()['min_depth']
        b_max, b_min = get_bounds(ray_o, ray_d, N_samples, min_depth)
        b_max_rays.append(b_max)
        b_min_rays.append(b_min)
    return b_max_rays, b_min_rays 

def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    # setup(rank, args.world_size)
    ###### set up logger
    out_dir = args.basedir #os.path.join(args.basedir, args.expname)
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    bound_min = np.array([-1.0000001,  -0.32, -1.0000001])
    bound_max = np.array([1.,        0.2, 1.   ])
    print('bound_min', bound_min)
    print('bound_max', bound_max)

    _, models = create_nerf(rank, args)
    resolution  = 512

    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    m=1
    net = models['net_{}'.format(m)]
    # pdb.set_trace()
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
                fg_pts = pts.cuda()
                ray_d = torch.zeros_like(pts).cuda()
                ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
                fg_viewdirs = ray_d / (ray_d_norm + 1e-20)     # [..., 3]

                input = torch.cat((net.nerf_net.fg_embedder_position(fg_pts),
                           net.nerf_net.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
                fg_raw = net.nerf_net.fg_net(input)
                val = fg_raw['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys),
                    zi * N:zi * N + len(zs)] = val 
    # print(np.histogram(u))      
    print(f"Saved voxel-grid at {out_dir}")
    np.save(f'{out_dir}/{args.expname}.npy', u)
    


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    ddp_test_nerf(0, args)


if __name__ == '__main__':
    setup_logger()
    test()