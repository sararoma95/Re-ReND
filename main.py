import os
import torch
from os.path import join
from types import SimpleNamespace

from internal.utils import *
from internal.mesh import Mesh
from internal.train import train
from internal.config import config_parser
from internal.test import test_quantization
from internal.textures import export_textures
from internal.model import create_nerf, render
from internal.dataset.data import load_test_dataset

if __name__ == "__main__":
    # Seeking for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = config_parser()
    args = parser.parse_args()

    name = f'lev_{args.level_set}_thr_{args.threshold}_{args.expname}.ply'
    args.mesh_path = join(args.datadir, 'meshes', name)

    args.folder = f'exp_lev_{args.level_set}_thr_{args.threshold}'

    # Initializing NeRF training
    init_train(args.seed)

    # Iterations start = 0
    start = 0

    # R2L setting
    trial_dict = {
        k.replace("trial.", ''): v
        for k, v in args.__dict__.items() if 'trial.' in k
    }
    args.trial = SimpleNamespace(**trial_dict)

    # Updating basedir
    args.basedir = join(args.basedir + '_' + args.folder,
                        name_wandb(args) + args.name_exp)
    if args.create_mesh:
        prBlue('CREATING MESH')

        # Creating mesh from NeRF 1
        Mesh(args.mesh_path,
            args=args,
            resolution=args.resolution,
            threshold=args.threshold,
            level_set=args.level_set,
            num_comp=args.num_comp,
            from_file=args.from_file)

        prYellow(f'Mesh exported to {args.mesh_path}')
        exit()

    # Loading mesh
    mesh = Mesh(args.mesh_path)

    if args.compute_metrics:
        prBlue('COMPUTING QUANTIZED METRICS')
        init_wandb(args, args.project_name+'_quanTest')

        # Declare folder to save quantize images
        testsavedir = folder_path(args, None)
        os.makedirs(testsavedir, exist_ok=True)

        # Calculate metrics
        dataset = load_test_dataset(args)
        psnr, ssmi, lpips = test_quantization(args, dataset, mesh, testsavedir)
        prYellow(f'PSNR: {psnr}, SSMI: {ssmi}, LPIPS: {lpips}')
        if args.with_wandb:
            wblog = {"PSNR": psnr, "SSMI": ssmi, "LPIPS": lpips}
            wandb.log(wblog)
        exit()
        
    # Updating b_min and b_max
    args.b_max = torch.Tensor(
        np.array(mesh.mesh.vertices.max(axis=0))).to(device)
    args.b_min = torch.Tensor(
        np.array(mesh.mesh.vertices.min(axis=0))).to(device)

    # Creating and loading model
    render_kwargs_train, render_kwargs_test, \
        start, grad_vars, optimizer = create_nerf(args=args)

    fn = render_kwargs_train['network_fn']
    fn_test = render_kwargs_test['network_fn']

    # Start from last iteration if there is a loaded model
    if isinstance(start, tuple):
        global_step = start[0]
        start = global_step // start[1] + 1
    else:
        global_step = start
        start = global_step + 1
    prYellow(f'Starting at {start}')

    if args.render_only:
        prBlue('COMPUTING CONTINUOS METRICS')
        # Initializing wandb
        init_wandb(args, args.project_name+'_test')

        # Creating folder to dump RGB information
        testsavedir = folder_path(args, global_step)
        os.makedirs(testsavedir, exist_ok=True)

        # Rendering test images and collecting metrics
        dataset = load_test_dataset(args)
        psnr, ssmi, lpips = render(
            dataset, mesh, fn_test, path=None, metrics=True)
        prYellow(f'PSNR: {psnr}, SSMI: {ssmi}, LPIPS: {lpips}')
        if args.with_wandb:
            wblog = {"PSNR": psnr, "SSMI": ssmi,
                     "LPIPS": lpips, "global_step": global_step}
            wandb.log(wblog)
        render(dataset, mesh, fn_test, path=testsavedir, metrics=False)
        prYellow(f'Done Rendering: {testsavedir}')
        exit()

    if args.export_textures:
        prBlue('EXPORTING FEATURES-TEXTURES')

        # Creating folder to seek mesh and dump texture and MLP
        out_path = folder_path(args, global_step)
        os.makedirs(join(*out_path.split('/')[:-1]), exist_ok=True)

        # Expoting features: mlp and textures
        export_textures(args, fn_test, out_path, mesh)
        exit()

    if args.train:
        prBlue('TRAINING Re-ReND')
        # Initializing wandb
        init_wandb(args, args.project_name)

        # Hard ratio weights array or a float
        if args.hard_ratio != '':
            if ',' not in args.hard_ratio:
                args.hard_ratio = float(args.hard_ratio)
            else:
                args.hard_ratio = [
                    float(x) for x in args.hard_ratio.split(',')
                ]
        if args.hard_ratio:
            hard_rays = torch.Tensor([]).to(device)

        # Training loop
        train(args=args,
              start=start,
              global_step=global_step,
              optimizer=optimizer,
              fn=fn,
              fn_test=fn_test,
              hard_rays=hard_rays,
              mesh=mesh)
