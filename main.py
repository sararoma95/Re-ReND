import torch
from os.path import join
from types import SimpleNamespace

from internal.utils import *
from internal.config import config_parser
from internal.mesh import Mesh
from internal.model import create_nerf
from internal.train import train

if __name__ == "__main__":
    # Seeking for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = config_parser()
    args = parser.parse_args()

    name = f'lev_{args.level_set}_thr_{args.threshold}_{args.expname}.ply'
    args.mesh_path = join(args.datadir, 'meshes', name)
   
    args.folder = f'exp_lev_{args.level_set}_thr_{args.threshold}' 

    if args.train:
        prBlue('TRAINING Re-ReND')

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
        args.basedir = join(args.basedir + '_' + args.folder, name_wandb(args) + args.name_exp)
        
        # Loading mesh
        mesh = Mesh(args.mesh_path).mesh
        
        # Updating b_min and b_max
        args.b_max =  torch.Tensor(np.array(mesh.vertices.max(axis=0))).to(device)
        args.b_min = torch.Tensor(np.array(mesh.vertices.min(axis=0))).to(device)
           
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

        # if args.render_only:
        #     prBlue('RENDER ONLY')
        #     # Initializing wandb
        #     init_wandb(args, args, main_dict, args_args, args.project_name+'_test')
        #     # Creating folder to dump RGB information
        #     testsavedir = folder_path(args, nerf_1, args, global_step)
        #     os.makedirs(testsavedir, exist_ok=True)

        #     # if args.mesh_parts >0  and args.total_psnr:

        #     if args.total_psnr:
        #         path_bg  = join(*testsavedir.split('/')[:5], testsavedir.split('/')[5][:-2]+'bg', *testsavedir.split('/')[6:])
        #         psnr, ssmi, lpips = total_psnr_tandt(args, ps, testsavedir, path_bg)
        #         if args.with_wandb:
        #             wblog = {"PSNR": psnr, "SSMI": ssmi, "LPIPS": lpips, "global_step": global_step}
        #             wandb.log(wblog)
        #         exit()
        #     # Rendering video or test images
        #     psnr, ssmi, lpips = render_imgs(args, nerf_1, args, fn, pe, ps, path=None, video=False, metrics=True)
        #     prYellow(f'PSNR: {psnr}, SSMI: {ssmi}, LPIPS: {lpips}')
        #     if args.with_wandb:
        #         wblog = {"PSNR": psnr, "SSMI": ssmi, "LPIPS": lpips, "global_step": global_step}
        #         wandb.log(wblog)
        #     render_imgs(args=args, nerf_1=nerf_1, args=args,
        #                 fn=fn_test,
        #                 pe=pe, 
        #                 ps=ps,
        #                 path=testsavedir,
        #                 video=False if args.render_test else True)
        #     prYellow(f'Done Rendering: {testsavedir}')
        #     exit()

        # if args.exporting_features:
        #     prBlue('EXPORTING FEATURES-TEXTURE')
        #     if args.train_bg:
        #         mesh_in_path = args.mesh_path_bg
        #     else:
        #         mesh_in_path = args.mesh_path_fg

        #     # Creating folder to seek mesh and dump texture and MLP
        #     out_path = folder_path(args, nerf_1, args, global_step)
        #     os.makedirs(join(*out_path.split('/')[:-1]), exist_ok=True)

        #     # Expoting features: mlp and textures
        #     exporting_features(args, args, fn_test, out_path, mesh_in_path, pe, ps) 
        #     exit()
        # if args.quantized_psnr:
        #     init_wandb(args, args, main_dict, args_args, args.project_name+'_quanTest')
        #     prBlue('COMPUTE QUANTIZED PSNR')
        #     if args.train_bg:
        #         mesh_in_path = args.mesh_path_bg
        #     else:
        #         mesh_in_path = args.mesh_path_fg

        #     # Declare folder to save quantize images
        #     testsavedir = folder_path(args, nerf_1, args, global_step)
        #     os.makedirs(testsavedir, exist_ok=True)

        #     # Calculate metrics
        #     if nerf_1.dataset_type == 'blender':
        #         psnr, ssmi, lpips  = test_metrics_after_quantization(args, args, ps, args.components, args.quad_size, 
        #         mesh_in_path, testsavedir)
        #     elif nerf_1.dataset_type == 'tanks_and_temples':
        #         psnr, ssmi, lpips  = test_metrics_after_quantization_tandt(args, args, ps, args.components, args.quad_size, 
        #         mesh_in_path, testsavedir)
        #     prYellow(f'PSNR: {psnr}, SSMI: {ssmi}, LPIPS: {lpips}')
        #     if args.with_wandb:
        #         wblog = {"PSNR": psnr, "SSMI": ssmi, "LPIPS": lpips, "global_step": global_step}
        #         wandb.log(wblog)
        #     exit()
        # Initializing wandb
        init_wandb(args, args.project_name)

        
        # import pdb; pdb.set_trace()

        # Hard ratio weights array or a float
        if args.hard_ratio != '':
            if ',' not in args.hard_ratio:
                args.hard_ratio = float(args.hard_ratio)
            else:
                args.hard_ratio = [
                    float(x) for x in args.hard_ratio.split(',')
                ]
        if args.hard_ratio: hard_rays = torch.Tensor([])


        # Training loop
        train(args=args,      
            start=start,
            global_step=global_step,
            optimizer=optimizer,
            fn=fn,
            fn_test=fn_test,
            hard_rays=hard_rays,
            mesh=mesh)