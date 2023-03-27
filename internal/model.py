import os
from os.path import join
import torch
import imageio
from tqdm import tqdm, trange
import torch.nn.functional as F

from .factorizedNeLF import *
from .utils import *


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    outputs_flat = batchify(fn, netchunk)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args):
    """Instantiate Factorized NeLF's model.
    """
    model = FactorizedNeLF(args, 3, 3, 3, 1).to(device)

    grad_vars = list(model.parameters())
   
    def network_query_fn(inputs, network_fn): return run_network(inputs, network_fn,
                                                    netchunk=args.netchunk)
    

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname
    if not os.path.exists(os.path.join(basedir, expname)):
        os.makedirs(os.path.join(basedir, expname))

    ##########################

    # Load checkpoints
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(
        os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        
        if args.load_ckpt < 0:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = os.path.join(basedir, expname) + f'/{str(args.load_ckpt).zfill(12)}.tar'
        
        prCyan(f'Reloading from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        
        model.load_state_dict(ckpt['network_fn_state_dict'])
        
        if 'num_iter_per_batch' in ckpt:
            num_iter_per_batch = ckpt['num_iter_per_batch']
            start = (start, num_iter_per_batch)
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'network_fn': model,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def render(dataset, mesh, fn, path=None, metrics=False):
    
    if path is None:
        psnr, total_loss = [], []
        name_bar = 'PSNR'
        color_bar = "magenta"
        if metrics:
            ssim_m, lpips_m = [], []
    else:
        name_bar = 'Test'
        color_bar = "white"
        

    for l in trange(0, len(dataset), desc=name_bar, colour=color_bar):
        H, W = dataset[l].H, dataset[l].W
        rays = dataset[l].get_all()
        gt_img = torch.Tensor(dataset[l].get_img()).to(device)
        rays_o, rays_d = rays['ray_o'], rays['ray_d']
        pts, idx_ray, _, _  = mesh.intersected_points(rays_o, rays_d)
        
        
        with torch.no_grad():
            pts = torch.Tensor(pts).to(device)
            rays_d = rays_d[idx_ray].to(device)
            rgb_total_train = fn(pts, rays_d)
           
        mask = torch.ones((H, W, 3)).reshape((-1, 3))
        mask[idx_ray] = rgb_total_train
        predicted_img = mask.reshape((H, W, 3))

        if path is None:
            loss_print = F.mse_loss(predicted_img, gt_img)
            total_loss.append(loss_print)
            psnr.append(mse2psnr(loss_print))
            if metrics:
                with torch.no_grad():
                    ssim_m.append(ssim_fn(predicted_img, gt_img))
                    lpips_m.append(lpips_fn(predicted_img, gt_img))

        else:
            rgb8 = to8b(predicted_img.cpu().numpy())
            filename = join(path, f'{l:03d}.png')
            imageio.imwrite(filename, rgb8)
       
    if path is None:
        total_loss = torch.stack(total_loss).mean().item()
        psnr = torch.stack(psnr).mean().item()
        if metrics:
            ssim_m = torch.stack(ssim_m).mean().item()
            lpips_m = torch.stack(lpips_m).mean().item()
            return psnr, ssim_m, lpips_m
        return total_loss, psnr

    