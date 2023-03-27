import os
import time
import wandb
from os import makedirs
from os.path import join
from tqdm import tqdm, trange
import torch.nn.functional as F

from .utils import *
from .model import render
from .dataset.data import load_test_dataset, load_train_dataset


def train(args, start, global_step,  optimizer, fn, fn_test, hard_rays, mesh):
    dataset_test = load_test_dataset(args)
    n_iters = args.n_iters
    batch_sz = args.batch_size
    hard_pool_full = False
    dc_rate = 0.1
    dc_steps = args.lrate_decay * 1000
    file = 0

    for i in trange(start, n_iters, desc='Iter', colour="cyan"):
        if file == 20:
            # Restart the file counter
            file = 0
        dataset_train = load_train_dataset(args.basedir, args.num_files, file)
        for k in trange(0, dataset_train.shape[0], batch_sz, desc='Data'):

            # Compute learning rate depending on global step
            new_lrate = compute_lr(args, global_step, dc_rate, dc_steps)

            # Asign new learning rate to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            # Get chunks from data
            pts, dir, rgb_gt, faceid = chunks(dataset_train, k, batch_sz)

            # Find hard examples
            if args.hard_ratio:
                n_hard_in, n_hard_out = hard_rays_fn(args, batch_sz)

            # if buffer of hard examples full, then concatenate current and
            # hard examples
            if hard_pool_full:
                rand_ix_out, pts, dir, rgb_gt, faceid = \
                    hard_pool_full_fn(n_hard_out, hard_rays,
                                      pts, dir, rgb_gt, faceid)

            # Forward the model
            output = fn(pts, dir)

            # Zero out the optimizer to not accumulate gradients
            optimizer.zero_grad()

            # Compute the loss
            loss = F.mse_loss(output, rgb_gt[..., :3])

            if args.with_wandb:
                for o in optimizer.param_groups:
                    wblog = {"Iter": i, "Data": k, "Loss": loss, 'lr': o['lr']}
                    wandb.log(wblog)

            # Backward process
            loss.backward()
            optimizer.step()

            # Collect hard examples
            if args.hard_ratio:
                hard_rays_ =\
                    collect_hard_rays_fn(batch_sz, output, n_hard_in,
                                         pts, dir, rgb_gt, faceid)

            if hard_pool_full:
                hard_rays[rand_ix_out[:n_hard_in]] = hard_rays_  # replace
            else:
                hard_rays = torch.cat([hard_rays, hard_rays_], dim=0)  # append
                if hard_rays.shape[0] >= batch_sz * args.hard_mul:
                    hard_pool_full = True

            # Print PSNR and test loss
            if global_step % args.i_print == 0 and global_step > 0:
                tloss, psnr = render(dataset_test, mesh,
                                     fn_test, path=None, metrics=False)
                datap = k * 100 / dataset_train.shape[0]
                tqdmlog = f'Iter: {i} Data: {datap:.2f}% PLoss: {tloss:.5f} PSNR: {psnr:.2f}'
                tqdm.write(tqdmlog)

                if args.with_wandb:  # Log in WandB
                    wb_log = {"Iter": i, "Data": k,
                              "Ploss": tloss, "PSNR": psnr}
                    wandb.log(wb_log)

            # Render test images
            if global_step % args.i_testset == 0 and global_step > 0:
                name = f'testset_{global_step:012d}'
                path = join(args.basedir, 'testset', name)
                makedirs(path, exist_ok=True)
                render(dataset_test, mesh, fn_test, path=None, metrics=False)
                prYellow(f'Done rendering {path}')

            # Save weights of the model
            if global_step % args.i_weights == 0 and global_step > 0:
                name = f'{global_step:012d}.tar'
                path = join(args.basedir, args.expname, name)
                makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({'global_step': global_step,
                            'num_iter_per_batch': dataset_train.shape[0] // batch_sz,
                            'network_fn_state_dict': fn.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path)
                prYellow(f'Saved checkpoints at {path}')

            # Increase global step by 1
            global_step += 1

        # Empty cache to avoid memory issues with mo
        del dataset_train
        file += args.num_files
        torch.cuda.empty_cache()

