import wandb
import torch
import random
import numpy as np

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prCyan(s):
    print("\033[96m\033[1m {}\033[00m".format(s))


def prYellow(s):
    print("\033[93m {}\033[00m".format(s))


def prBlue(s):
    print("\033[94m {}\033[00m".format(s))


def prRed(s):
    print("\033[91m {}\033[00m".format(s))


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def init_train(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def name_wandb(args):
    obj = args.expname.split('_')[-1]
    namejob = '{}_hash.{}.{}_comp.{}'.format(
        obj, args.n_levels, args.n_features_per_level, args.components)
    return namejob


def init_wandb(args, project_name):
    if args.with_wandb:
        namejob = name_wandb(args)
        wandb.init(project=project_name, name=namejob, mode='online')
        wandb.config.update(args)
    else:
        wandb.init(project="Re-ReND", name=args.expname, mode='disabled')

def hard_rays_fn(args, batch_size):
    if isinstance(args.hard_ratio, list):
        n_hard_in = int(
            args.hard_ratio[0] *
            batch_size)  # the number of hard samples into the hard pool
        n_hard_out = int(
            args.hard_ratio[1] *
            batch_size)  # the number of hard samples out of the hard pool
    else:
        n_hard_in = int(args.hard_ratio * batch_size)
        n_hard_out = n_hard_in
    n_hard_in = min(n_hard_in, n_hard_out)  # n_hard_in <= n_hard_out
    return n_hard_in, n_hard_out


def hard_pool_full_fn(n_hard_out, hard_rays, pts, dir, rgb_gt, faceid):
    rand_ix_out = np.random.permutation(hard_rays.shape[0])[:n_hard_out]
    picked_hard_rays = hard_rays[rand_ix_out]
    pts = torch.cat([pts, picked_hard_rays[:, :3]], dim=0)
    dir = torch.cat([dir, picked_hard_rays[:, 3:6]], dim=0)
    rgb_gt = torch.cat([rgb_gt, picked_hard_rays[:, 6:9]], dim=0)
    faceid = torch.cat([faceid, picked_hard_rays[:, 9:]], dim=0)
    return rand_ix_out, pts, dir, rgb_gt, faceid


def collect_hard_rays_fn(batch_size, output, n_hard_in, pts, dir, rgb_gt,
                         faceid):
    _, indices = torch.sort(
        torch.mean((output[:batch_size] - rgb_gt[:batch_size, :3])**2, dim=1))
    hard_indices = indices[-n_hard_in:]

    hard_rays_ = torch.cat([
        pts[hard_indices], dir[hard_indices], rgb_gt[hard_indices],
        faceid[hard_indices]
    ],
                           dim=-1)
    return hard_rays_


def compute_lr(args, global_step, decay_rate, decay_steps):
    if args.warmup_lr:  # @mst: example '0.0001,2000'
        start_lr, end_iter = [float(x) for x in args.warmup_lr.split(',')]
        if global_step < end_iter:  # increase lr until args.lrate
            new_lrate = (args.lrate -
                         start_lr) / end_iter * global_step + start_lr
        else:  # decrease lr as before
            new_lrate = args.lrate * (decay_rate**(
                (global_step - end_iter) / decay_steps))
    else:
        new_lrate = args.lrate * (decay_rate**(global_step / decay_steps))
    return new_lrate


def chunks(data, k, batch_size):
    pts = data[k:k + batch_size, :3].to(device)
    dir = data[k:k + batch_size, 3:6].to(device)
    gt = data[k:k + batch_size, 6:9].to(device)
    id = data[k:k + batch_size, 9:].to(device)
    return pts, dir, gt, id

def ssim_fn(img, gt):
    return ssim(img[None,...].permute(0,3,1,2).cpu(), gt[None,...].permute(0,3,1,2).cpu())

def lpips_fn(img, gt):
    return lpips(img[None,...].permute(0,3,1,2).cpu(), gt[None,...].permute(0,3,1,2).cpu())

def ssim_fn_bz(img, gt):
    return ssim(img.permute(0,3,1,2).cpu(), gt.permute(0,3,1,2).cpu())

def lpips_fn_bz(img, gt):
    return lpips(img.permute(0,3,1,2).cpu(), gt.permute(0,3,1,2).cpu())