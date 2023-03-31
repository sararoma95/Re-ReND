import json
import wandb
import torch
import random
import numpy as np
from PIL import Image
from os.path import join

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

def save_img(img, path, name = "texture"):
    # store image in disk
    name = f'{name}.png'
    Image.fromarray(to8b(img)).save(join(path, name))

def save_min_max(min_uvw, max_uvw, min_b, max_b, out_path):
    path = join(*out_path.split('/')[:-1])
    minmax = {}
    minmax['min_u'] = min_uvw[..., 0].tolist()
    minmax['min_v'] = min_uvw[..., 1].tolist()
    minmax['min_w'] = min_uvw[..., 2].tolist()
    minmax['min_b'] = min_b.detach().cpu().numpy().tolist()
    minmax['max_u'] = max_uvw[..., 0].tolist()
    minmax['max_v'] = max_uvw[..., 1].tolist()
    minmax['max_w'] = max_uvw[..., 2].tolist()
    minmax['max_b'] = max_b.detach().cpu().numpy().tolist()
    with open(join(join(*out_path.split('/')[:-1]), 'minmax.json'), 'w') as write_file:
        json.dump(minmax, write_file, indent=4)
    prYellow(f'Saved point features at {path}')

def init_train(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def name_wandb(args):
    obj = args.expname.split('_')[-1]
    namejob = 'Re-ReND_{}'.format(obj)
    return namejob


def init_wandb(args, project_name):
    if args.with_wandb:
        namejob = name_wandb(args)
        wandb.init(project=project_name, name=namejob, mode='online')
        wandb.config.update(args)
    else:
        wandb.init(project="Re-ReND", name=args.expname, mode='disabled')


def folder_path(args, start):
    if args.render_only:
        name = f'renderonly_{start:06d}'
        path = join(args.basedir, args.expname, name)
        return path
    elif args.export_textures:
        name = join(args.basedir.split('/')[-3])
        name = f'{name}.obj'
        path = join(args.basedir, f'meshes_textures_{args.tri_size}', name)
    elif args.compute_metrics:
        name = f'quantized_{args.tri_size}'
        path = join(args.basedir, args.expname, name)
    return path


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
    return ssim(img[None, ...].permute(0, 3, 1, 2).cpu(), gt[None, ...].permute(0, 3, 1, 2).cpu())


def lpips_fn(img, gt):
    return lpips(img[None, ...].permute(0, 3, 1, 2).cpu(), gt[None, ...].permute(0, 3, 1, 2).cpu())


def ssim_fn_bz(img, gt):
    return ssim(img.permute(0, 3, 1, 2).cpu(), gt.permute(0, 3, 1, 2).cpu())


def lpips_fn_bz(img, gt):
    return lpips(img.permute(0, 3, 1, 2).cpu(), gt.permute(0, 3, 1, 2).cpu())


def spherical_to_cartesian(ear):
    elev, azim, r = ear[..., 0], ear[..., 1], ear[..., 2]
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.cos(elev) * torch.sin(azim)
    z = r * torch.sin(elev)
    return torch.stack([x, y, z], dim=-1)


def grid_direction(num_sample_elev, num_sample_azim):
    # Creating uniform sampling in spherical coords.
    elev = torch.linspace(0, 360, num_sample_elev)
    azim = torch.linspace(0, 360, num_sample_azim)
    xy, yx = torch.meshgrid(elev, azim)

    # From degrees to radians
    xy = torch.deg2rad(xy).reshape(-1)
    yx = torch.deg2rad(yx).reshape(-1)

    # From Spherical Coords. to Cartesian Coords.
    ear = torch.stack((xy, yx, torch.ones_like(yx)), dim=-1)
    # .reshape(num_sample_elev, num_sample_azim,3)
    return spherical_to_cartesian(ear)

def read_png(args, path, min, max, list_feat):
    uvw = []
    for i, ft in enumerate(list_feat):
        feat = Image.open(
            join(path, f'meshes_textures_{args.tri_size}', f'feat_{ft}.png'))
        feat = reorganized(feat, col=4, row=args.components/4/4)
        uvw.append(unquantize(feat, min[:, i], max[:, i]))
    return np.stack(uvw, axis=-1)

def reorganized(feat, col, row):
    feat = np.split(np.array(feat), col, axis=1)
    feat = np.dstack(feat)
    feat = np.split(feat, row, axis=0)
    return np.dstack(feat)

def unquantize(feat, min, max):
    feat = feat / 255
    feat = feat * max
    feat = feat + min
    return feat

def read_minmax(args):
    prYellow('Reading max and min')
    path_dir = join(args.basedir, f'meshes_textures_{args.tri_size}',
                    'minmax.json')
    f = open(path_dir)
    minmax = json.load(f)
    min_u = np.array(minmax['min_u'])
    min_v = np.array(minmax['min_v'])
    min_w = np.array(minmax['min_w'])
    min_b = np.array(minmax['min_b'])[:, None]
    max_u = np.array(minmax['max_u'])
    max_v = np.array(minmax['max_v'])
    max_w = np.array(minmax['max_w'])
    max_b = np.array(minmax['max_b'])[:, None]
    min_uvw = np.stack([min_u, min_v, min_w], axis=-1)
    max_uvw = np.stack([max_u, max_v, max_w], axis=-1)
    return min_uvw, max_uvw, min_b, max_b

