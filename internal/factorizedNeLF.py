import torch.nn as nn
import torch
from .encoding import MultiResHashGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEmbedder():

    def __init__(self, L, include_input=True):
        self.weights = 2**torch.linspace(0, L - 1, steps=L).to(device)  # [L]
        self.include_input = include_input
        self.embed_dim = 2 * L + 1 if include_input else 2 * L

    def __call__(self, x):
        y = x[
            ...,
            None] * self.weights  # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
        y = torch.cat([torch.sin(y), torch.cos(y)],
                      dim=-1)  # [n_ray, dim_pts, 2L]
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)],
                          dim=-1)  # [n_ray, dim_pts, 2L+1]
        return y.reshape(y.shape[0],
                         -1)  # [n_ray, dim_pts*(2L+1)], example: 48*21=1008

    def embed(self, x):
        ''''for CNN-style. Keep this for back-compatibility, please use embed_cnnstyle'''
        y = x[..., :, None] * self.weights
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)
        return y  # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]

    def embed_cnnstyle(self, x):
        y = x[..., :, None] * self.weights
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)
        return y  # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]


class ResMLP(nn.Module):

    def __init__(self,
                 width,
                 inact=nn.ReLU(True),
                 outact=None,
                 res_scale=1,
                 n_learnable=2):
        '''inact is the activation func within block. outact is the activation func right before output'''
        super(ResMLP, self).__init__()
        m = [nn.Linear(width, width)]
        for _ in range(n_learnable - 1):
            if inact is not None:
                m += [inact]
            m += [nn.Linear(width, width)]
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.outact = outact

    def forward(self, x):

        x = self.body(x).mul(self.res_scale) + x
        if self.outact is not None:
            x = self.outact(x)
        return x


def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'lrelu':
        func = nn.LeakyReLU(inplace=True)
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func


class R2L(nn.Module):
    '''Code from R2L: Distilling Neural Radiance Field to Neural Light 
                Field for Efficient Novel View Synthesis
       https://arxiv.org/abs/2203.17261'''

    def __init__(self, args, input_dim, output_dim):
        super(R2L, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]

        # the main non-linear activation func
        act = get_activation(args.act)

        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), act])

        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(Ws[i - 1], Ws[i]), act]

        # >>> new implementation of the body. Will replace the above

        if hasattr(args, 'trial'):
            inact = get_activation(args.trial.inact)
            outact = get_activation(args.trial.outact)
            if args.trial.body_arch in ['resmlp']:
                n_block = (
                    D - 2
                ) // 2  # 2 layers in a ResMLP, deprecated since there can be >2 layers in a block, use --trial.n_block
                if args.trial.n_block > 0:
                    n_block = args.trial.n_block
                body = [
                    ResMLP(W,
                           inact=inact,
                           outact=outact,
                           res_scale=args.trial.res_scale,
                           n_learnable=args.trial.n_learnable)
                    for _ in range(n_block)
                ]
            elif args.trial.body_arch in ['mlp']:
                body = []
                for i in range(1, D - 1):
                    body += [nn.Linear(Ws[i - 1], Ws[i]), act]
        # <<<

        self.body = nn.Sequential(*body)

        # tail
        self.output_dim = output_dim
        self.components = args.components
        self.tail = nn.Linear(
            Ws[D - 2], args.components * self.output_dim)

    def forward(self, x):  # x: embedded position coordinates
        if x.shape[-1] != self.input_dim:  # [N, C, H, W]
            x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        uvwb = self.tail(x).reshape(-1, self.components, self.output_dim)
        return uvwb


class FactorizedNeLF(nn.Module):
    '''Based on NeRF_v3, move positional embedding out'''

    def __init__(self, args, input_dim_pts, output_dim_pts,
                 input_dim_dir, output_dim_dir):

        super(FactorizedNeLF, self).__init__()
        self.b_max = args.b_max
        self.b_min = args.b_min

        # Hash Enconding for position-dependent network L_pos
        self.hash_enc = MultiResHashGrid(dim=input_dim_pts,
                                         n_levels=args.n_levels,
                                         n_features_per_level=args.n_features_per_level,
                                         log2_hashmap_size=15,
                                         base_resolution=16,
                                         finest_resolution=512,)  # 3D data

        # Position-dependent network L_pos
        self.model_pts = R2L(args, self.hash_enc.output_dim, output_dim_pts)

        # Positional Enconding for direction-dependent network L_dir
        self.pe_dir = PositionalEmbedder(L=args.multires_dir)

        # Direction-dependent network L_dir
        args.netdepth, args.netwidth = args.netdepth_d, args.netwidth_d
        self.model_dir = R2L(args, input_dim_dir *
                             self.pe_dir.embed_dim, output_dim_dir)

    def forward(self, pts, dir, features=False):
        # Getting Beta
        dir = self.pe_dir(dir)
        beta = torch.clamp(self.model_dir(dir), min=-10, max=10)
        # Scaling scene from 0 to 1
        pts = (pts-self.b_min)/(self.b_max-self.b_min)
        # Getting UVW
        pts = self.hash_enc(pts)
        uvw = torch.clamp(self.model_pts(pts), min=-10, max=10)
        # Matrix multiplication
        rgb = (beta * uvw).sum(dim=1)
        if features:
            return uvw, beta
        return torch.sigmoid(rgb)
