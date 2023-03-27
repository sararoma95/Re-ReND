import configargparse


def config_parser():
    parser = configargparse.ArgParser()
    # Experiment path
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/',
                        help='input data directory')
    # parser.add_argument("--folder", type=str, default='')
    parser.add_argument("--name_exp", type=str, default='')

    # WandB
    parser.add_argument("--with_wandb", action='store_true')
    parser.add_argument("--project_name", type=str, default='Re-ReND')

    # Create the mesh
    parser.add_argument('--create_mesh', action='store_true')
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_comp", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--level_set", type=float, default=0.)
    parser.add_argument('--b_min', nargs='+', help='config file path')
    parser.add_argument('--b_max', nargs='+', help='config file path')

    # Pseudo GT from origins to intersected points
    parser.add_argument("--divide_data", action='store_true')

    # Model parameters
    parser.add_argument("--netdepth", type=int, default=88,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_d", type=int, default=88,
                        help='layers in network dir')
    parser.add_argument("--netwidth_d", type=int, default=256,
                        help='channels per layer dir')
    parser.add_argument("--components", type=int, default=32,
                        help='components of the basis')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument("--multires_dir", type=int, default=0.,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument('--layerwise_netwidths', type=str, default='')
    parser.add_argument('--act',
                        type=str,
                        default='relu',
                        choices=['relu', 'lrelu'],
                        help='main activation func in a network')
    # Try related features
    parser.add_argument('--trial.ON', action='store_true')
    parser.add_argument('--trial.body_arch',
                        type=str,
                        default='mlp',
                        choices=['mlp', 'resmlp'])
    parser.add_argument('--trial.res_scale', type=float, default=1.)
    parser.add_argument('--trial.n_learnable',
                        type=int,
                        default=2,
                        help='num of learnable layers')
    parser.add_argument('--trial.inact',
                        default='relu',
                        choices=['none', 'relu', 'lrelu'],
                        help='the within activation func in a block')
    parser.add_argument('--trial.outact',
                        default='none',
                        choices=['none', 'relu', 'lrelu'],
                        help='the output activation func in a block')
    parser.add_argument('--trial.n_block',
                        type=int,
                        default=-1,
                        help='num of block in network body')
    parser.add_argument('--trial.near', type=float, default=-1)
    parser.add_argument('--trial.far', type=float, default=-1)
    parser.add_argument('--hard_ratio',
                        type=str,
                        default='',
                        help='hard rays ratio in a batch; seperated by comma')
    parser.add_argument('--hard_mul',
                        type=float,
                        default=1,
                        help='hard_mul * batch_size is the size of hard ray pool')
    parser.add_argument('--warmup_lr', type=str, default='')
    # Positional encoding
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location) NERF 2')
    parser.add_argument("--n_levels", type=int, default=16,
                        help='levels hash encoding')
    parser.add_argument("--n_features_per_level", type=int, default=2,
                        help='levn_features_per_levelels hash encoding')
    parser.add_argument("--n_levels_dir", type=int, default=10,
                        help='levels hash encoding')
    # checkpoint, batch size, learning rate
    parser.add_argument("--load_ckpt", type=int, default=-1,
                        help='load specific checkpoint')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--num_files", type=int, default=4,
                        help='how many files load to cpu memory')
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=25000,
                        help='frequency of testset saving')
    # Seed
    parser.add_argument("--seed", type=int, default=23)

    # Texture exportation
    parser.add_argument("--export_textures", action='store_true')
    parser.add_argument('--tri_size', type=int, default=6)
    parser.add_argument('--features_size', type=int, default=4)
    parser.add_argument("--num_sample_elev", type=int, default=1024)
    parser.add_argument("--num_sample_azim", type=int, default=1024)

    # Training
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--load_data_device", type=str, default='cpu')
    parser.add_argument("--batch_size", type=int, default=200000)

    # Testing
    parser.add_argument("--total_psnr", action='store_true')
    parser.add_argument("--quantized_psnr", action='store_true')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    return parser
