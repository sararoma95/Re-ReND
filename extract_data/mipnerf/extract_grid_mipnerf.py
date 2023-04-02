import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np

from internal import datasets
from internal import math
from internal import models
from internal import utils
from internal import vis
import torch
import pdb

from tqdm import tqdm
FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_bool(
    'eval_once', True,
    'If True, evaluate the model only once, otherwise keeping evaluating new'
    'checkpoints if any exist.')
flags.DEFINE_bool('save_output', True,
                  'If True, save predicted images to disk.')

RESOLUTION = 512
def extract_fields(bound_min, bound_max, near, far, resolution, model, config):
    '''Creating point grid cube to extract density
    Args:
      bound_min: the minimun bound that the scene can reach (e.g. [-1,-1,-1])
      bound_max: the maximun bound that the scene can reach (e.g. [1,1,1])
      resolution:  is the number of distinct points in each dimension (x,y,z) 
        that the point grid cube is compose.
      query_func: function used for passing queries to network_fn.
      fn: function. Model for predicting RGB and density at each point
        in space.
    Returns:
      u: Estimated density per each point of the point grid cube. 
    '''
    N = 256
    print(resolution)
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    raws = []
    
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                xx, yy, zz = np.array(xx), np.array(yy), np.array(zz)
                # print(xi, yi, zi)
                origins = np.concatenate([
                    xx.reshape(-1, 1),
                    yy.reshape(-1, 1),
                    zz.reshape(-1, 1)
                ],
                                axis=-1)
                # origins = np.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3))
                # pdb.set_trace()
                directions = np.zeros_like(origins)
                viewdirs = np.zeros_like(origins)
                radii = np.ones_like(origins[..., :1]) * 0.0005
                ones = np.ones_like(origins[..., :1])
                rays = utils.Rays(
                        origins=origins,
                        directions=directions,
                        viewdirs=viewdirs,
                        radii=radii,
                        lossmult=ones,
                        near=ones * near,
                        far=ones * far)
                # print('input:', origins.shape)
                
                for i in tqdm(range(0, rays[0].shape[0], config.chunk)):
                    # print('chunk', i)
                    # chunk_rays = namedtuple_map(lambda r: r[i:i + config.chunk].astype(np.float64), rays)
                    chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[i:i + config.chunk]),
                                      rays)
                    # pdb.set_trace()
                    raw = model(None, chunk_rays)[-1][-1].squeeze()
                    raws.append(np.mean(raw, axis=1))

                # pdb.set_trace()
                sigma = np.concatenate(raws, axis=0)
                sigma = np.maximum((sigma), 0)
                # print('output:', sigma.shape)
                val =  sigma.reshape(len(xs), len(ys),len(zs))
                raws = []
                u[xi * N:xi * N + len(xs), 
                  yi * N:yi * N + len(ys),
                  zi * N:zi * N + len(zs)] = val
    return u

def extract_mesh(unused_argv):
    config = utils.load_config()

    dataset = datasets.get_dataset('generate', FLAGS.data_dir, config)
    #   import pdb; pdb.set_trace()
    print(FLAGS.seed)
    model, init_variables = models.construct_mipnerf(
        random.PRNGKey(FLAGS.seed), dataset.peek())
    optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates 'speckle' artifacts.
    def render_eval_fn(variables, _, rays):
        return jax.lax.all_gather(
            model.apply(
                variables,
                random.PRNGKey(0),  # Unused.
                rays,
                randomized=False,
                white_bkgd=config.white_bkgd),
            axis_name='batch')

    # pmap over only the data input.
    render_eval_pfn = jax.pmap(
        render_eval_fn,
        in_axes=(None, None, 0),
        donate_argnums=2,
        axis_name='batch',
    )


    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.optimizer.state.step)


    near = 2.0
    far = 6.0
    obj = FLAGS.data_dir.split('/')[-1]
    if obj == 'hotdog':
        print('hotdog scale')
        xmin, xmax = [-1.5, 1.3]
        ymin, ymax = [-1.5, 1.2]
        zmin, zmax = [-1.2, 1.2]
    elif obj == 'mic':
        print('mic scale')
        xmin, xmax = [-1.5, 1.2]
        ymin, ymax = [-1.2, 1.2]
        zmin, zmax = [-1.2, 1.2]
    elif obj == 'ship':
        print('ship scale')
        xmin, xmax = [-1.5, 1.5]
        ymin, ymax = [-1.5, 1.5]
        zmin, zmax = [-1.2, 1.2]
    else:
        print('rest scale')
        xmin, xmax = [-1.2, 1.2]
        ymin, ymax = [-1.2, 1.2]
        zmin, zmax = [-1.2, 1.2]
    bound_min = np.array([xmin, ymin, zmin])
    bound_max = np.array([xmax, ymax, zmax])
    fn = functools.partial(render_eval_pfn, state.optimizer.target)
    u = extract_fields(bound_min, bound_max, near, far, RESOLUTION, fn, FLAGS)
    np.save(f"{FLAGS.data_dir}/{obj}.npy", u)
    print(f"Saved at {FLAGS.data_dir}/{obj}.npy")

if __name__ == '__main__':
  app.run(extract_mesh)