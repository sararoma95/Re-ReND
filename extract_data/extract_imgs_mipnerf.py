# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for mip-NeRF."""
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
import os
FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_bool(
    'eval_once', True,
    'If True, evaluate the model only once, otherwise keeping evaluating new'
    'checkpoints if any exist.')
flags.DEFINE_bool('save_output', True,
                  'If True, save predicted images to disk.')

def save_pc(PC, PC_color, filename):
    # Saving a point cloud with PC_color and format '.ply'
    from plyfile import PlyElement, PlyData
    PC = np.concatenate((PC, PC_color), axis=1)
    PC = [tuple(element) for element in PC]
    el = PlyElement.describe(
        np.array(PC,
                 dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                        ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([el]).write(filename)

def prYellow(s):
    print("\033[93m {}\033[00m".format(s))

def export_data(id, expname, gt, path):
    name = f'{path}/pdata/'
    os.makedirs(name, exist_ok=True)
    name = f'{name}/blender_paper_{expname}_{id}.pt'
    torch.save(gt, name)
    prYellow(f'Saved at {name} and size: {gt.shape}')

def main(unused_argv):
  config = utils.load_config()

  dataset = datasets.get_dataset('generate', FLAGS.data_dir, config)
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


  out_dir = path.join(FLAGS.train_dir,
                      'path_renders' if config.render_path else 'test_preds')
  
  state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)

  if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
  gt = torch.empty((dataset.h * dataset.w * config.img_generated, 9))
  cont = 0 
  for idx in range(dataset.size):
    print(f'Generating {idx+1}/{dataset.size}')
    
    batch = next(dataset)
    
    pred_color, pred_distance, pred_acc = models.render_image(
        functools.partial(render_eval_pfn, state.optimizer.target),
        batch['rays'],
        None,
        chunk=FLAGS.chunk)
    rgb = np.clip(np.nan_to_num(pred_color), 0., 1.).reshape(-1, 3)
    origins = np.array(batch['rays'].origins).reshape(-1,3)
    directions = np.array(batch['rays'].directions).reshape(-1,3)
    
    gt[cont:cont + len(batch['rays'].origins)**2] = torch.cat((
                        torch.Tensor(origins),
                        torch.Tensor(directions),
                        torch.Tensor(rgb)),
                        dim=-1)
    cont += len(batch['rays'].origins)**2
  export_data(FLAGS.seed, FLAGS.data_dir.split('/')[-1], gt, FLAGS.data_dir)


if __name__ == '__main__':
  app.run(main)