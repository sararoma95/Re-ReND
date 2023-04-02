#!/bin/bash
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
export LD_LIBRARY_PATH="/home/rojass/anaconda3/envs/multinerf/lib"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.8/site-packages/tensorrt/
export CUDA_VISIBLE_DEVICES=0
export TF_ENABLE_ONEDNN_OPTS=0 
# Script for evaluating on the Blender dataset.
SCENE='chair'
SEED=8789
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -u)
    shift # past argument
    SCENE="$1"
    shift # past value
    ;;
    -c)
    shift # past argument
    SEED="$1"
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo $SCENE
echo $SEED

TRAIN_DIR=/ibex/ai/home/rojass/mipnerf-pytorch/data/nerf_synthetic/nerf_results_mipnerf/$SCENE
DATA_DIR=/ibex/ai/home/rojass/albert/data/nerf_synthetic/$SCENE

python -m grid \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --chunk=3076 \
  --gin_file=configs/blender.gin \
  --logtostderr \
  --seed=$SEED
