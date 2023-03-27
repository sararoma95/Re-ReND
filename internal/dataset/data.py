import os
from glob import glob
import time
import torch
from os.path import exists, join

from internal.dataset.load_blender import load_blender_data
from internal.dataset.load_tandt import load_tandt_data
from internal.utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_dataset(args):
    if args.dataset_type == 'blender':
        dataset = load_blender_data(args.datadir, 'test', half_res=False, testskip=args.testskip)
    elif args.dataset_type == 'tanks_and_temples':
        dataset = load_tandt_data(args.datadir, args.expname, 'test', try_load_min_depth=False)
    return dataset

def load_train_dataset(path, num_files, file=0, device='cpu'):
    prYellow('Loading data ...')
    start_time = time.time()

    # Where the data is saved
    path = os.path.dirname(path)
    files = sorted(glob(join(path, '*.pt')))

    # Creating .txt to load data faster (only once)
    for i in range(0, len(files), num_files):
        if not exists(join(path, f'size{i}.txt')):
            with open(join(path, f'size{i}.txt'), 'w') as fp:
                for fl in files[i:i + num_files]:
                    fp.write("%s\n" % torch.load(fl).shape[0])
            prYellow(f'Created size{i}.txt file')

    # Reading .txt to know total data to load
    f = open(join(path, f'size{file}.txt'), "r")
    total_data = 0
    for x in f:
        total_data += int(x)
    data = torch.zeros(total_data, 10, device=torch.device(device))
    cont = 0
    
    # Read only num_files files and load them
    prBlue(f'Reading size{file}.txt')
    for fl in files[file:file + num_files]:
        prCyan(f'Loading file: {fl}')
        _data = torch.load(fl, map_location=torch.device(device))
        data[cont:cont + _data.shape[0]] = _data
        cont += _data.shape[0]

    prYellow(f'Loading took {time.time() - start_time}')
    return data