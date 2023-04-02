import os
import glob
import imageio
import shutil
import json
import argparse
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/zarzarj/git/FactorizedNeLF/NeLF_out/",
        help="Path containing scene folders",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="chair",
        help="Scene name",
    )
    parser.add_argument(
        "--num_tex",
        type=int,
        default=8,
        help="Number of texture map squares (8 for 32 components, 16 for 64 components)",
    )
    parser.add_argument(
        "--split_x",
        type=int,
        default=4,
        help="Number of horizontal splits to texture maps",
    )
    parser.add_argument(
        "--split_y",
        type=int,
        default=1,
        help="Number of vertical splits to texture maps (1 for 32 components and quad size 6, 4 for 64 components)",
    )
    parser.add_argument(
        "--quad_size",
        type=int,
        default=6,
        help="Quad size",
    )

    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    texture_names = ["u", "v", "w", "b"]
    qs = args.quad_size
    num_tex = args.num_tex
    in_folder = os.path.join(args.path, args.scene, f'meshes_textures_{qs}_{num_tex}')
    out_folder = os.path.join(args.path, args.scene, f'meshes_textures_{qs}_{num_tex}_split')
    os.makedirs(out_folder, exist_ok=True)
    split_x = args.split_x
    split_y = args.split_y
    for tex in texture_names:
        tex_file = os.path.join(in_folder, f'feat_{tex}.png')
        texture = imageio.imread(tex_file)
        for i in range(split_x):
            for j in range(split_y):
                tex_idx = j * split_y + i
                out_tex_file = os.path.join(out_folder, f"feat_{tex}_{tex_idx}.png")
                out_texture = texture[texture.shape[0]//split_y * j:texture.shape[0]//split_y * (j+1), texture.shape[1]//split_x * i:texture.shape[1]//split_x * (i+1)]
                imageio.imsave(out_tex_file, out_texture)


    with open(os.path.join(in_folder, 'minmax.json')) as f:
        minmax = json.load(f)
    out_d = {}
    num_x, num_y = 4, num_tex // 4
    indices = np.arange(num_tex*4).reshape(num_y, num_x,4)
    new_indices = []
    for j in range(split_y):
        for i in range(split_x):
            # new_indices.append(indices[(num_y//split_y) * j:(num_y//split_y) * (j+1), (num_x//split_x) * i:(num_x//split_x) * (i+1),:].transpose((1,0,2)).reshape(-1))
            new_indices.append(indices[(num_y//split_y) * j:(num_y//split_y) * (j+1), (num_x//split_x) * i:(num_x//split_x) * (i+1),:].reshape(-1))
    new_indices = np.concatenate(new_indices)

    for tex in texture_names:
        mini = np.array(minmax[f"min_{tex}"])
        out_d[f"min_{tex}"] = mini[new_indices].tolist()
        maxi = np.array(minmax[f"max_{tex}"])
        out_d[f"max_{tex}"] = maxi[new_indices].tolist()
    with open(os.path.join(out_folder, "minmax.json"), "w") as f:
        json.dump(out_d, f)
        
    obj_in_path = os.path.join(in_folder, f'{args.scene}.obj')
    obj_out_path = os.path.join(out_folder, f'{args.scene}.obj')
    os.system(f"cp {obj_in_path} {obj_out_path}")