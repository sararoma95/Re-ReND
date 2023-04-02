# Extract data from Neural Radiance fields
You will have to download [MipNeRF](https://github.com/google/mipnerf) and [NeRF++](https://github.com/Kai-46/nerfplusplus) and download their models [here](https://drive.google.com/drive/folders/1bqp7a-ipvPkFIu5II7xDN0UFoZscToCW?usp=share_link).
After training and keeping their enviroments. You should do this.

## [MipNeRF](https://github.com/google/mipnerf)
1. Put inside the <code>scripts</code> folder the following scripts <code>extract_imgs.sh</code> and <code>extract_grid.sh</code>. 
  Also, put <code>extract_imgs_mipnerf.py</code> and <code>extract_grid_mipnerf.py</code> in the main directory.
2. Change *TRAIN_DIR* and *DATA_DIR* in <code>extract_imgs.sh</code> and <code>extract_grid.sh</code>
3. Change in the <code>internal</code> folder the <code>utils.py</code> script with the one provided.

### Extract meshes
4. To extract the grid of densities and the mesh, you should run:
```
sh scripts/extract_grid.sh -u SCENE
```
5. To extract the mesh from the density grid, you should run:
```
python main.py --config configs/chair.txt    --create_mesh  --from_file {PATH}/chair.npy      --threshold 49 
```
  You will have to use <code>--from_file</code> for the path of the grid of densities.
### Extract images
6. To extract the 500 images, you should run:
```
sh scripts/extract_imags.sh -u SCENE -c SEED
```
  Each time you run this you get a file <code>{SCENE}_{SEED}.pt</code> with 500 images.
  
  You should run it 20 times to get the 10k images.
  
  <code>-c</code> is the seed in order to synthetize different chunks of 500 images. Also, it means that the name of the file will be <code>blender_paper_chair_{SEED}.pt</code>.
  
  Files will be drop in <code>pdata</code> folder
7. Divide the data, from origins and directions to intersected points.
```
python main.py --config configs/chair.txt --divide
```

## [NeRF++](https://github.com/Kai-46/nerfplusplus)
1. Put <code>extract_imgs_nerf++.py</code> and <code>extract_grid_nerf++.py</code> in the main directory.
2. Add the following line in <code>ddp_train_nerf.py</code> script in line 609:
```
parser.add_argument("--id", type=int, default=0, help='id gpu generated data')
```
### Extract meshes
2. To extract the grid of densities and the mesh, you should run:
```
python extract_grid_nerf++.py --config configs/SCENE.txt 
```
3. To extract the mesh from the density grid, you should run:
```
python main.py --config configs/tat_training_truck.txt    --create_mesh  --from_file {PATH}/truck.npy      --threshold 12 
```
### Extract images
4. To extract the 500 images, you should run:
```
python extract_grid_nerf++.py --config configs/SCENE.txt --folder pdata --id SEED
```
  Each time you run this you get a file <code>{SCENE}_{SEED}.pt</code> with 500 images.
  
  You should run it 20 times to get the 10k images.
 <code>--id</code> is the seed in order to synthetize different chunks of 500 images. Also, it means that the name of the file will be <code>tat_training_Truck_{SEED}.pt</code>.

7. Divide the data, from origins and directions to intersected points.
```
python main.py --config configs/tat_training_truck.txt --divide
```