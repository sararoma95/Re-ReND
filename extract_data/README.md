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
5. To extract the mesh from the density grid, you will have to use <code>--from_file</code> for the path of the grid of densities.
```
python main.py --config configs/chair.txt    --create_mesh  --from_file {PATH}/chair.npy      --threshold 49 
```
### Extract the 10k imgs
6. To extract the 500 images, you should run:
```
sh scripts/extract_imags.sh -u SCENE -c SEED
```
  Each time you run this you get a file <code>{SCENE}_{SEED}.pt</code> with 500 images.
  
  You should run it 20 times to get the 10k images.
  
  <code>-c</code> is the seed in order to synthetize different chunks of 500 images. Also, it means that the name of the file will be <code>blender_paper_chair_{SEED}.pt</code>.
  
  Files will be drop in <code>pseudo</code> folder


