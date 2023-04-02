# Extract data from Neural Radiance fields
You will have to download [MipNeRF](https://github.com/google/mipnerf) and [NeRF++](https://github.com/Kai-46/nerfplusplus) and train their models.
After training and keeping their enviroments. You should do this.
## [MipNeRF](https://github.com/google/mipnerf)
1. Put inside <code>scripts</code> folder the following scripts <code>extract_imgs.sh</code> and <code>extract_grid.sh</code>
2. Change *TRAIN_DIR* and *DATA_DIR* in <code>extract_imgs.sh</code> and <code>extract_grid.sh</code>
3. Change in <code>internal</code> folder utils with the one provided.

4. To extract the grid of densities, you should run:
```
sh scripts/extract_grid.sh -u chair
```
5. To extract the 500 images, you should run:
```
sh scripts/extract_imags.sh -u SCENE -c SEED
```
    Each time you run this you get a file <code>{SCENE}_{SEED}.pt</code> 

    <code>-c</code> is the seed in order to synthetize diferent chucks of 500 images. Also, it means the name of the file blender_paper_chair_{SEED}.pt

    You should run it 20 times to get the 10k images.

    Files will be drop in <code>pseudo</code> folder

