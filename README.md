# Re-ReND: Real-time Rendering of NeRFs across Devices

### [ArXiv](https://arxiv.org/abs/2303.08717) | [PDF](https://arxiv.org/pdf/2303.08717.pdf) 

This repository is for the Re-ReND method.

**[TL;DR]** We propose Re-ReND for efficient real-time rendering of pre-trained Neural Radiance Fields (NeRFs) on resource-limited devices. Re-ReND achieves this by distilling the NeRF representation into a mesh of learned densities and a set of matrices representing the learned light field, which can be queried using inexpensive matrix multiplications.

Training Re-ReND:
<div align="center">
    <a><img src="figs/Training.png"  width="700" ></a>
</div>
Rendering a NeRF using Re-ReND. 
<div align="center">
    <a><img src="figs/Rendering.png"  width="700" ></a>
</div>

## Reproducing Our Results
### 0. Download the code
```
git clone https://github.com/sararoma95/Re-ReND.git && cd Re-ReND
```
### 2. Set up environment with Anaconda
- `conda create --name Re-ReND python=3.7.13`
- `conda activate Re-ReND`
- `pip install -r requirements.txt` 
- `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
- `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- `pip install pymeshlab`
- `pip install PyMCubes`
- `pip install scikit-fmm`
- `pip install trimesh`
- `conda install -c anaconda scikit-image`
- `pip install lpips`
- `conda install -c conda-forge pyembree`

### 2. Download data for Re-ReND
We extract 10k images and a mesh for each scene from MipNeRF and NeRF++ for the synthetic dataset and the Tanks & Temples dataset, respectively.
You can download them [here](https://drive.google.com/drive/folders/1FZPRaU7w9S0aaBSUpyYHUUeHTJ68gJqD?usp=share_link).
Note that each scene is ~120GB.

### 3. Training

We train on a GPU A100 for 2.5 days to reach 380k iters. However, training for 1 day is enough to obtain good results. 

```
python main.py --config configs/chair.txt --train
```
### 4. Evaluate before quantization (continuos)
```
python main.py --config configs/chair.txt --render_only
```
### 5. Export textures UVWB
```
python main.py --config configs/chair.txt --export_textures
```
### 5. Evaluate after quantization 
```
python main.py --config configs/chair.txt --compute_metrics
```
### 6. Running the viewer
The viewer code is provided in this repo, as three .html files for two types of datasets.
You can set up a local server on your machine, e.g.,
```
cd folder_containing_the_html
python -m http.server
```
Then open
```
localhost:8000/view_syn.html?obj=chair
```
Note that you should put the meshes+textures of the chair model in a folder chair_phone. The folder should be in the same directory as the html file.

Please allow some time for the scenes to load. Use left mouse button to rotate, right mouse button to pan (especially for forward-facing scenes), and scroll wheel to zoom. On phones, Use you fingers to rotate or pan or zoom. Resize the window (or landscape<->portrait your phone) to show the resolution.

Based on MobileNeRF github.

### Note: Create the data by yourself
The scripts to extract the data are also provided for this especif implementations. Feel free to recreate the data by yourself.

## Citation
@article{rojas2023rerend,
  title={{R}e-{R}e{ND}: {R}eal-time {R}endering of {N}e{RF}s across {D}evices},
  author={Rojas, Sara and Zarzar, Jesus and {P{\'e}rez}, Juan C. and Sanakoyeu, Artsiom and Thabet, Ali and Pumarola, Albert and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2303.08717},
  year={2023}
}
