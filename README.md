# Re-ReND: Real-time Rendering of NeRFs across Devices

### [ArXiv](https://arxiv.org/abs/2303.08717) | [PDF](https://arxiv.org/pdf/2303.08717.pdf) 

This repository is the official implementation of Re-ReND, a method for real-time rendering of NeRFs.

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
```bash
git clone https://github.com/sararoma95/Re-ReND.git && cd Re-ReND
```
### 1. Set up environment with Anaconda
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
We extract 10k images and a mesh for each scene of the *Blender Synthetic dataset* and the *Tanks & Temples dataset* from MipNeRF and NeRF++, respectively.
You can download them [here](https://drive.google.com/drive/folders/1FZPRaU7w9S0aaBSUpyYHUUeHTJ68gJqD?usp=share_link).
Note that each scene is large (~120GB).

### 3. Training
We train on an A100 GPU for 2.5 days to reach 380k iterations for synthetic scenes and 1 day to reach 150k iters for Tanks & Temples scenes.
```bash
python main.py --config configs/chair.txt --train
```
### 4. Evaluate before quantization (continuous)
```bash
python main.py --config configs/chair.txt --render_only
```
### 5. Export textures UVWB
```bash
python main.py --config configs/chair.txt --export_textures
```
### 5. Evaluate after quantization 
```bash
python main.py --config configs/chair.txt --compute_metrics
```
### 6. Running the viewer
The viewer code is provided in this repo, as three .html files for two types of datasets.
The instructions to use are inside the folder viewer.

### Note: Create the data by yourself
The scripts to extract the data are also provided for this specific implementations. Feel free to recreate the data by yourself.

## Citation
```
@article{rojas2023rerend,
  title={{R}e-{R}e{ND}: {R}eal-time {R}endering of {N}e{RF}s across {D}evices},
  author={Rojas, Sara and Zarzar, Jesus and {P{\'e}rez}, Juan C. and Sanakoyeu, Artsiom and Thabet, Ali and Pumarola, Albert and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2303.08717},
  year={2023}
}
```
