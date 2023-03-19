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