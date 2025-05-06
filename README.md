# Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition

## Quick Start
Take increment-calibrated caching for DiT as an example.
# Setup
Download and setup the repo:
```bash
git clone https://github.com/ccccczzy/icc.git
cd icc/DiT
```
Create the environment and install required packages:
```
conda env create -f environment.yml
conda activate DiT
```
# Usage
To generate calibration parameters with SVD at the specified rank:
```bash
python gen_decomp.py --use-wtv False --rank 128 
```
To generate calibration parameters with CA-SVD at the specified rank:
```bash
python gen_decomp.py --use-wtv True --wtv-src "mag" --num-steps 50 --num-samples 256 --rank 128 --data-path /path/to/imagenet/train
```
To generate calibration parameters with CD-SVD at the specified rank:
```bash
python gen_decomp.py --use-wtv True --wtv-src "delta_mag" --num-steps 50 --num-samples 256 --rank 128 --data-path /path/to/imagenet/train
```
To sample images with the generated calibration parameters:
```
torchrun --nnodes=1 --nproc_per_node=N sample_ddp_sp_step.py --num-fid-samples 50_000 --results-dir /path/to/calibration/parameters
```

# Acknowledgments
This codebase borrows from [DiT](https://github.com/facebookresearch/DiT) and [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha). Thanks to the authors for their wonderful work and codebase!
