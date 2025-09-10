Please note this repo is no longer being developed.

For SOG compression, please use https://github.com/playcanvas/splat-transform.

# SOGS

Python package to compress Gaussian Splats with [Self-Organizing Gaussians](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians)

Code forked from gsplat's [png_compression](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/compression/png_compression.py) module and produces a compressed bundle suitable for rendering with PlayCanvas' SuperSplat.

## Installation

Requires [torch](https://pytorch.org/get-started/locally/), [torchpq](https://github.com/DeMoriarty/TorchPQ) (which requires [cupy](https://cupy.dev/), and [PLAS](https://github.com/fraunhoferhhi/PLAS), which require CUDA. These must be manually installed as they require installation against a specific version of CUDA (the one you have installed).

For instance, if you're running CUDA 12.6 on Windows you may install these dependencies (ideally in some kind of virtual environment):

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x
pip install torchpq
pip install git+https://github.com/fraunhoferhhi/PLAS.git
pip install git+https://github.com/playcanvas/sogs.git
```

## Usage

`sogs-compress --ply your_ply_file.ply --output-dir directory_to_store_images_and_metadata`

## Development

In order to develop and run the local version, install sogs like this instead:

```
git clone https://github.com/playcanvas/sogs.git
cd sogs
pip install -e ./sogs
```

And invoke it from the `/src` folder like so:
```
python.exe -m sogs.cli --ply filename.ply --output_dir directory_name
```
