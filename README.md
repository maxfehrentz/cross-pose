<h1 align="center">

CT-Informed Intraoperative Non-Rigid Gaussian Splatting

</h1>

<br>

## Overview
This repository contains the implementation of CT-Informed Intraoperative Non-Rigid Gaussian Splatting.

## Installation

### Prerequisites
- CUDA 12.1 or higher
- Python 3.8 or higher
- Conda package manager

### Installation Steps
1. Create the conda environment from the provided file. This was generated with conda-minify.
```bash
conda env create -f environment.yaml
conda activate bridgesplat
```
In case that does not work, there is also the direct export from the environment. This contains lots of extra packages from experimenting with SAM, Dust3r, DepthAnything etc. and is huge!
```bash
conda env create -f environment_huge.yaml
conda activate bridgesplat
```

2. Install the custom cpp extensions and submodules:
```bash
cd src/cpp_extensions/arap
python setup.py install
cd ../../..

# Install Gaussian Rasterization
cd src/cpp_extensions/gaussian_rasterization
python setup.py install
cd ../../..

# Install Diff Gaussian Rasterization
cd src/cpp_extensions/diff_gaussian_rasterization
python setup.py install
cd ../../..
```

Note that we use `arap.cpp` and `arap.h` in the ARAP extension. Neither the experimental nor the CUDA version are yielding satisfactory results at the moment. Leaving them in for now to fix them at some point.

Depth Anything v2 is also part of this repository; it is currently not used but we have experimented with depth supervision. Either modify the code and just provide empty tensors instead (no impact on functionality) or follow their instructions and download the appropriate checkpoints.

## Usage

1. Prepare your configuration file (for example, `configs/ATLAS/trial11_vid08_dataset1_rect.yaml`)

2. Run the optimization with the configuration file:
```bash
python run.py configs/ATLAS/trial11_vid08_dataset1_rect.yaml --visualize
```

### Configuration

The main configuration parameters can be set in the YAML config files. Pay attention to the inheritance structure. This is also where ARAP can be activated/deactivated.

### Visualization

The optimization process generates several visualization outputs in the specified output directory:
- `mapping/`: Collection of several results, including inital and deformed mesh, rendered image, ...
- `semantic/`: Rendered images
- `overlays/`: Overlay visualizations of the deformed mesh
- `deformation/`: Deformation field visualizations
- `mesh/`: Mesh visualizations

Note: semantic will be renamed to rendering, was previously used for rendering semantics.