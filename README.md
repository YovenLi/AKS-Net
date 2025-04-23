# AKS-Net: No-Reference Point Cloud Quality Assessment with Adaptive Keyframe Selection

This repository contains the official implementation of "No-Reference Point Cloud Quality Assessment with Adaptive Keyframe Selection" presented at the 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP).

[![IEEE Paper](https://img.shields.io/badge/IEEE-10.1109%2FVCIP63160.2024.10849833-blue)](https://doi.org/10.1109/VCIP63160.2024.10849833)

## Overview

AKS-Net is a framework for point cloud quality assessment that leverages an adaptive keyframe selection approach. The system consists of two main components:
1. Adaptive Keyframe Selection Module
2. Quality Assessment Module based on selected keyframes

The method projects 3D point clouds into 2D images from multiple viewpoints, selects the most representative keyframes, and performs quality assessment using these keyframes, significantly reducing computational complexity while maintaining accuracy.

## Project Structure

The project is organized into two main modules:

- `get_keyframe/`: Implementation of the adaptive keyframe selection module
- `train/`: Implementation of the keyframe-based quality assessment module

## Key Features

- 3D-to-2D projection with multiple rotation viewpoints (120 frames)
- Adaptive keyframe selection using deep learning models
- No-reference point cloud quality assessment
- Support for multiple datasets (SJTU, WPC)

## Installation

### Requirements

- Python >= 3.7
- PyTorch 1.7+
- CUDA-capable GPU
- Dependencies: numpy, pandas, scipy, PIL, torchvision, open3d

> **Note:** This project was developed and tested on Ubuntu 20.04, Python 3.8, PyTorch 2.0.1, and Open3D 0.17.0

### Quick Setup

```bash
# Create environment
conda create -n aksnet python=3.8
conda activate aksnet

# Install PyTorch
conda install pytorch torchvision cudatoolkit -c pytorch

# Install other dependencies
pip install numpy pandas scipy pillow open3d
```

## Usage

### Complete Pipeline

1. **Generate 120 frames from point cloud projections**:
```bash
python train/rotation/rotation.py --path ./ply/ --frame_path ./frames/
```

2. **Extract keyframes**:
```bash
python get_keyframe/get_keyframe.py --database sjtu --num_keyframes 9 --data_dir_frame ./frames/ --output_base_dir ./keyframes/
```

3. **Train quality assessment model**:
```bash
sh train/train_sjtu.sh  # SJTU dataset
sh train/train_wpc.sh   # WPC dataset
```

## Dataset Support

The project supports the following datasets:
- SJTU: Shanghai Jiao Tong University Point Cloud Video Quality Dataset
- WPC: Waterloo Point Cloud Dataset

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10849833,
  author={Wang, Haomiao and Wang, Xiaochuan and Yuan, Xianpeng and Chen, Xianming and Li, Haisheng},
  booktitle={2024 IEEE International Conference on Visual Communications and Image Processing (VCIP)}, 
  title={No-Reference Point Cloud Quality Assessment with Adaptive Keyframe Selection}, 
  year={2024},
  pages={1-5},
  keywords={Point cloud compression;Training;Image quality;Visual communication;Image processing;Video sequences;Redundancy;Particle measurements;Loss measurement;Quality assessment;Point cloud quality assessment;viewpoint selection;keyframe;no-reference;projection-based metrics},
  doi={10.1109/VCIP63160.2024.10849833}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Haomiao Wang
- Xiaochuan Wang
- Xianpeng Yuan
- Xianming Chen
- Haisheng Li 