# Visual Prompt-Agnostic Evolution 

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://arxiv.org/abs/2601.20232)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Visual Prompt-Agnostic Evolution"** (ICLR 2026).


## Key Features

- **1.41x faster convergence** on average across VPT variants
- **1-3% accuracy gains** on 25 datasets with multiple downstream tasks
- **Prompt-agnostic**: Works with any VPT variant without backbone modification
- **No inference overhead**: Only affects training dynamics

## Installation

### Requirements

```bash
# Core dependencies
torch>=1.7.1
torchvision>=0.8.2
timm==0.4.12
opencv-python>=4.5.0
Pillow>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0  # Note: Use NumPy 1.x (avoid NumPy 2.x)
scipy>=1.5.0
pandas>=1.1.0
scikit-learn>=0.24.0

# For VTAB datasets
tensorflow-metadata>=1.0.0
tfds-nightly
```

### Pretrained Backbones

We support the following pretrained backbones (all pretrained on **ImageNet-21k**):

| Backbone | Architecture | Params | Download |
|----------|-------------|--------|----------|
| ViT-B/16 | Vision Transformer Base | 85.8M | [ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) |
| ViT-L/16 | Vision Transformer Large | 303.3M | [ViT-L_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz) |
| ViT-H/14 | Vision Transformer Huge | 630.8M | [ViT-H_14.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz) |
| Swin-B | Swin Transformer Base | 87.8M | [swin_base_patch4_window7_224_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) |

**Self-supervised backbone:**
| Backbone | Architecture | Download |
|----------|-------------|----------|
| MAE | ViT-B/16 (Masked Autoencoder) | [mae_pretrain_vit_base.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) |

**Segmentation backbone:**
| Backbone | Architecture | Download |
|----------|-------------|----------|
| SETR | ViT-L/16 | [SETR weights](https://github.com/fudan-zvg/SETR) |

**Setup:**
1. Download the desired backbone weights
2. Place in your model directory, e.g., `/path/to/data/weights/`
3. Set `MODEL.MODEL_ROOT=/path/to/data` in training commands

### Datasets & Benchmarks

We evaluate on **25 datasets** across multiple downstream tasks:

#### FGVC (Fine-Grained Visual Classification)

| Dataset | Classes | Train | Val | Test |
|---------|---------|-------|-----|------|
| [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | 200 | 5,394 | 600 | 5,794 |
| [NABirds](https://dl.allaboutbirds.org/nabirds) | 555 | 21,536 | 2,393 | 24,633 |
| [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 102 | 1,020 | 1,020 | 6,149 |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | 120 | 10,800 | 1,200 | 8,580 |
| [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | 196 | 7,329 | 815 | 8,041 |

#### VTAB-1k (Visual Task Adaptation Benchmark)

19 diverse visual tasks with 1,000 training samples each, grouped into three categories:

**Natural (7 tasks):** CIFAR-100, Caltech101, DTD, Flowers102, Pets, SVHN, SUN397

**Specialized (4 tasks):** Patch Camelyon, EuroSAT, RESISC45, Retinopathy

**Structured (8 tasks):** CLEVR/count, CLEVR/distance, DMLab, KITTI/distance, dSprites/location, dSprites/orientation, SmallNORB/azimuth, SmallNORB/elevation

> VTAB-1k datasets can be downloaded via [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/overview).

#### Semantic Segmentation

| Dataset | Classes | Train | Val |
|---------|---------|-------|-----|
| [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | 150 | 20,210 | 2,000 |

## Quick Start

### Training with PAE (CUB-200-2011)

```bash
python train.py \
  --config-file configs/prompt/cub.yaml \
  MODEL.TYPE vit \
  DATA.BATCH_SIZE 128 \
  MODEL.PROMPT.INITIATION mpa \
  MODEL.PROMPT.KOOPMAN_ENABLED True \
  MODEL.PROMPT.KOOPMAN_DIM 256 \
  MODEL.PROMPT.KOOPMAN_WEIGHT 0.5 \
  MODEL.PROMPT.LYAPUNOV_WEIGHT 0.2 \
  MODEL.PROMPT.KOOPMAN_MOD global \
  MODEL.PROMPT.DEEP True \
  MODEL.PROMPT.DROPOUT 0.1 \
  DATA.FEATURE sup_vitb16_imagenet21k \
  DATA.NAME CUB \
  DATA.NUMBER_CLASSES 200 \
  SOLVER.BASE_LR 0.25 \
  SOLVER.WEIGHT_DECAY 0.001 \
  SEED 42 \
  MODEL.MODEL_ROOT /path/to/data \
  DATA.DATAPATH /path/to/data/CUB_200_2011 \
  OUTPUT_DIR output
```

### Key Configuration Options

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `MODEL.PROMPT.INITIATION` | Initialization method (`random`, `mpa`) | `mpa` | 
| `MODEL.PROMPT.KOOPMAN_ENABLED` | Enable KLD regularization | `True` | 
| `MODEL.PROMPT.KOOPMAN_MOD` | Koopman mode (`global`, `layerwise`) | `global` |

## Method Overview

### MPA: Modal Pre-Alignment

1. **Phase I - Frequency Shortcut Discovery**: Generate sliding window masks in frequency domain, evaluate task loss, select top-T masks
2. **Phase II - Prompt Initialization**: Energy-weighted pooling of filtered patch tokens, propagate through frozen encoder

### KLD: Koopman-Lyapunov Dynamical System

- **Koopman Evolution**: `z_{i+1} = z_i @ K`
- **Consistency Loss**: `L_kp = Σ ||z_{i+1} - z_i @ K||²` 
- **Lyapunov Stability**: `L_stab = Σ max(0, V(z_{i+1}) - V(z_i))`

## Project Structure

```
├── configs/              # Configuration files
│   └── prompt/
│       └── cub.yaml      # CUB-200-2011 config
├── src/
│   ├── models/
│   │   ├── vit_prompt/
│   │   │   └── vit.py    # PromptedTransformer with KLD
│   │   └── vit_models.py # ViT model with MPA
│   ├── engine/
│   │   └── trainer.py    # Training loop with KLD loss
│   └── solver/
│       └── optimizer.py  # Optimizer with Koopman param groups
├── train.py              # Main training script
└── run.sh                # Example training scripts
```

## Citation

```bibtex
@article{wang2026visual,
  title={Visual Prompt-Agnostic Evolution},
  author={Wang, Junze and Fan, Lei and Zhang, Dezheng and Jing, Weipeng and Di, Donglin and Song, Yang and Liu, Sidong and Cong, Cong},
  journal={arXiv preprint arXiv:2601.20232},
  year={2026}
}
```

## Acknowledgements

This codebase builds upon [VPT](https://github.com/kmnp/vpt). We thank the authors for their excellent work.

