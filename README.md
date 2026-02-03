# Visual Prompt-Agnostic Evolution 

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://arxiv.org/abs/2601.20232)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Visual Prompt-Agnostic Evolution"** (ICLR 2026).

PAE strengthens Visual Prompt Tuning (VPT) by explicitly modeling the dynamics of learnable prompts through:
- **MPA (Modal Pre-Alignment)**: Task-aware prompt initialization via frequency-domain shortcuts
- **KLD (Koopman-Lyapunov Dynamical System)**: Cross-layer prompt evolution with stability guarantees

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

### Pretrained Backbone

We use **ViT-B/16** pretrained on **ImageNet-21k**:

1. Download: [ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)
2. Place in your model directory, e.g., `/path/to/data/weights/ViT-B_16.npz`
3. Set `MODEL.MODEL_ROOT=/path/to/data` in training commands

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

