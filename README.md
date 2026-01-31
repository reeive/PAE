# Visual Prompt-Agnostic Evolution

> **Demo available.** This repository provides a minimal working example to reproduce the core pipeline.  
> The complete implementation and full documentation will be released later.

## Environment

- `torch==1.7.1+cu110`, `torchvision==0.8.2+cu110`, `torchaudio==0.7.2`
`timm==0.4.12`, `opencv-python==4.12.0.88`, `Pillow==11.3.0`, `matplotlib==3.9.4`
`numpy==1.26.4`, `scipy==1.13.1`, `pandas==2.3.2`, `scikit-learn==1.6.1` 
`tensorflow-metadata==1.17.2`, `tfds-nightly==4.4.0.dev202201080107`

> **Note:** PyTorch 1.7.x works best with NumPy 1.x. Please avoid NumPy 2.x.

## Pretrained Vision Backbone

We use the **ViT-B/16** backbone pretrained on **ImageNet-21k**.

- **Download link**: [ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)
- Suggested location:
  - Put the file under your model root, e.g. `/path/to/data/weights/ViT-B_16.npz`
  - Make sure `MODEL.MODEL_ROOT=/path/to/data` in the training command

## Examples for training([CUB-200-2011](https://data.caltech.edu/records/65de6-vp158))

Launch training [VPT](https://github.com/kmnp/vpt) with **MPA** initialization and **KLD** optimization:

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
  MODEL.PROMPT.DEEP True \
  MODEL.PROMPT.DROPOUT 0.1 \
  DATA.FEATURE sup_vitb16_imagenet21k \
  DATA.NAME CUB \
  DATA.NUMBER_CLASSES 200 \
  SOLVER.BASE_LR 0.25 \
  SOLVER.WEIGHT_DECAY 0.001 \
  SEED 666 \
  MODEL.MODEL_ROOT /path/to/data \
  DATA.DATAPATH /path/to/data/CUB_200_2011 \
  OUTPUT_DIR output
