#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging
import time
logger = logging.get_logger("prompt_agnostic_evolution")
import torch.fft as fft_module


def _fftshift(x):

    h, w = x.shape[-2:]
    return torch.roll(x, shifts=(h // 2, w // 2), dims=(-2, -1))

def _ifftshift(x):

    h, w = x.shape[-2:]
    return torch.roll(x, shifts=(-(h // 2), -(w // 2)), dims=(-2, -1))

def _fft2(x):
    temp = fft_module.fft(x, dim=-2)

    return fft_module.fft(temp, dim=-1)

def _ifft2(x):
    temp = fft_module.ifft(x, dim=-2)
    return fft_module.ifft(temp, dim=-1)

class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)

    def init_mpa(self, train_loader, window_size=16, stride=8):
        """
        Modal Pre-Alignment (MPA) initialization following the paper.
        
        Args:
            train_loader: DataLoader for training data
            window_size: Size of the sliding window (w in paper, default 16)
            stride: Stride of the sliding window (r in paper, default 8)
        """
        if not hasattr(self.enc, 'transformer') or not hasattr(self.enc.transformer, 'prompt_embeddings'):
            logger.error("MPA Error: The backbone 'self.enc' is not a PromptedVisionTransformer. Aborting MPA.")
            return

        logger.info("Starting MPA initialization (Paper-aligned sliding window)...")
        logger.info(f"   Window size w={window_size}, stride r={stride}")
        # ==================== METRICS SETUP ====================
        start_time = time.time()
        device = self.head.parameters().__next__().device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        # =======================================================

        self.eval()

        try:
            batch = next(iter(train_loader))
            x_batch = batch['image']
            y_batch = batch['label']
        except StopIteration:
            logger.error("Train loader is empty, cannot perform MPA.")
            return

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        prompt_cfg = self.cfg.MODEL.PROMPT
        num_tokens = prompt_cfg.NUM_TOKENS
        B, C, H, W = x_batch.shape

        # ========== Paper formula: sliding window frequency masks ==========
        # S = (floor((H-w)/r) + 1) * (floor((W-w)/r) + 1)
        w = window_size
        r = stride
        
        num_h = (H - w) // r + 1  # vertical window count
        num_w = (W - w) // r + 1  # horizontal window count
        S = num_h * num_w  # total candidate masks
        
        logger.info(f"MPA Phase I: Generating {S} candidate masks ({num_h} x {num_w} grid)")

        candidate_masks_list = []
        for i in range(num_h):
            for j in range(num_w):
                h_start = i * r
                w_start = j * r
                mask = torch.zeros(H, W, device=device)
                mask[h_start:h_start + w, w_start:w_start + w] = 1.0
                candidate_masks_list.append(mask)

        if not candidate_masks_list:
            raise ValueError(
                f"Patch size k={patch_size_k} may be larger than image dimensions H={H}, W={W}. No masks generated.")

        candidate_masks = torch.stack(candidate_masks_list, dim=0)
        s_candidate_masks = candidate_masks.shape[0]
        logger.info(f"Generated {s_candidate_masks} systematic candidate patch masks.")
        # =================================================================

        losses = torch.zeros(s_candidate_masks, device=device)
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for s in range(s_candidate_masks):
                mask = candidate_masks[s].view(1, 1, H, W)
                x_freq = _fftshift(_fft2(x_batch))
                filtered_freq = x_freq * mask
                x_hat = torch.real(_ifft2(_ifftshift(filtered_freq)))

                logits = self(x_hat)
                losses[s] = loss_fn(logits, y_batch)

        _, top_indices = torch.topk(losses, num_tokens, largest=False)
        selected_masks = candidate_masks[top_indices]
        logger.info(f"Selected {num_tokens} frequency shortcut masks.")


        logger.info("MPA Phase II: Initializing Prompts...")
        patch_embedder = self.enc.transformer.embeddings
        representative_tokens = []
        with torch.no_grad():
            for r in range(num_tokens):
                mask = selected_masks[r].view(1, 1, H, W)
                x_freq = _fftshift(_fft2(x_batch))
                filtered_freq = x_freq * mask
                x_hat_r = torch.real(_ifft2(_ifftshift(filtered_freq)))

                patch_tokens = patch_embedder(x_hat_r)[:, 1:, :]
                energies = torch.sum(patch_tokens ** 2, dim=-1)
                weights = energies / torch.sum(energies)
                t_r = torch.sum(weights.unsqueeze(-1) * patch_tokens, dim=(0, 1))
                representative_tokens.append(t_r)

        P_init_1 = torch.stack(representative_tokens, dim=0)
        self.enc.transformer.prompt_embeddings.data.copy_(P_init_1.unsqueeze(0))
        logger.info("Initialized shallow prompts.")

        if prompt_cfg.DEEP:
            logger.info("Propagating prompts for deep layers...")
            current_prompts = P_init_1.unsqueeze(0)
            deep_prompts = self.enc.transformer.deep_prompt_embeddings
            num_deep_layers = deep_prompts.shape[0]

            with torch.no_grad():
                for i in range(num_deep_layers):
                    output_prompts, _ = self.enc.transformer.encoder.layer[i](current_prompts)
                    deep_prompts.data[i].copy_(output_prompts.squeeze(0))
                    current_prompts = output_prompts
            logger.info("Initialized deep prompts.")



    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg,
            cfg.MODEL.MODEL_ROOT, adapter_cfg, load_pretrain, vis
        )

        # ---- Koopman parameter whitelist: parameters matching these keys are KLD-related ----
        def is_koopman_param(name: str) -> bool:
            lname = name.lower()
            # Covers:
            #   enc.transformer.koopman_in / koopman_out
            #   enc.transformer.K_layers.* / L_layers.*
            #   enc.transformer.K_global / L_global
            koop_keys = [
                "koop",      # koopman_in / koopman_out
                "k_layers",  # layerwise K
                "l_layers",  # layerwise L
                "k_global",  # global K
                "l_global",  # global L
            ]
            return any(k in lname for k in koop_keys)

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if (
                        f"transformer.encoder.layer.{total_layer - 1}" not in k
                        and "transformer.encoder.encoder_norm" not in k
                        and not is_koopman_param(k)
                ):
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if (
                        f"transformer.encoder.layer.{total_layer - 1}" not in k
                        and f"transformer.encoder.layer.{total_layer - 2}" not in k
                        and "transformer.encoder.encoder_norm" not in k
                        and not is_koopman_param(k)
                ):
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if (
                        f"transformer.encoder.layer.{total_layer - 1}" not in k
                        and f"transformer.encoder.layer.{total_layer - 2}" not in k
                        and f"transformer.encoder.layer.{total_layer - 3}" not in k
                        and f"transformer.encoder.layer.{total_layer - 4}" not in k
                        and "transformer.encoder.encoder_norm" not in k
                        and not is_koopman_param(k)
                ):
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            # Linear head or side-tuning: freeze encoder but allow Koopman params
            for k, p in self.enc.named_parameters():
                if not is_koopman_param(k):
                    p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if ("bias" not in k) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if (
                        "prompt" not in lname
                        and "embeddings.patch_embeddings.weight" not in k
                        and "embeddings.patch_embeddings.bias" not in k
                        and not is_koopman_param(k)
                ):
                    p.requires_grad = False

        elif transfer_type == "prompt":
            # Train only prompt + Koopman params, freeze rest of encoder
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if ("prompt" not in lname) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if ("prompt" not in lname) and ("bias" not in lname) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "prompt-noupdate":
            # Freeze entire encoder including Koopman
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if ("cls_token" not in k) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if ("cls_token" not in k) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if ("prompt" not in lname) and ("cls_token" not in k) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if ("prompt" not in lname) and ("cls_token" not in k) and (not is_koopman_param(k)):
                    p.requires_grad = False

        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                lname = k.lower()
                if ("adapter" not in lname) and (not is_koopman_param(k)):
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x, x
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class Swin(ViT):
    """Swin-related model."""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1, total_blocks - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(total_layer - 2) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION in ["below"]:
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "sidetune":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.weight" not in k  and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))
