import math
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer
from ...utils import logging

logger = logging.get_logger("prompt_agnostic_evolution")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        super().__init__(config, img_size, vis)

        assert prompt_config.LOCATION == "prepend"
        if prompt_config.INITIATION == "mpa":
            assert prompt_config.PROJECT == -1, "MPA initialization requires PROJECT to be -1"
        if prompt_config.DEEP:
            assert prompt_config.NUM_DEEP_LAYERS is None
            assert not prompt_config.DEEP_SHARED

        self.prompt_config = prompt_config
        self.vit_config = config

        patch_size = _pair(config.patches["size"])
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config.DROPOUT)

        if self.prompt_config.PROJECT > -1:
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out'
            )
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        if self.prompt_config.INITIATION == "random":
            logger.info("Initializing prompts randomly.")
            val = math.sqrt(
                6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim)
            )
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim)
            )
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim)
                )
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val
                )

        elif self.prompt_config.INITIATION == "mpa":
            logger.info("Prompts will be initialized by MPA. Make sure to call model.init_mpa(loader).")
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim)
            )
            if self.prompt_config.DEEP:
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim)
                )
        # ------------------------------------

        else:
            raise ValueError("Unsupported prompt initiation scheme.")


        self.koopman_enabled = getattr(prompt_config, "KOOPMAN_ENABLED", False)
        self.koopman_dim = getattr(prompt_config, "KOOPMAN_DIM", config.hidden_size)
        self.K = nn.Parameter(torch.eye(self.koopman_dim))

        self.koopman_in = nn.Linear(config.hidden_size, self.koopman_dim, bias=False)
        self.koopman_out = nn.Linear(self.koopman_dim, config.hidden_size, bias=False)


        self.L = nn.Parameter(torch.eye(self.koopman_dim))

        self.koopman_weight = getattr(prompt_config, "KOOPMAN_WEIGHT", 0.0)
        self.lyapunov_weight = getattr(prompt_config, "LYAPUNOV_WEIGHT", 0.0)

        self.koopman_loss = 0.0
        self.lyapunov_loss = 0.0

    def reset_koopman_loss(self):
        self.koopman_loss = 0.0
        self.lyapunov_loss = 0.0

    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
            if self.koopman_enabled:
                self.K.requires_grad_(True)
                self.L.requires_grad_(True)
                self.koopman_in.train()
                self.koopman_out.train()
        else:
            for module in self.children():
                module.train(mode)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.embeddings(x)  # (B, 1 + n_patches, hidden_dim)
        p_emb = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings)).expand(B, -1, -1)
        x = torch.cat((x[:, :1, :], p_emb, x[:, 1:, :]), dim=1)
        return x

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]
        # deep_prompt_embeddings形状: (num_layers-1, num_tokens, prompt_dim)


        hidden_states, weights = self.encoder.layer[0](embedding_output)
        if self.encoder.vis:
            attn_weights.append(weights)

        for i in range(1, num_layers):

            if i <= self.deep_prompt_embeddings.shape[0]:
                P_i = self.deep_prompt_embeddings[i - 1]  # (num_tokens, prompt_dim)

                if self.koopman_enabled and i < self.deep_prompt_embeddings.shape[0]:
                    P_next = self.deep_prompt_embeddings[i]  # (num_tokens, prompt_dim)

                    z_i = self.koopman_in(P_i)
                    z_next = self.koopman_in(P_next)

                    z_next_pred = torch.matmul(z_i, self.K)
                    self.koopman_loss += self.koopman_weight * torch.mean(
                        (z_next_pred - z_next) ** 2
                    )


                    Q = torch.matmul(self.L.transpose(0, 1), self.L)
                    V_i = torch.sum((torch.matmul(z_i, Q)) * z_i, dim=-1)  # (num_tokens,)
                    V_next = torch.sum((torch.matmul(z_next_pred, Q)) * z_next_pred, dim=-1)
                    lv_penalty = torch.mean(torch.relu(V_next - V_i))
                    self.lyapunov_loss += self.lyapunov_weight * lv_penalty

                p_emb = self.prompt_dropout(self.prompt_proj(P_i)).expand(B, -1, -1)
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    p_emb,
                    hidden_states[:, (1 + self.num_tokens):, :]
                ), dim=1)

            hidden_states, weights = self.encoder.layer[i](hidden_states)
            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        self.reset_koopman_loss()

        embedding_output = self.incorporate_prompt(x)
        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, prompt_cfg, model_type,
                 img_size=224, num_classes=21843, vis=False):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super().__init__(model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError(
                "prompt_cfg cannot be None if using PromptedVisionTransformer"
            )
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis
        )

    def init_mpa(self, train_loader, s_candidate_masks=512):

        logger.info("🚀 Starting MPA initialization...")
        # 确保模型处于评估模式，且梯度不被计算
        self.eval()

        # 1. 从数据加载器中获取一个批次的数据
        try:
            x_batch, y_batch = next(iter(train_loader))
        except StopIteration:
            logger.error("Train loader is empty, cannot perform MPA.")
            return

        device = self.head.weight.device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        num_tokens = self.prompt_cfg.NUM_TOKENS  # 即论文中的 T_p
        B, C, H, W = x_batch.shape


        logger.info("MPA Phase I: Discovering Frequency Shortcuts...")


        candidate_masks = torch.zeros(s_candidate_masks, H, W, device=device)
        for i in range(s_candidate_masks):
            ratio = 0.05 + 0.35 * torch.rand(1).item()
            h_k, w_k = int(H * ratio), int(W * ratio)
            h_start, w_start = (H - h_k) // 2, (W - w_k) // 2
            candidate_masks[i, h_start:h_start + h_k, w_start:w_start + w_k] = 1

        losses = torch.zeros(s_candidate_masks, device=device)
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for s in range(s_candidate_masks):
                mask = candidate_masks[s].view(1, 1, H, W)

                x_freq = torch.fft.fftshift(torch.fft.fft2(x_batch), dim=(-2, -1))
                filtered_freq = x_freq * mask
                x_hat = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered_freq, dim=(-2, -1))))

                logits = self(x_hat)
                losses[s] = loss_fn(logits, y_batch)


        _, top_indices = torch.topk(losses, num_tokens, largest=False)
        selected_masks = candidate_masks[top_indices]  # Shape: (T_p, H, W)
        logger.info(f"Selected {num_tokens} frequency shortcut masks.")

        logger.info("MPA Phase II: Initializing Prompts...")

        patch_embedder = self.transformer.embeddings
        representative_tokens = []
        with torch.no_grad():
            for r in range(num_tokens):
                mask = selected_masks[r].view(1, 1, H, W)


                x_freq = torch.fft.fftshift(torch.fft.fft2(x_batch), dim=(-2, -1))
                filtered_freq = x_freq * mask
                x_hat_r = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered_freq, dim=(-2, -1))))


                patch_tokens = patch_embedder(x_hat_r)[:, 1:, :]  # (B, N_p, D)


                energies = torch.sum(patch_tokens ** 2, dim=-1)  # (B, N_p)
                weights = energies / torch.sum(energies)  # (B, N_p)


                t_r = torch.sum(weights.unsqueeze(-1) * patch_tokens, dim=(0, 1))  # (D,)
                representative_tokens.append(t_r)

        P_init_1 = torch.stack(representative_tokens, dim=0)  # (T_p, D)

        self.transformer.prompt_embeddings.data.copy_(P_init_1.unsqueeze(0))
        logger.info("Initialized shallow prompts.")


        if self.prompt_cfg.DEEP:
            logger.info("Propagating prompts for deep layers...")
            current_prompts = P_init_1.unsqueeze(0)  # (1, T_p, D)
            num_deep_layers = self.transformer.deep_prompt_embeddings.shape[0]

            with torch.no_grad():
                for i in range(num_deep_layers):
                    output_prompts, _ = self.transformer.encoder.layer[i](current_prompts)

                    self.transformer.deep_prompt_embeddings.data[i].copy_(output_prompts.squeeze(0))

                    current_prompts = output_prompts
            logger.info("Initialized deep prompts.")

        logger.info("✅ MPA initialization finished.")

    # --------------------------------------------------------

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        # 取CLS token
        x = x[:, 0]
        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights