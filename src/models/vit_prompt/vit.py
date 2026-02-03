import math
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
import numpy as np

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer
from ...utils import logging

logger = logging.get_logger("prompt_agnostic_evolution")

def _safe_eig(K: torch.Tensor):
    """
    Compatible wrapper for torch.linalg.eig / torch.eig.
    Forces computation on CPU with float64 to avoid MAGMA GPU errors.
    """
    K_cpu = K.detach().to("cpu", dtype=torch.float64)

    if hasattr(torch.linalg, "eig"):
        return torch.linalg.eig(K_cpu)

    # Legacy torch.eig fallback
    evals, evecs = torch.eig(K_cpu, eigenvectors=True)  # evals: (n,2), evecs: (n,n)
    eigvals = torch.view_as_complex(evals.contiguous())  # combine to complex (n,)
    return eigvals, evecs


def _safe_svd(K: torch.Tensor):
    """
    Compatible wrapper for torch.linalg.svd / torch.svd.
    Forces computation on CPU with float64.
    """
    K_cpu = K.detach().to("cpu", dtype=torch.float64)

    if hasattr(torch.linalg, "svd"):
        return torch.linalg.svd(K_cpu)

    U, S, V = torch.svd(K_cpu)
    Vh = V.transpose(-2, -1)
    return U, S, Vh


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
        else:
            raise ValueError("Unsupported prompt initiation scheme.")

        # ------------------- KLD Configuration -----------------------
        self.koopman_enabled = getattr(prompt_config, "KOOPMAN_ENABLED", False)
        self.koopman_dim = getattr(prompt_config, "KOOPMAN_DIM", config.hidden_size)

        # Koopman mode: "layerwise" (per-layer K) or "global" (shared K)
        self.koopman_mode = getattr(prompt_config, "KOOPMAN_MOD", "layerwise")

        if self.prompt_config.DEEP:
            num_koopman_layers = config.transformer["num_layers"] - 1
        else:
            num_koopman_layers = 1

        # Initialize Koopman operators based on mode
        if self.koopman_mode == "layerwise":
            # Per-layer independent Koopman operators
            self.K_layers = nn.ParameterList([
                nn.Parameter(torch.eye(self.koopman_dim))
                for _ in range(num_koopman_layers)
            ])
            self.L_layers = nn.ParameterList([
                nn.Parameter(torch.eye(self.koopman_dim))
                for _ in range(num_koopman_layers)
            ])
        elif self.koopman_mode == "global":
            # Global shared Koopman operator (paper default)
            self.K_global = nn.Parameter(torch.eye(self.koopman_dim))
            self.L_global = nn.Parameter(torch.eye(self.koopman_dim))
        else:
            raise ValueError(f"Unknown koopman_mode: {self.koopman_mode}")

        # Projection layers: hidden_size <-> koopman_dim
        self.koopman_in = nn.Linear(config.hidden_size, self.koopman_dim, bias=False)
        self.koopman_out = nn.Linear(self.koopman_dim, config.hidden_size, bias=False)

        # Loss weights
        self.koopman_weight = getattr(prompt_config, "KOOPMAN_WEIGHT", 0.0)
        self.lyapunov_weight = getattr(prompt_config, "LYAPUNOV_WEIGHT", 0.0)

        self.koopman_loss = 0.0
        self.lyapunov_loss = 0.0

        # Store trajectory data for modal energy analysis
        self.prompt_trajectories = []

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
                if self.koopman_mode == "layerwise":
                    for K in self.K_layers:
                        K.requires_grad_(True)
                    for L in self.L_layers:
                        L.requires_grad_(True)
                else:
                    self.K_global.requires_grad_(True)
                    self.L_global.requires_grad_(True)
                self.koopman_in.train()
                self.koopman_out.train()
        else:
            for module in self.children():
                module.train(mode)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.embeddings(x)
        p_emb = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings)).expand(B, -1, -1)
        x = torch.cat((x[:, :1, :], p_emb, x[:, 1:, :]), dim=1)
        return x

    def forward_deep_prompt(self, embedding_output, store_trajectory=False):
        """
        Forward pass with deep prompts and KLD regularization.
        
        Args:
            embedding_output: Input embeddings
            store_trajectory: If True, store prompt trajectories for analysis
        """
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        if store_trajectory:
            self.prompt_trajectories = []

        hidden_states, weights = self.encoder.layer[0](embedding_output)
        if self.encoder.vis:
            attn_weights.append(weights)

        for i in range(1, num_layers):
            if i <= self.deep_prompt_embeddings.shape[0]:
                P_i = self.deep_prompt_embeddings[i - 1]

                if self.koopman_enabled and i < self.deep_prompt_embeddings.shape[0]:
                    P_next = self.deep_prompt_embeddings[i]
                    layer_idx = i - 1

                    z_i = self.koopman_in(P_i)
                    z_next = self.koopman_in(P_next)

                    # Store trajectory data for analysis
                    if store_trajectory:
                        self.prompt_trajectories.append((layer_idx, z_i.detach().clone()))

                    # Select K/L matrices based on mode
                    if self.koopman_mode == "layerwise":
                        K = self.K_layers[layer_idx]
                        L = self.L_layers[layer_idx]
                    else:  # global
                        K = self.K_global
                        L = self.L_global

                    # ========== Koopman Consistency Loss (Eq. 8) ==========
                    z_next_pred = torch.matmul(z_i, K)
                    
                    # Numerically stable MSE
                    koopman_diff = z_next_pred - z_next
                    koopman_mse = torch.mean(koopman_diff ** 2)
                    self.koopman_loss += self.koopman_weight * koopman_mse

                    # ========== Lyapunov Stability Regularizer (Eq. 10) ==========
                    # Q = L^T L ensures positive semi-definite (Cholesky-style)
                    Q = torch.matmul(L.transpose(0, 1), L)
                    
                    # Numerical stability improvements:
                    # 1) Spectral normalization of Q to control max eigenvalue
                    Q_norm = torch.linalg.matrix_norm(Q, ord=2)
                    Q_stable = Q / (Q_norm + 1e-8)
                    
                    # 2) Conditional L2 normalization (preserve direction, control magnitude)
                    z_i_norm = torch.norm(z_i, dim=-1, keepdim=True) + 1e-8
                    z_next_norm = torch.norm(z_next_pred, dim=-1, keepdim=True) + 1e-8
                    
                    # Only normalize when norm > 1 to avoid numerical explosion
                    z_i_stable = z_i / z_i_norm.clamp(min=1.0)
                    z_next_stable = z_next_pred / z_next_norm.clamp(min=1.0)
                    
                    # V(z) = z @ Q @ z^T (Lyapunov function)
                    V_i = torch.sum(torch.matmul(z_i_stable, Q_stable) * z_i_stable, dim=-1)
                    V_next = torch.sum(torch.matmul(z_next_stable, Q_stable) * z_next_stable, dim=-1)
                    
                    # 3) Softplus instead of ReLU for smooth penalty
                    delta_V = V_next - V_i
                    temperature = 1.0
                    lv_penalty = torch.mean(torch.nn.functional.softplus(delta_V * temperature) / temperature)
                    
                    # 4) Soft saturation for gradient clipping
                    lv_penalty = torch.tanh(lv_penalty / 10.0) * 10.0
                    
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

    def forward(self, x, store_trajectory=False):
        self.reset_koopman_loss()

        embedding_output = self.incorporate_prompt(x)
        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(embedding_output, store_trajectory)
        else:
            encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

    # ==================== Spectral Analysis Methods ====================

    def compute_koopman_eigenanalysis(self):
        """Compute eigenvalues and eigenvectors of Koopman operators."""
        eigenvalues_list = []
        eigenvectors_list = []
        eigenvalues_magnitude = []
        spectral_radius = []
        singular_values_list = []

        with torch.no_grad():
            if self.koopman_mode == "layerwise":
                K_matrices = self.K_layers
            else:  # global
                K_matrices = [self.K_global] * len(self.K_layers) if hasattr(self, 'K_layers') else [self.K_global]

            for layer_idx, K in enumerate(K_matrices):
                # Eigendecomposition using compatible API
                eigenvalues, eigenvectors = _safe_eig(K)
                eigenvalues_list.append(eigenvalues.cpu())
                eigenvectors_list.append(eigenvectors.cpu())

                # Eigenvalue magnitudes
                eig_mag = torch.abs(eigenvalues).cpu()
                eigenvalues_magnitude.append(eig_mag)

                # Spectral radius
                spec_rad = torch.max(eig_mag).item()
                spectral_radius.append(spec_rad)

                # SVD using compatible API
                U, S, Vh = _safe_svd(K)
                singular_values_list.append(S.cpu())

                logger.info(f"Layer {layer_idx}: ρ(K)={spec_rad:.4f}, σ_max={S[0].item():.4f}")

        return {
            'eigenvalues': eigenvalues_list,
            'eigenvectors': eigenvectors_list,
            'eigenvalues_magnitude': eigenvalues_magnitude,
            'spectral_radius': spectral_radius,
            'singular_values': singular_values_list,
            'mode': self.koopman_mode
        }

    def analyze_modal_energy(self, layer_idx=0):
        """
        Analyze modal energy distribution.
        Projects stored trajectory data onto eigenvector basis.
        """
        if not hasattr(self, 'prompt_trajectories') or len(self.prompt_trajectories) == 0:
            logger.warning("No trajectory data stored. Run forward with store_trajectory=True first.")
            return None

        # Get K matrix for this layer
        if self.koopman_mode == "layerwise":
            K = self.K_layers[layer_idx]
        else:
            K = self.K_global

        # Find trajectory for this layer
        Z_i = None
        for idx, z in self.prompt_trajectories:
            if idx == layer_idx:
                Z_i = z
                break

        if Z_i is None:
            logger.warning(f"No trajectory data for layer {layer_idx}")
            return None

        with torch.no_grad():
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eig(K)

            # Project onto eigenvector basis: c = V^{-1} z
            modal_energies = []
            for token_idx in range(Z_i.shape[0]):
                z = Z_i[token_idx].unsqueeze(0)  # (1, koopman_dim)
                c = torch.matmul(z, eigenvectors)  # (1, koopman_dim)
                energy = torch.abs(c).squeeze() ** 2  # |c_j|^2
                modal_energies.append(energy.cpu())

            modal_energies = torch.stack(modal_energies)  # (num_tokens, koopman_dim)

            # Map back to hidden space
            modes_hidden = []
            for j in range(eigenvectors.shape[1]):
                mode_koopman = eigenvectors[:, j].real  # (koopman_dim,)
                mode_hidden = self.koopman_out(mode_koopman.unsqueeze(0))  # (1, hidden_size)
                modes_hidden.append(mode_hidden.squeeze().cpu())

            modes_hidden = torch.stack(modes_hidden)  # (koopman_dim, hidden_size)
            mode_norms = torch.norm(modes_hidden, dim=1)  # L2 norms

        return {
            'modal_energies': modal_energies,
            'modes_hidden': modes_hidden,
            'mode_norms': mode_norms,
            'eigenvalues': eigenvalues.cpu(),
            'mean_energy_per_mode': modal_energies.mean(dim=0)
        }

    def compute_spectral_alignment(self):
        """
        Compute cross-layer spectral alignment (principal angles).
        Only meaningful for layerwise mode.
        """
        if self.koopman_mode != "layerwise":
            logger.warning("Spectral alignment is only meaningful for layerwise mode")
            return None

        alignments = []
        with torch.no_grad():
            for i in range(len(self.K_layers) - 1):
                K1 = self.K_layers[i]
                K2 = self.K_layers[i + 1]

                _, V1 = _safe_eig(K1)
                _, V2 = _safe_eig(K2)

                # Handle both real and complex eigenvectors
                V1_real = V1.real if torch.is_complex(V1) else V1
                V2_real = V2.real if torch.is_complex(V2) else V2

                M = torch.matmul(V1_real.T, V2_real)

                _, S, _ = _safe_svd(M)
                principal_angles = torch.acos(torch.clamp(S, -1, 1))

                alignments.append({
                    'layer_pair': (i, i + 1),
                    'principal_angles': principal_angles.cpu(),
                    'mean_angle': principal_angles.mean().item(),
                    'alignment_score': S.mean().item()
                })

        return alignments


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


    def init_mpa(self, train_loader, window_size=16, stride=8):
        """
        Modal Pre-Alignment (MPA) initialization following the paper.
        
        Args:
            train_loader: DataLoader for training data
            window_size: Size of the sliding window (w in paper, default 16)
            stride: Stride of the sliding window (r in paper, default 8)
        """
        logger.info("Starting MPA initialization...")
        logger.info(f"   Window size w={window_size}, stride r={stride}")
        # Set model to eval mode, disable gradient computation
        self.eval()

        # Get a batch from the data loader
        try:
            x_batch, y_batch = next(iter(train_loader))
        except StopIteration:
            logger.error("Train loader is empty, cannot perform MPA.")
            return

        device = self.head.weight.device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        num_tokens = self.prompt_cfg.NUM_TOKENS  # T_p in paper
        B, C, H, W = x_batch.shape

        logger.info("MPA Phase I: Discovering Frequency Shortcuts...")

        # ========== Paper formula: sliding window frequency masks ==========
        # S = (floor((H-w)/r) + 1) * (floor((W-w)/r) + 1)
        w = window_size
        r = stride
        
        num_h = (H - w) // r + 1  # vertical window count
        num_w = (W - w) // r + 1  # horizontal window count
        S = num_h * num_w  # total candidate masks
        
        logger.info(f"   Generating {S} candidate masks ({num_h} x {num_w} grid)")
        
        # Generate all sliding window masks
        candidate_masks = torch.zeros(S, H, W, device=device)
        mask_idx = 0
        for i in range(num_h):
            for j in range(num_w):
                h_start = i * r
                w_start = j * r
                # Window position corresponds to specific frequency range after centering
                candidate_masks[mask_idx, h_start:h_start + w, w_start:w_start + w] = 1
                mask_idx += 1

        losses = torch.zeros(S, device=device)
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for s in range(S):
                mask = candidate_masks[s].view(1, 1, H, W)

                # Frequency domain filtering: Eq. (2) in paper
                x_freq = torch.fft.fftshift(torch.fft.fft2(x_batch), dim=(-2, -1))
                filtered_freq = x_freq * mask
                x_hat = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered_freq, dim=(-2, -1))))

                logits = self(x_hat)
                losses[s] = loss_fn(logits, y_batch)

        # Select top-T_p masks with lowest loss (most discriminative)
        _, top_indices = torch.topk(losses, num_tokens, largest=False)
        selected_masks = candidate_masks[top_indices]  # Shape: (T_p, H, W)
        
        # Log selected mask positions
        selected_positions = []
        for idx in top_indices:
            i = idx.item() // num_w
            j = idx.item() % num_w
            selected_positions.append((i * r, j * r))
        logger.info(f"Selected {num_tokens} frequency shortcut masks at positions: {selected_positions[:5]}...")

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

                # Energy-weighted pooling (Eq. 3-4 in paper)
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

        logger.info("MPA initialization finished.")

    # --------------------------------------------------------

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        # Extract CLS token
        x = x[:, 0]
        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights