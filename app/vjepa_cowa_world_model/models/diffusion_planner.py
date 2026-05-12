# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Diffusion-based Planner for trajectory prediction.

Adapted from XTR's DiT (Diffusion Transformer) architecture.
Uses VP-SDE for the forward diffusion process and DPM-Solver++ for fast sampling.

Key differences from XTR:
- Cross-attention conditioning comes from z_ar tokens (ViT encoder/predictor output)
  instead of XTR's scene features (actors + map + static objects).
- Single ego agent only (no multi-agent attention mask).
- Trajectory is [x, y, vx, vy, cos_yaw, sin_yaw] (6-dim per timestep).
- Multi-modality achieved via K noise samples, not built-in K modes.
- Output interface matches MultiModalTemporalPlanner:
  {"trajectories": [B, K, num_poses, 3], "confidences": [B, K]}
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion_utils import dpm_solver_pytorch as dpm
from ..diffusion_utils.sampling import dpm_sampler
from ..diffusion_utils.sde import VPSDE_linear

# ============================================================
# Building blocks (adapted from XTR / timm)
# ============================================================


class Mlp(nn.Module):
    """Simple MLP with GELU activation (replaces timm.models.layers.Mlp)."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if act_layer is None:
            act_layer = nn.GELU
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning + self-attention + cross-attention.

    Adapted from XTR: removed neighbor_current_mask (single ego agent).
    """

    def __init__(self, dim=256, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)

    def forward(self, x, cross_c, y):
        """
        Args:
            x: [B, L, D] — trajectory tokens
            cross_c: [B, N, D] — conditioning tokens (projected z_ar)
            y: [B, D] — conditioning vector (timestep + status)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).unsqueeze(1).chunk(6, dim=-1)
        )

        # Self-attention with adaLN-Zero
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(modulated_x, modulated_x, modulated_x)[0]

        # FFN with adaLN-Zero
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        # Cross-attention to conditioning tokens
        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = x + self.mlp2(self.norm4(x))

        return x


class DitBlockV3(nn.Module):
    """Legacy-shaped DiT block without residual additions on cross-attn and mlp2."""

    def __init__(self, dim=256, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)

    def forward(self, x, cross_c, y):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).unsqueeze(1).chunk(6, dim=-1)
        )

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(modulated_x, modulated_x, modulated_x)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x


class DiTBlockV2(nn.Module):
    """DiT block with full adaLN-Zero on self-attn, cross-attn, and MLP."""

    def __init__(self, dim=256, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_cond = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))

    def forward(self, x, cross_c, y):
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).unsqueeze(1).chunk(9, dim=-1)
        )

        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_sa * attn_out

        x_norm_ca = modulate(self.norm_cross(x), shift_ca, scale_ca)
        cond_kv = self.norm_cond(cross_c)
        ca_out, _ = self.cross_attn(x_norm_ca, cond_kv, cond_kv)
        x = x + gate_ca * ca_out

        x_norm_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm_mlp)

        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT: adaLN modulation + optional multi-modal projection + regression.

    When ``num_modes > 1`` (XTR-aligned):
        Splits hidden state into K modes via ``proj``, produces per-mode
        classification scores (``cls``) and per-mode trajectory regression (``reg``).
    When ``num_modes == 1`` (legacy):
        Outputs the denoised trajectory directly (no classification head).

    ``mode_token_expansion`` (opt-in, ``num_modes > 1`` only):
        Input is already K×TF tokens carrying mode identity (set upstream by
        :class:`TrajectoryDiT`).  ``proj`` is bypassed — regression acts on
        every token directly, and classification is obtained by pooling each
        mode's TF tokens.  Output layout matches the legacy per_pose_token
        path so downstream ``_training_forward_multimodal`` / DPM sampling do
        not need to change.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        num_modes=1,
        squeeze_dim=True,
        mode_token_expansion=False,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.squeeze_dim = squeeze_dim
        self.mode_token_expansion = bool(mode_token_expansion) and num_modes > 1
        self.norm_final = nn.LayerNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        if num_modes > 1:
            if not self.mode_token_expansion:
                # XTR-style: project to K independent mode representations
                self.proj = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size * num_modes, bias=True),
                )
            else:
                # K already lives in the sequence dimension — proj is redundant.
                self.proj = None
            # Classification head: per-mode confidence score
            self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.proj = None
            self.cls = None

        self.reg = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )

    def forward(self, x, y):
        """
        Args:
            x: [B, L, D] — trajectory token(s). ``L`` equals ``K*TF`` when
               ``mode_token_expansion=True`` (tokens ordered mode-major so
               ``view(B, K, TF, D)`` recovers per-mode streams).
            y: [B, D] — conditioning vector

        Returns
        -------
        (x_cls, x_reg):
            num_modes > 1:
                x_cls [B, K] raw confidence logits
                x_reg [B, K * output_size] (squeeze_dim + L==1)
                   or [B, K * L * output_size] (per_pose_token)
            num_modes == 1:
                x_cls None
                x_reg [B, output_size] or [B, L, output_size]
        """
        shift, scale = self.adaLN_modulation(y).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        if self.num_modes > 1:
            K = self.num_modes
            if self.mode_token_expansion:
                # x: [B, K*TF, D] — tokens ordered mode-major.
                B, KTF, D = x.shape
                assert KTF % K == 0, f"sequence length {KTF} not divisible by K={K}"
                TF = KTF // K
                x_per_mode = x.view(B, K, TF, D)
                # cls: pool each mode's TF tokens → [B, K, D] → cls → [B, K]
                x_cls = self.cls(x_per_mode.mean(dim=2)).squeeze(-1)
                # reg: per-token regression → [B, K*TF, output_size] → flatten
                x_reg = self.reg(x).view(B, K, TF, -1).reshape(B, -1)
                return x_cls, x_reg

            B, L, _ = x.shape
            x = self.proj(x).view(B, L, K, -1)  # [B, L, K, D]
            x_cls = self.cls(x).squeeze(-1)  # [B, L, K] — raw logits
            x_reg = self.reg(x)  # [B, L, K, output_size]
            if self.squeeze_dim and L == 1:
                x_cls = x_cls.squeeze(1)  # [B, K]
                x_reg = x_reg.squeeze(1).reshape(B, -1)  # [B, K * output_size]
            else:
                # per_pose_token: average cls across poses → [B, K]
                x_cls = x_cls.mean(dim=1)
                # reorder to (K, L, out) then flatten for DPM solver
                x_reg = x_reg.permute(0, 2, 1, 3).reshape(B, -1)
            return x_cls, x_reg
        else:
            x = self.reg(x)
            if self.squeeze_dim:
                return None, x.squeeze(1)
            return None, x


# ============================================================
# Core DiT network for trajectory denoising
# ============================================================


class TrajectoryDiT(nn.Module):
    """
    Diffusion Transformer for ego trajectory denoising.

    Takes a noisy flattened trajectory [B, num_poses * 6] and diffusion time [B,],
    conditioned on context tokens [B, N, D] and status vector [B, D].
    Predicts the clean trajectory (x_start prediction).
    """

    def __init__(
        self,
        num_poses: int,
        traj_dim: int = 6,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        trajectory_token_mode: str = "single_token",
        num_modes: int = 1,
        use_anchor_frame: bool = False,
        adaln_version: str = "legacy",
        mode_token_expansion: bool = False,
    ):
        super().__init__()
        self.num_poses = num_poses
        self.traj_dim = traj_dim
        self.num_modes = num_modes
        self.use_anchor_frame = use_anchor_frame
        self._model_type = "x_start"
        self.trajectory_token_mode = trajectory_token_mode
        self.adaln_version = adaln_version
        # Opt-in: expand K modes into the sequence dimension with a learnable
        # mode embedding so every DiT layer can diversify modes via self-
        # attention.  Only active for per_pose_token with num_modes > 1.
        self.mode_token_expansion = (
            bool(mode_token_expansion) and num_modes > 1 and trajectory_token_mode == "per_pose_token"
        )
        if bool(mode_token_expansion) and not self.mode_token_expansion:
            raise ValueError(
                "mode_token_expansion=True requires trajectory_token_mode='per_pose_token' "
                f"and num_modes>1 (got token_mode={trajectory_token_mode!r}, num_modes={num_modes})"
            )

        if self.trajectory_token_mode not in {"single_token", "per_pose_token"}:
            raise ValueError(
                f"Unsupported trajectory_token_mode={self.trajectory_token_mode!r}; "
                "expected one of {'single_token', 'per_pose_token'}"
            )

        if self.adaln_version not in {"legacy", "v2", "v3"}:
            raise ValueError(
                f"Unsupported adaln_version={self.adaln_version!r}; expected one of {'legacy', 'v2', 'v3'}"
            )

        # When use_anchor_frame=True, the network processes (1 + num_poses)
        # frames: the first frame is the clean current ego state (anchor) and
        # the remaining are (noisy) future poses.  This follows the XTR design
        # which provides an explicit starting-point constraint.
        self.total_frames = num_poses + 1 if use_anchor_frame else num_poses

        # When num_modes > 1, the input contains K modes concatenated.
        in_dim = self.total_frames * traj_dim * num_modes
        output_dim = self.total_frames * traj_dim

        if self.trajectory_token_mode == "single_token":
            # Flatten the full (multi-modal) trajectory into one latent token.
            self.preproj = Mlp(
                in_features=in_dim,
                hidden_features=min(512, hidden_dim * 2),
                out_features=hidden_dim,
                act_layer=nn.GELU,
                drop=0.0,
            )
            self.pose_embed = None
            self.mode_embed = None
            self.final_layer = FinalLayer(hidden_dim, output_dim, num_modes=num_modes, squeeze_dim=True)
        else:
            # Tokenized design: one token per frame (including anchor if used).
            if self.mode_token_expansion:
                # NEW path: K modes expanded into sequence dimension.
                # preproj takes a single mode's pose and lifts it to hidden_dim;
                # DiT then attends over K*TF tokens with explicit mode identity.
                self.preproj = Mlp(
                    in_features=traj_dim,
                    hidden_features=min(256, hidden_dim * 2),
                    out_features=hidden_dim,
                    act_layer=nn.GELU,
                    drop=0.0,
                )
                self.pose_embed = nn.Parameter(torch.zeros(1, self.total_frames, hidden_dim))
                self.mode_embed = nn.Parameter(torch.zeros(1, num_modes, 1, hidden_dim))
                self.final_layer = FinalLayer(
                    hidden_dim,
                    traj_dim,
                    num_modes=num_modes,
                    squeeze_dim=False,
                    mode_token_expansion=True,
                )
            else:
                # Legacy path: K modes fused at input via a single MLP, then
                # DiT sees TF tokens and FinalLayer.proj splits into K at the
                # end.  Kept to preserve checkpoint compatibility.
                self.preproj = Mlp(
                    in_features=traj_dim * num_modes,
                    hidden_features=min(256, hidden_dim * 2),
                    out_features=hidden_dim,
                    act_layer=nn.GELU,
                    drop=0.0,
                )
                self.pose_embed = nn.Parameter(torch.zeros(1, self.total_frames, hidden_dim))
                self.mode_embed = None
                self.final_layer = FinalLayer(hidden_dim, traj_dim, num_modes=num_modes, squeeze_dim=False)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # DiT blocks
        if self.adaln_version == "v2":
            block_cls = DiTBlockV2
        elif self.adaln_version == "v3":
            block_cls = DitBlockV3
        else:
            block_cls = DiTBlock
        self.blocks = nn.ModuleList([block_cls(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)])

        self._initialize_weights()

    @property
    def model_type(self):
        return self._model_type

    def _initialize_weights(self):
        """Initialize weights following DiT conventions."""
        # Initialize adaLN modulation layers to zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Initialize final regression to zero (so initial output = 0 = identity)
        nn.init.constant_(self.final_layer.reg[-1].weight, 0)
        nn.init.constant_(self.final_layer.reg[-1].bias, 0)
        if self.pose_embed is not None:
            nn.init.normal_(self.pose_embed, std=0.02)
        if getattr(self, "mode_embed", None) is not None:
            nn.init.normal_(self.mode_embed, std=0.02)

    def forward(self, x, t, cross_c, status_emb):
        """
        Args:
            x: [B, flat_dim] — noisy flattened trajectory.
               flat_dim = num_modes * num_poses * traj_dim  when num_modes > 1
               flat_dim = num_poses * traj_dim               when num_modes == 1
            t: [B,] — diffusion timestep in [eps, 1]
            cross_c: [B, N, hidden_dim] — projected context tokens
            status_emb: [B, hidden_dim] — projected status feature

        Returns:
            (x_cls, x_pred):
                num_modes > 1: x_cls [B, K] raw logits, x_pred [B, K*num_poses*traj_dim]
                num_modes == 1: x_cls None, x_pred [B, num_poses*traj_dim]
        """
        B = x.shape[0]
        K = self.num_modes
        TF = self.total_frames  # num_poses (+1 if anchor)

        if self.trajectory_token_mode == "single_token":
            # Flatten the full (multi-modal) trajectory into one latent token.
            x = self.preproj(x).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            # Tokenized design: one token per frame (including anchor).
            if self.mode_token_expansion and K > 1:
                # NEW path: expand K modes into sequence dim with mode identity.
                # Input: [B, K * TF * traj_dim] → [B, K, TF, traj_dim]
                # preproj: [B, K, TF, hidden_dim]  (shared MLP over every (k,t))
                # + pose_embed broadcast over K, + mode_embed broadcast over TF
                # → flatten mode-major so [B, K*TF, hidden_dim]
                x = x.view(B, K, TF, self.traj_dim)
                x = self.preproj(x)
                x = x + self.mode_embed + self.pose_embed.unsqueeze(1)
                x = x.reshape(B, K * TF, -1)
            elif K > 1:
                # Legacy: [B, K * TF * traj_dim]
                # → [B, K, TF, traj_dim] → [B, TF, K * traj_dim]
                x = x.view(B, K, TF, self.traj_dim)
                x = x.permute(0, 2, 1, 3).reshape(B, TF, K * self.traj_dim)
                x = self.preproj(x) + self.pose_embed  # [B, TF, hidden_dim]
            else:
                x = x.view(B, TF, self.traj_dim)
                x = self.preproj(x) + self.pose_embed  # [B, TF, hidden_dim]

        # Conditioning vector: timestep + status
        y = self.t_embedder(t) + status_emb  # [B, hidden_dim]

        # DiT blocks with cross-attention to context
        for block in self.blocks:
            x = block(x, cross_c, y)

        # Final layer returns (cls, reg) tuple
        x_cls, x_pred = self.final_layer(x, y)

        # For legacy per_pose_token with num_modes==1, FinalLayer returns
        # [B, L, traj_dim]; flatten to [B, total_frames * traj_dim].
        if K == 1 and self.trajectory_token_mode != "single_token":
            x_pred = x_pred.reshape(B, -1)

        return x_cls, x_pred


# ============================================================
# TrajectoryConfidenceHead: post-hoc per-mode scoring
# ============================================================


class TrajectoryConfidenceHead(nn.Module):
    """Estimates per-mode trajectory quality given context.

    Takes K denoised trajectories + pooled context tokens, outputs K raw
    confidence logits.  Used only when ``independent_modes=True`` in
    :class:`DiffusionPlanner`.
    """

    def __init__(self, traj_dim: int = 6, num_poses: int = 10, context_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.context_dim = context_dim
        self.traj_dim = traj_dim
        self._has_velocity = traj_dim >= 6
        self._yaw_start = 2 if traj_dim == 4 else 4
        # Group-wise LayerNorm: xy(0:2), [vel(2:4)], yaw(start:start+2)
        self.xy_norm = nn.LayerNorm(2)
        if self._has_velocity:
            self.vel_norm = nn.LayerNorm(2)
        self.yaw_norm = nn.LayerNorm(2)
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_dim * num_poses, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention pooling: learnable query attends over N context tokens
        self.context_query = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajs: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajs: [B, K, num_poses, traj_dim] — K denoised trajectories
            context_tokens: [B, N, context_dim] — scene context

        Returns:
            [B, K] raw confidence logits
        """
        B, K, T, D = trajs.shape
        x = trajs.reshape(B * K * T, D)
        # Group-wise norm: each semantic group keeps its own statistics
        parts = [self.xy_norm(x[:, 0:2])]
        if self._has_velocity:
            parts.append(self.vel_norm(x[:, 2:4]))
        ys = self._yaw_start
        parts.append(self.yaw_norm(x[:, ys : ys + 2]))
        x = torch.cat(parts, dim=1).reshape(B * K, T * D)
        traj_feat = self.traj_encoder(x)

        # Attention pooling over context tokens instead of naive mean
        query = self.context_query.expand(B, -1, -1)  # [B, 1, context_dim]
        scale = self.context_dim**0.5
        attn = F.softmax(torch.bmm(query, context_tokens.transpose(1, 2)) / scale, dim=-1)  # [B, 1, N]
        ctx_feat = torch.bmm(attn, context_tokens).squeeze(1)  # [B, context_dim]

        ctx_feat = ctx_feat.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        return self.scorer(torch.cat([traj_feat, ctx_feat], dim=-1)).view(B, K)


# ============================================================
# DiffusionPlanner: the full planner module
# ============================================================


class DiffusionPlanner(nn.Module):
    """
    Diffusion-based planner for trajectory prediction.

    Wraps TrajectoryDiT with VP-SDE forward process and DPM-Solver++ sampling.
    Supports both training (returns diffusion loss) and inference (returns K sampled trajectories).

    Output interface matches MultiModalTemporalPlanner:
        {"trajectories": [B, K, num_poses, 3], "confidences": [B, K]}

    Training mode (gt_trajectory is provided):
        Returns loss dict: {"loss", "reg_loss", "conf_loss", "cover_loss"}

    Inference mode (gt_trajectory is None):
        Returns prediction dict: {"trajectories", "confidences"}
    """

    def __init__(
        self,
        encoder_dim: int = 1408,
        num_poses: int = 7,
        status_dim: int = 7,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        traj_dim: int = 6,
        # SDE parameters
        sde_beta_min: float = 0.1,
        sde_beta_max: float = 20.0,
        # Inference parameters
        num_samples: int = 6,
        inference_steps: int = 2,
        # Input options
        use_z_context: bool = False,
        tokens_per_frame: int = 256,
        trajectory_token_mode: str = "single_token",
        use_last_frame_only: bool = True,
        use_action_history: bool = False,
        action_history_dim: int = 3,
        num_observed_frames: int = 1,
        # Multi-modal parameters (XTR-aligned, built-in K modes)
        num_modes: int = 1,
        # Independent mode processing: B->B*K batch expansion (anti-collapse)
        independent_modes: bool = False,
        # Anchor frame: prepend current ego state to trajectory (XTR-aligned)
        use_anchor_frame: bool = False,
        # Loss weights (effective when num_modes > 1)
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        vel_loss_weight: float = 0.5,
        yaw_loss_weight: float = 0.5,
        reg_timestep_weights: Optional[torch.Tensor] = None,
        # Hybrid WTA hyperparams: aWTA regression + XTR-style gated soft-CE
        awta_init_temperature: float = 8.0,
        awta_min_temperature: float = 0.1,
        conf_temperature: float = 1.5,
        cls_th: float = 2.0,
        cls_ignore: float = 0.2,
        command_dim: int = 0,
        adaln_version: str = "legacy",
        mode_token_expansion: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.num_poses = num_poses
        self.traj_dim = traj_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.num_modes = num_modes
        self.independent_modes = independent_modes
        self.inference_steps = inference_steps
        self.use_z_context = use_z_context
        self.tokens_per_frame = tokens_per_frame
        self.trajectory_token_mode = trajectory_token_mode
        self.use_last_frame_only = use_last_frame_only
        self.use_action_history = use_action_history
        self.action_history_dim = action_history_dim
        self.num_observed_frames = num_observed_frames
        self.use_anchor_frame = use_anchor_frame
        self.total_frames = num_poses + 1 if use_anchor_frame else num_poses
        self.command_dim = command_dim
        self.adaln_version = adaln_version
        self.mode_token_expansion = bool(mode_token_expansion)

        # Loss weights (for num_modes > 1 WTA training)
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.yaw_loss_weight = yaw_loss_weight
        if reg_timestep_weights is not None:
            reg_timestep_weights = torch.as_tensor(reg_timestep_weights, dtype=torch.float32)
            if reg_timestep_weights.ndim != 1:
                raise ValueError(f"reg_timestep_weights must be 1D, got shape {tuple(reg_timestep_weights.shape)}")
            if reg_timestep_weights.numel() != num_poses:
                raise ValueError(
                    f"reg_timestep_weights must have length num_poses={num_poses}, "
                    f"got {reg_timestep_weights.numel()}"
                )
        self.register_buffer("reg_timestep_weights", reg_timestep_weights, persistent=False)
        # Hybrid WTA hyperparams
        self.awta_min_temperature = float(awta_min_temperature)
        self.conf_temperature = float(conf_temperature)
        self.cls_th = float(cls_th)
        self.cls_ignore = float(cls_ignore)
        # Runtime-mutable aWTA temperature — external epoch scheduler calls
        # ``set_awta_temperature`` once per epoch to anneal it down.
        self.register_buffer(
            "awta_temperature",
            torch.tensor(float(awta_init_temperature)),
            persistent=False,
        )

        # Context projection: encoder tokens → hidden_dim
        self.context_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Status projection: status_feature → hidden_dim
        if command_dim > 0:
            kinematics_dim = status_dim - command_dim
            self.command_proj = nn.Sequential(
                nn.Linear(command_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.kinematics_proj = nn.Sequential(
                nn.LayerNorm(kinematics_dim),
                nn.Linear(kinematics_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.status_proj = nn.Sequential(
                nn.Linear(status_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if use_action_history:
            self.action_history_proj = nn.Sequential(
                nn.LayerNorm(action_history_dim),
                nn.Linear(action_history_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_history_pos_embedding = nn.Embedding(num_observed_frames, hidden_dim)

        # The denoising network — single-mode when independent_modes=True
        dit_num_modes = 1 if independent_modes else num_modes
        # mode_token_expansion only makes sense when the DiT itself is multi-mode.
        dit_mode_token_expansion = bool(mode_token_expansion) and dit_num_modes > 1
        if bool(mode_token_expansion) and not dit_mode_token_expansion:
            raise ValueError(
                "mode_token_expansion=True requires the inner DiT to be multi-mode "
                f"(num_modes>1 and independent_modes=False). Got num_modes={num_modes}, "
                f"independent_modes={independent_modes}."
            )
        self.dit = TrajectoryDiT(
            num_poses=num_poses,
            traj_dim=traj_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            trajectory_token_mode=trajectory_token_mode,
            num_modes=dit_num_modes,
            use_anchor_frame=use_anchor_frame,
            adaln_version=adaln_version,
            mode_token_expansion=dit_mode_token_expansion,
        )

        # Confidence head for independent mode: post-hoc per-mode scoring
        if independent_modes and num_modes > 1:
            self.confidence_head = TrajectoryConfidenceHead(
                traj_dim=traj_dim,
                num_poses=num_poses,
                context_dim=hidden_dim,
                hidden_dim=128,
            )
        else:
            self.confidence_head = None

        # VP-SDE for forward diffusion
        self.sde = VPSDE_linear(beta_max=sde_beta_max, beta_min=sde_beta_min)

    def _get_reg_timestep_weights(
        self, num_poses: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        weights = getattr(self, "reg_timestep_weights", None)
        if weights is None:
            return None
        weights = weights.to(device=device, dtype=dtype)
        if weights.ndim != 1 or weights.numel() != num_poses:
            raise ValueError(f"reg_timestep_weights must have shape [{num_poses}], got {tuple(weights.shape)}")
        return weights

    def _xy_regression_loss_per_mode(self, pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
        """Return smooth-L1 XY regression loss reduced over timestep and XY dims."""
        per_step_loss = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none").sum(dim=-1)
        num_future = max(per_step_loss.shape[-1], 1)
        weights = self._get_reg_timestep_weights(num_future, per_step_loss.device, per_step_loss.dtype)
        if weights is not None:
            view_shape = (1,) * (per_step_loss.ndim - 1) + (num_future,)
            per_step_loss = per_step_loss * weights.view(view_shape)
        return per_step_loss.sum(dim=-1) / num_future

    # ------------------------------------------------------------------
    # Helper properties for batch-expansion vs joint-mode dispatch
    # ------------------------------------------------------------------

    @property
    def _uses_batch_expansion(self):
        """True when modes are processed via B->B*K batch expansion."""
        return self.independent_modes or self.num_modes == 1

    @property
    def _batch_K(self):
        """Effective K for batch expansion (independent_modes or legacy)."""
        return self.num_modes if self.independent_modes else self.num_samples

    def _prepare_context(self, z_ar, z_context=None, z_observed=None, action_history=None):
        """
        Project encoder/predictor tokens to hidden_dim for cross-attention.

        Args:
            z_ar: [B, T*P, encoder_dim] — predictor output tokens
            z_context: [B, P, encoder_dim] — optional first-frame encoder tokens
            z_observed: [B, T_obs*P, encoder_dim] — optional observed-frame encoder tokens
            action_history: [B, T_obs, action_history_dim] — optional observed history tokens

        Returns:
            context_tokens: [B, N, hidden_dim]
        """
        if self.use_z_context and z_context is not None:
            # Use only first-frame context (like train_context / train_command).
            # z_context is already a single frame — no truncation needed.
            tokens = z_context
        else:
            # When use_last_frame_only=True, keep only the last frame of z_ar.
            # Predictor is frame-causal (RoPE + causal mask), so the last frame
            # has attended to all prior frames and serves as a compressed
            # summary.  This reduces cross-attention cost from O(T*P) to O(P).
            ar_tokens = z_ar
            if self.use_last_frame_only and ar_tokens.size(1) > self.tokens_per_frame:
                ar_tokens = ar_tokens[:, -self.tokens_per_frame :]

            if z_observed is not None:
                # z_observed comes from the encoder (non-causal), so all its
                # tokens are independently useful — always keep them intact.
                tokens = torch.cat([z_observed, ar_tokens], dim=1)
            else:
                tokens = ar_tokens

        context_tokens = self.context_proj(tokens)
        if self.use_action_history:
            context_tokens = torch.cat([context_tokens, self._prepare_action_history(action_history)], dim=1)
        return context_tokens

    def _prepare_action_history(self, action_history):
        if not self.use_action_history:
            raise RuntimeError("_prepare_action_history() called when use_action_history=False")
        assert action_history is not None, "use_action_history=True but action_history is None"
        assert action_history.ndim == 3, f"Expected action_history shape [B, T_obs, D], got ndim={action_history.ndim}"
        assert (
            action_history.shape[-1] == self.action_history_dim
        ), f"action_history dim mismatch: got D={action_history.shape[-1]}, expected {self.action_history_dim}"
        assert (
            action_history.shape[1] <= self.num_observed_frames
        ), f"action_history length mismatch: got T={action_history.shape[1]}, expected <= {self.num_observed_frames}"

        tokens = self.action_history_proj(action_history)
        num_tokens = tokens.shape[1]
        tokens = tokens + self.action_history_pos_embedding.weight[:num_tokens].unsqueeze(0)
        return tokens

    def _prepare_status(self, status_feature):
        """Project status feature to hidden_dim.

        When command_dim > 0, splits status into categorical (command) and
        continuous (kinematics) portions, projects them independently, then
        sums the embeddings to produce a single hidden_dim vector.
        """
        if self.command_dim > 0:
            cmd_emb = self.command_proj(status_feature[:, : self.command_dim])
            kin_emb = self.kinematics_proj(status_feature[:, self.command_dim :])
            return cmd_emb + kin_emb
        return self.status_proj(status_feature)

    @property
    def _yaw_slice(self):
        """cos_yaw, sin_yaw indices in trajectory vector."""
        return slice(2, 4) if self.traj_dim == 4 else slice(4, 6)

    @property
    def _has_velocity(self):
        return self.traj_dim >= 6

    def _get_anchor(self, anchor_state: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        """
        Get the anchor frame (current ego state) for trajectory anchoring.

        When ``anchor_state`` is provided, it is used directly.
        Otherwise a default anchor at the ego-relative origin is created:
        4D: ``[0, 0, 1, 0]`` — (x=0, y=0, cos(0)=1, sin(0)=0)
        6D: ``[0, 0, 0, 0, 1, 0]`` — (x=0, y=0, vx=0, vy=0, cos(0)=1, sin(0)=0)

        Args:
            anchor_state: [B, traj_dim] or [B, 1, traj_dim] or None
            B: batch size
            device: target device

        Returns:
            [B, 1, traj_dim] anchor frame
        """
        if anchor_state is not None:
            a = anchor_state
            if a.dim() == 2:
                a = a.unsqueeze(1)
            return a
        anchor = torch.zeros(B, 1, self.traj_dim, device=device)
        anchor[..., self._yaw_slice.start] = 1.0  # cos(0) = 1
        return anchor

    def init_interleaved_inference_state(
        self,
        status_feature: torch.Tensor,
        total_condition_updates: int,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize a stateful diffusion sampler for predictor-interleaved inference.

        This mode is inference-only. The denoiser state x_t is carried across
        predictor rollout updates, and each newly available predictor prefix can
        advance the diffusion process by a subset of the solver steps.
        """
        if total_condition_updates <= 0:
            raise ValueError(f"total_condition_updates must be positive, got {total_condition_updates}")

        B = status_feature.shape[0]
        device = status_feature.device
        status_emb = self._prepare_status(status_feature)

        if not self._uses_batch_expansion:
            # Built-in K modes: no batch expansion needed
            K = self.num_modes
            if self.use_anchor_frame:
                anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
                anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, 1, 6]
                noise = torch.randn(B, K, self.num_poses, self.traj_dim, device=device)
                x_t = torch.cat([anchor_k, noise], dim=-2).reshape(B, -1)
            else:
                flat_dim = K * self.num_poses * self.traj_dim
                x_t = torch.randn(B, flat_dim, device=device)
            status_k = status_emb  # [B, hidden_dim]
        else:
            # Legacy or independent modes: K noise samples with batch expansion
            K = self._batch_K
            if self.use_anchor_frame:
                anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
                anchor_exp = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, self.traj_dim)
                noise = torch.randn(B * K, self.num_poses, self.traj_dim, device=device)
                x_t = torch.cat([anchor_exp, noise], dim=-2).reshape(B * K, -1)
            else:
                flat_dim = self.num_poses * self.traj_dim
                x_t = torch.randn(B * K, flat_dim, device=device)
            status_k = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        noise_schedule = dpm.NoiseScheduleVP(schedule="linear")
        dummy_solver = dpm.DPM_Solver(lambda x, t: (None, x), noise_schedule, algorithm_type="dpmsolver++")
        timesteps = dummy_solver.get_time_steps(
            skip_type="logSNR",
            t_T=noise_schedule.T,
            t_0=1.0 / noise_schedule.total_N,
            N=self.inference_steps,
            device=device,
        )

        # Store anchor info for correction during interleaved steps
        anchor_info = None
        if self.use_anchor_frame:
            if not self._uses_batch_expansion:
                anchor_info = anchor_k  # [B, K, 1, 6]
            else:
                anchor_info = anchor_exp  # [B*K, 1, 6]

        return {
            "x_t": x_t,
            "status_k": status_k,
            "noise_schedule": noise_schedule,
            "timesteps": timesteps,
            "batch_size": B,
            "total_condition_updates": total_condition_updates,
            "completed_condition_updates": 0,
            "completed_sampling_steps": 0,
            "anchor_info": anchor_info,
        }

    def _interleaved_target_sampling_steps(
        self,
        completed_condition_updates: int,
        total_condition_updates: int,
    ) -> int:
        return (completed_condition_updates * self.inference_steps) // total_condition_updates

    def _run_interleaved_solver_step(
        self,
        x_t: torch.Tensor,
        context_k: torch.Tensor,
        status_k: torch.Tensor,
        noise_schedule: dpm.NoiseScheduleVP,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self.dit.model_type,
            model_kwargs={
                "cross_c": context_k,
                "status_emb": status_k,
            },
        )
        # model_fn returns (cls, noise) tuple; solver arithmetic expects a
        # plain tensor.  Pre-compute model_s and pass it explicitly so that
        # dpm_solver_first_update skips the internal model_fn call.
        _, model_s = model_fn(x_t, s)
        solver = dpm.DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
        return solver.dpm_solver_first_update(x_t, s, t, model_s=model_s)

    @torch.no_grad()
    def advance_interleaved_inference(
        self,
        state: Dict[str, torch.Tensor],
        z_ar: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance the interleaved sampler using the latest predictor prefix.

        One call corresponds to one newly available predictor future frame.
        Depending on the ratio between predictor updates and diffusion steps,
        this call may execute 0, 1, or multiple first-order solver substeps.
        """
        completed_condition_updates = int(state["completed_condition_updates"])
        total_condition_updates = int(state["total_condition_updates"])
        if completed_condition_updates >= total_condition_updates:
            return state

        next_condition_updates = completed_condition_updates + 1
        target_sampling_steps = self._interleaved_target_sampling_steps(
            next_condition_updates,
            total_condition_updates,
        )
        steps_to_run = target_sampling_steps - int(state["completed_sampling_steps"])

        context_tokens = self._prepare_context(z_ar, z_context, z_observed, action_history)
        B = int(state["batch_size"])
        if not self._uses_batch_expansion:
            context_k = context_tokens  # [B, N, hidden_dim]
        else:
            K = self._batch_K
            context_k = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)

        for _ in range(steps_to_run):
            step_idx = int(state["completed_sampling_steps"])
            s = state["timesteps"][step_idx]
            t = state["timesteps"][step_idx + 1]
            state["x_t"] = self._run_interleaved_solver_step(
                state["x_t"],
                context_k,
                state["status_k"],
                state["noise_schedule"],
                s,
                t,
            )
            # Anchor correction: reset the first frame to clean anchor
            if self.use_anchor_frame and state.get("anchor_info") is not None:
                state["x_t"] = self._correct_anchor_xt(state["x_t"], state["anchor_info"])
            state["completed_sampling_steps"] = step_idx + 1

        state["completed_condition_updates"] = next_condition_updates
        return state

    @torch.no_grad()
    def finalize_interleaved_inference(
        self,
        state: Dict[str, torch.Tensor],
        z_ar: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize interleaved sampling and decode trajectories.

        If some solver steps remain because the caller did not provide all
        expected predictor updates, they are consumed using the last context.
        """
        while int(state["completed_condition_updates"]) < int(state["total_condition_updates"]):
            state = self.advance_interleaved_inference(
                state,
                z_ar,
                z_context=z_context,
                z_observed=z_observed,
                action_history=action_history,
            )

        B = int(state["batch_size"])
        t_0 = state["timesteps"][-1].reshape(1)
        final_context_tokens = self._prepare_context(z_ar, z_context, z_observed, action_history)

        if not self._uses_batch_expansion:
            K = self.num_modes
            final_context_k = final_context_tokens
        else:
            K = self._batch_K
            final_context_k = (
                final_context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
            )

        final_solver = dpm.DPM_Solver(
            dpm.model_wrapper(
                self.dit,
                state["noise_schedule"],
                model_type=self.dit.model_type,
                model_kwargs={
                    "cross_c": final_context_k,
                    "status_emb": state["status_k"],
                },
            ),
            state["noise_schedule"],
            algorithm_type="dpmsolver++",
        )
        cls_pred, x_0 = final_solver.denoise_to_zero_fn(state["x_t"], t_0)
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]  # strip anchor frame
        traj_3d = self._convert_6d_to_3d(x_0)

        if self.independent_modes and self.confidence_head is not None:
            cls_pred = self.confidence_head(x_0, final_context_tokens)
            confidences = F.softmax(cls_pred, dim=-1)
        elif not self._uses_batch_expansion and cls_pred is not None:
            confidences = F.softmax(cls_pred, dim=-1)  # [B, K]
        else:
            confidences = torch.ones(B, K, device=traj_3d.device) / K

        return {
            "trajectories": traj_3d,
            "confidences": confidences,
        }

    def _correct_anchor_xt(self, x_t: torch.Tensor, anchor_info: torch.Tensor) -> torch.Tensor:
        """
        Reset the anchor (first frame) of x_t to clean values.

        Works for both multi-modal [B, K*TF*6] and legacy [B*K, TF*6] layouts.

        Args:
            x_t: flat noisy trajectory
            anchor_info: [B, K, 1, 6] (multi-modal) or [B*K, 1, 6] (legacy)
        """
        if not self._uses_batch_expansion:
            K = self.num_modes
            B = anchor_info.shape[0]
            TF = self.total_frames
            xt_4d = x_t.view(B, K, TF, self.traj_dim)
            xt_4d[:, :, :1, :] = anchor_info
            return xt_4d.reshape(B, -1)
        else:
            BK = anchor_info.shape[0]
            TF = self.total_frames
            xt_3d = x_t.view(BK, TF, self.traj_dim)
            xt_3d[:, :1, :] = anchor_info
            return xt_3d.reshape(BK, -1)

    def forward(
        self,
        z_ar: torch.Tensor,
        status_feature: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        gt_trajectory: Optional[torch.Tensor] = None,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            z_ar: [B, T*P, encoder_dim] — predictor output tokens
            status_feature: [B, status_dim] — ego status
            z_context: [B, P, encoder_dim] — optional first-frame encoder tokens
            z_observed: [B, T_obs*P, encoder_dim] — optional observed-frame encoder tokens
            action_history: [B, T_obs, action_history_dim] — optional observed history tokens
            gt_trajectory: [B, num_poses, 6] — ground truth in 6-dim format
                           (x, y, vx, vy, cos_yaw, sin_yaw). Only needed for training.
            anchor_state: [B, 6] or [B, 1, 6] — current ego state for anchor frame.
                          Only used when ``use_anchor_frame=True``. If *None*, a
                          default zero-origin anchor is used.

        Returns:
            Training: {"loss", "reg_loss", "conf_loss", "cover_loss"}
            Inference: {"trajectories": [B, K, num_poses, 3], "confidences": [B, K]}
        """
        # Prepare conditioning
        context_tokens = self._prepare_context(z_ar, z_context, z_observed, action_history)  # [B, N, hidden_dim]
        status_emb = self._prepare_status(status_feature)  # [B, hidden_dim]

        if gt_trajectory is not None and self.training:
            return self._training_forward(context_tokens, status_emb, gt_trajectory, anchor_state)
        else:
            return self._inference_forward(context_tokens, status_emb, anchor_state)

    def _training_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Diffusion training: add noise to GT, predict clean trajectory, compute loss.

        When ``use_anchor_frame=True``, the current ego state is prepended as
        frame 0.  Only future frames are noised; the anchor stays clean.  Loss
        is computed on future frames only.

        When ``num_modes > 1`` uses WTA loss with focal classification,
        smooth-L1 regression, velocity and yaw supervision (XTR-aligned).
        When ``num_modes == 1`` falls back to simple MSE denoising loss.

        Args:
            context_tokens: [B, N, hidden_dim]
            status_emb: [B, hidden_dim]
            gt_trajectory: [B, num_poses, 6]
            anchor_state: [B, 6] or None

        Returns:
            {"loss", "reg_loss", "conf_loss", "cover_loss"}
        """
        B = gt_trajectory.shape[0]
        device = gt_trajectory.device
        K = self.num_modes

        if K > 1:
            if self.independent_modes:
                return self._training_forward_independent(context_tokens, status_emb, gt_trajectory, anchor_state)
            return self._training_forward_multimodal(context_tokens, status_emb, gt_trajectory, anchor_state)

        # # ── Legacy single-mode: simple MSE loss ──────────────────────
        # eps = 1e-3
        # t = torch.rand(B, device=device) * (1 - eps) + eps

        # # Noise only future frames
        # gt_flat = gt_trajectory.reshape(B, -1)
        # mean, std = self.sde.marginal_prob(gt_flat, t)
        # z = torch.randn_like(gt_flat)
        # noised_future = mean + std * z  # [B, num_poses * 6]

        # if self.use_anchor_frame:
        #     anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
        #     anchor_flat = anchor.reshape(B, -1)  # [B, 6]
        #     x_t = torch.cat([anchor_flat, noised_future], dim=-1)  # [B, (1+T)*6]
        # else:
        #     x_t = noised_future

        # _, x_pred = self.dit(x_t, t, context_tokens, status_emb)

        # if self.use_anchor_frame:
        #     # Strip anchor prediction, loss on future only
        #     x_pred = x_pred.view(B, self.total_frames, self.traj_dim)[:, 1:, :].reshape(B, -1)

        # reg_loss = F.mse_loss(x_pred, gt_flat)
        # return {
        #     "loss": reg_loss,
        #     "reg_loss": reg_loss,
        #     "conf_loss": torch.tensor(0.0, device=device),
        #     "cover_loss": torch.tensor(0.0, device=device),
        # }
        # ── Single-mode regression with separated XY / vel / yaw losses ──
        eps = 1e-3
        t = torch.rand(B, device=device) * (1 - eps) + eps
        TF = self.total_frames
        td = self.traj_dim

        # Noise only future frames
        gt_flat = gt_trajectory.reshape(B, -1)
        mean, std = self.sde.marginal_prob(gt_flat, t)
        z = torch.randn_like(gt_flat)
        noised_future = mean + std * z

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
            anchor_flat = anchor.reshape(B, -1)
            x_t = torch.cat([anchor_flat, noised_future], dim=-1)
        else:
            x_t = noised_future

        _, x_pred = self.dit(x_t, t, context_tokens, status_emb)

        x_pred_4d = x_pred.view(B, TF, td)
        if self.use_anchor_frame:
            x_pred_4d = x_pred_4d[:, 1:, :]  # [B, num_poses, td]

        # Separated losses — same recipe as K>1 path so weight knobs work
        gt_xy = gt_trajectory[:, :, :2]
        pred_xy = x_pred_4d[:, :, :2]
        reg_loss = self._xy_regression_loss_per_mode(pred_xy, gt_xy).mean()

        if self._has_velocity:
            vel_loss = F.smooth_l1_loss(x_pred_4d[..., 2], gt_trajectory[..., 2], reduction="mean") + F.smooth_l1_loss(
                x_pred_4d[..., 3], gt_trajectory[..., 3], reduction="mean"
            )
        else:
            vel_loss = torch.tensor(0.0, device=device)

        ys = self._yaw_slice
        cos_sim = F.cosine_similarity(x_pred_4d[..., ys], gt_trajectory[..., ys], dim=-1)
        yaw_loss = ((1.0 - cos_sim) / 2.0).mean()

        total_loss = (
            self.reg_loss_weight * reg_loss + self.vel_loss_weight * vel_loss + self.yaw_loss_weight * yaw_loss
        )
        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": torch.tensor(0.0, device=device),
            "cover_loss": torch.tensor(0.0, device=device),
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
        }

    def set_awta_temperature(self, T: float) -> None:
        """Anneal the aWTA temperature (called per-epoch by external scheduler).

        The runtime temperature is clamped to ``awta_min_temperature`` to
        prevent full collapse to hard WTA in late epochs.
        """
        T_eff = max(float(T), self.awta_min_temperature)
        self.awta_temperature.fill_(T_eff)

    def _training_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Independent-mode training: B->B*K batch expansion, single-mode DiT.

        Each of the K modes gets its own noise realisation and is denoised
        independently through the (single-mode) DiT.  A separate
        ``TrajectoryConfidenceHead`` predicts per-mode confidence scores.
        """
        B = gt_trajectory.shape[0]
        K = self.num_modes
        device = gt_trajectory.device
        TF = self.total_frames
        td = self.traj_dim
        num_poses = self.num_poses

        # 1. Replicate GT K times, flatten to batch dim
        gt_rep = gt_trajectory.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, num_poses, td)

        # 2. Same timestep per original sample (fair distance comparison)
        eps = 1e-3
        t = torch.rand(B, device=device) * (1 - eps) + eps
        t_bk = t.unsqueeze(1).expand(-1, K).reshape(B * K)

        # 3. Independent noise per mode
        future_flat = gt_rep.reshape(B * K, -1)
        mean, std = self.sde.marginal_prob(future_flat, t_bk)
        z = torch.randn_like(future_flat)
        noised_future = mean + std * z

        # 4. Anchor frame handling
        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)
            anchor_bk = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, td)
            noised_3d = noised_future.view(B * K, num_poses, td)
            x_t = torch.cat([anchor_bk, noised_3d], dim=1).reshape(B * K, -1)
        else:
            x_t = noised_future

        # 5. Expand context / status to B*K
        ctx_bk = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_bk = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        # 6. Single-mode DiT forward
        _, x_pred = self.dit(x_t, t_bk, ctx_bk, status_bk)
        x_pred_4d = x_pred.view(B * K, TF, td)
        if self.use_anchor_frame:
            x_pred_4d = x_pred_4d[:, 1:, :]
        x_pred_4d = x_pred_4d.view(B, K, num_poses, td)

        # 7. Per-mode distance and aWTA weights (same as joint multimodal path)
        gt_xy = gt_trajectory[:, :, :2]
        pred_xy = x_pred_4d[:, :, :, :2]
        dist_xy = (pred_xy - gt_xy.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
        min_dist, winner_idx = dist_xy.min(dim=1)

        _LOGIT_CLAMP = 50.0
        awta_T = float(torch.clamp(self.awta_temperature, min=self.awta_min_temperature).item())
        awta_logits = (-dist_xy / awta_T).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        awta_w = F.softmax(awta_logits, dim=1).detach()

        # 8. Regression losses
        gt_xy_k = gt_xy.unsqueeze(1).expand(-1, K, -1, -1)
        per_mode_reg = self._xy_regression_loss_per_mode(pred_xy, gt_xy_k)

        if self._has_velocity:
            gt_vx_k = gt_trajectory[:, :, 2].unsqueeze(1).expand(-1, K, -1)
            gt_vy_k = gt_trajectory[:, :, 3].unsqueeze(1).expand(-1, K, -1)
            per_mode_vx = F.smooth_l1_loss(x_pred_4d[..., 2], gt_vx_k, reduction="none").mean(dim=-1)
            per_mode_vy = F.smooth_l1_loss(x_pred_4d[..., 3], gt_vy_k, reduction="none").mean(dim=-1)
            vel_loss = (awta_w * (per_mode_vx + per_mode_vy)).sum(dim=1).mean()
        else:
            vel_loss = torch.tensor(0.0, device=device)

        ys = self._yaw_slice
        gt_ang_k = gt_trajectory[..., ys].unsqueeze(1).expand(-1, K, -1, -1)
        cos_sim = F.cosine_similarity(x_pred_4d[..., ys], gt_ang_k, dim=-1)
        per_mode_yaw = ((1.0 - cos_sim) / 2.0).mean(dim=-1)

        reg_loss = (awta_w * per_mode_reg).sum(dim=1).mean()
        yaw_loss = (awta_w * per_mode_yaw).sum(dim=1).mean()

        # 9. Confidence from separate head (detached traj — DiT only gets reg gradients)
        cls_pred = self.confidence_head(x_pred_4d.detach(), context_tokens)

        # 10. XTR double-gate classification loss
        conf_logits_target = (-dist_xy / self.conf_temperature).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        soft_target = F.softmax(conf_logits_target, dim=1).detach()

        sample_valid = min_dist < self.cls_th
        if bool(sample_valid.any()):
            mode_keep = (dist_xy - min_dist.unsqueeze(1)) > self.cls_ignore
            winner_onehot = F.one_hot(winner_idx, num_classes=K).bool()
            mode_keep = mode_keep | winner_onehot
            log_probs = F.log_softmax(cls_pred, dim=1)
            per_sample_ce = -(soft_target * log_probs * mode_keep.float()).sum(dim=1)
            cls_loss = per_sample_ce[sample_valid].mean()
        else:
            cls_loss = cls_pred.sum() * 0.0

        total_loss = (
            self.reg_loss_weight * reg_loss
            + self.vel_loss_weight * vel_loss
            + self.yaw_loss_weight * yaw_loss
            + self.cls_loss_weight * cls_loss
        )

        winner_traj = x_pred_4d[torch.arange(B, device=device), winner_idx]
        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": cls_loss,
            "cover_loss": torch.zeros((), device=device),
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
            "winner_idx": winner_idx,
            "winner_traj_3d": self._convert_6d_to_3d(winner_traj.detach()),
            "awta_temperature": torch.tensor(awta_T, device=device),
            "cls_sample_valid_ratio": sample_valid.float().mean().detach(),
        }

    def _training_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Hybrid WTA training: aWTA weighted regression + XTR-gated soft-CE.

        Loss topology (see diffusion_planner design doc):
        - A single ``awta_w = softmax(-dist_xy / awta_T).detach()`` drives the
          per-mode weights for ``reg``, ``vel`` and ``yaw`` — all three
          regression heads share the same weight set so the DiT backbone
          receives a consistent "how K should participate" signal.
        - ``cls_loss`` uses a soft target ``softmax(-dist_xy / conf_T)`` with
          log-softmax CE, gated by two XTR-style masks:
            * ``mask0 = min_dist < cls_th`` — if no mode is close enough to GT
              (high-noise samples), skip cls for the whole sample.
            * ``mask1 = dist - min_dist > cls_ignore`` — drop non-winner modes
              too close to the winner (ambiguous modes).
        - Winner is kept only for logging / viz.

        Args:
            context_tokens: [B, N, hidden_dim]
            status_emb:    [B, hidden_dim]
            gt_trajectory: [B, num_poses, 6] — (x, y, vx, vy, cos_yaw, sin_yaw)
            anchor_state:  [B, 6] or None
        """
        B = gt_trajectory.shape[0]
        device = gt_trajectory.device
        K = self.num_modes

        # Replicate GT K times and diffuse only future frames
        gt_rep = gt_trajectory.unsqueeze(1).expand(-1, K, -1, -1)
        eps = 1e-3
        t = torch.rand(B, device=device) * (1 - eps) + eps

        future_flat = gt_rep.reshape(B, -1)
        mean, std = self.sde.marginal_prob(future_flat, t)
        z = torch.randn_like(future_flat)
        noised_future = mean + std * z

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
            anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, 1, 6]
            noised_future_4d = noised_future.view(B, K, self.num_poses, self.traj_dim)
            x_t = torch.cat([anchor_k, noised_future_4d], dim=-2)  # [B, K, TF, 6]
            x_t = x_t.reshape(B, -1)
        else:
            x_t = noised_future

        cls_pred, x_pred = self.dit(x_t, t, context_tokens, status_emb)
        # cls_pred: [B, K] raw logits (softmax via log_softmax below)
        # x_pred:   [B, K * total_frames * traj_dim]

        x_pred_4d = x_pred.view(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_pred_4d = x_pred_4d[:, :, 1:, :]  # [B, K, num_poses, 6]

        gt_full = gt_trajectory  # [B, T, 6]
        gt_xy = gt_full[:, :, :2]  # [B, T, 2]
        pred_xy = x_pred_4d[:, :, :, :2]  # [B, K, T, 2]

        # ── Per-mode ADE on xy: drives aWTA weights, soft CE target, and gates
        dist_xy = (pred_xy - gt_xy.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # [B, K]
        min_dist, winner_idx = dist_xy.min(dim=1)  # [B]

        # ── aWTA shared weights for reg / vel / yaw ──────────────────
        _LOGIT_CLAMP = 50.0
        awta_T = float(torch.clamp(self.awta_temperature, min=self.awta_min_temperature).item())
        awta_logits = (-dist_xy / awta_T).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        awta_w = F.softmax(awta_logits, dim=1).detach()  # [B, K]

        # ── Per-mode regression terms ────────────────────────────────
        gt_xy_k = gt_xy.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, T, 2]
        per_mode_reg = self._xy_regression_loss_per_mode(pred_xy, gt_xy_k)  # [B, K]

        if self._has_velocity:
            gt_vx_k = gt_full[:, :, 2].unsqueeze(1).expand(-1, K, -1)
            gt_vy_k = gt_full[:, :, 3].unsqueeze(1).expand(-1, K, -1)
            per_mode_vx = F.smooth_l1_loss(x_pred_4d[..., 2], gt_vx_k, reduction="none").mean(dim=-1)
            per_mode_vy = F.smooth_l1_loss(x_pred_4d[..., 3], gt_vy_k, reduction="none").mean(dim=-1)
            vel_loss = (awta_w * (per_mode_vx + per_mode_vy)).sum(dim=1).mean()
        else:
            vel_loss = torch.tensor(0.0, device=device)

        ys = self._yaw_slice
        gt_ang_k = gt_full[..., ys].unsqueeze(1).expand(-1, K, -1, -1)
        cos_sim = F.cosine_similarity(x_pred_4d[..., ys], gt_ang_k, dim=-1)
        per_mode_yaw = ((1.0 - cos_sim) / 2.0).mean(dim=-1)

        reg_loss = (awta_w * per_mode_reg).sum(dim=1).mean()
        yaw_loss = (awta_w * per_mode_yaw).sum(dim=1).mean()

        # ── Soft-CE classification with XTR double-gate ──────────────
        conf_logits_target = (-dist_xy / self.conf_temperature).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        soft_target = F.softmax(conf_logits_target, dim=1).detach()  # [B, K]

        # Gate 1 (sample-level): winner must be close enough to GT.
        sample_valid = min_dist < self.cls_th  # [B]

        if bool(sample_valid.any()):
            # Gate 2 (mode-level): drop ambiguous non-winner modes; keep winner always.
            mode_keep = (dist_xy - min_dist.unsqueeze(1)) > self.cls_ignore
            winner_onehot = F.one_hot(winner_idx, num_classes=K).bool()
            mode_keep = mode_keep | winner_onehot  # [B, K]

            log_probs = F.log_softmax(cls_pred, dim=1)  # [B, K]
            per_sample_ce = -(soft_target * log_probs * mode_keep.float()).sum(dim=1)
            cls_loss = per_sample_ce[sample_valid].mean()
        else:
            cls_loss = cls_pred.sum() * 0.0

        # ── Total ────────────────────────────────────────────────────
        total_loss = (
            self.reg_loss_weight * reg_loss
            + self.vel_loss_weight * vel_loss
            + self.yaw_loss_weight * yaw_loss
            + self.cls_loss_weight * cls_loss
        )

        winner_traj = x_pred_4d[torch.arange(B, device=device), winner_idx]

        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": cls_loss,
            "cover_loss": torch.zeros((), device=device),
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
            "winner_idx": winner_idx,
            "winner_traj_3d": self._convert_6d_to_3d(winner_traj.detach()),
            "awta_temperature": torch.tensor(awta_T, device=device),
            "cls_sample_valid_ratio": sample_valid.float().mean().detach(),
        }

    def _inference_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Diffusion inference: sample K trajectories via DPM-Solver++.

        When ``num_modes > 1``: K modes are built into the model; no batch
        expansion is needed.  The model produces K trajectories + K confidence
        scores in a single forward pass.
        When ``num_modes == 1``: legacy behaviour with K noise samples.

        When ``use_anchor_frame=True``, the anchor is prepended and a
        ``correcting_xt_fn`` is used to reset frame 0 after each solver step.

        Args:
            context_tokens: [B, N, hidden_dim]
            status_emb: [B, hidden_dim]
            anchor_state: [B, 6] or None

        Returns:
            {"trajectories": [B, K, num_poses, 3], "confidences": [B, K]}
        """
        B = context_tokens.shape[0]
        device = context_tokens.device

        if self.num_modes > 1:
            if self.independent_modes:
                return self._inference_forward_independent(context_tokens, status_emb, anchor_state)
            return self._inference_forward_multimodal(context_tokens, status_emb, anchor_state)

        # ── Legacy: K independent noise samples ──────────────────────
        K = self.num_samples
        context_k = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_k = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        dpm_solver_params = {}

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
            anchor_exp = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, self.traj_dim)
            noise = torch.randn(B * K, self.num_poses, self.traj_dim, device=device)
            x_T = torch.cat([anchor_exp, noise], dim=-2).reshape(B * K, -1)
            # Correction function to keep frame 0 clean
            BK = B * K
            TF = self.total_frames
            td = self.traj_dim
            _anchor_exp = anchor_exp  # capture for closure

            def correcting_xt_fn(xt, t_val, step):
                xt_3d = xt.view(BK, TF, td)
                xt_3d[:, :1, :] = _anchor_exp
                return xt_3d.reshape(BK, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn
        else:
            flat_dim = self.num_poses * self.traj_dim
            x_T = torch.randn(B * K, flat_dim, device=device)

        _, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": context_k, "status_emb": status_k},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]  # strip anchor
        traj_3d = self._convert_6d_to_3d(x_0)
        confidences = torch.ones(B, K, device=device) / K
        return {"trajectories": traj_3d, "confidences": confidences}

    def _inference_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Independent-mode inference: B->B*K batch expansion with single-mode DiT."""
        B = context_tokens.shape[0]
        K = self.num_modes
        device = context_tokens.device
        TF = self.total_frames
        td = self.traj_dim

        ctx_bk = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_bk = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        dpm_solver_params = {}

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)
            anchor_bk = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, td)
            noise = torch.randn(B * K, self.num_poses, td, device=device)
            x_T = torch.cat([anchor_bk, noise], dim=-2).reshape(B * K, -1)
            _anchor_bk = anchor_bk

            def correcting_xt_fn(xt, t_val, step):
                xt_3d = xt.view(B * K, TF, td)
                xt_3d[:, :1, :] = _anchor_bk
                return xt_3d.reshape(B * K, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn
        else:
            flat_dim = self.num_poses * self.traj_dim
            x_T = torch.randn(B * K, flat_dim, device=device)

        _, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": ctx_bk, "status_emb": status_bk},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )

        x_0 = x_0.reshape(B, K, TF, td)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]

        traj_3d = self._convert_6d_to_3d(x_0)
        cls_pred = self.confidence_head(x_0, context_tokens)
        confidences = F.softmax(cls_pred, dim=-1)

        return {"trajectories": traj_3d, "confidences": confidences}

    def _inference_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference with built-in K modes (XTR-aligned, no batch expansion).

        Args:
            context_tokens: [B, N, hidden_dim]
            status_emb: [B, hidden_dim]
            anchor_state: [B, 6] or None

        Returns:
            {"trajectories": [B, K, num_poses, 3], "confidences": [B, K]}
        """
        B = context_tokens.shape[0]
        device = context_tokens.device
        K = self.num_modes

        dpm_solver_params = {}

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)  # [B, 1, 6]
            anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, 1, 6]
            noise = torch.randn(B, K, self.num_poses, self.traj_dim, device=device)
            x_T = torch.cat([anchor_k, noise], dim=-2).reshape(B, -1)
            # Correction function to keep frame 0 clean
            TF = self.total_frames
            td = self.traj_dim
            _anchor_k = anchor_k  # capture for closure

            def correcting_xt_fn(xt, t_val, step):
                xt_4d = xt.view(B, K, TF, td)
                xt_4d[:, :, :1, :] = _anchor_k
                return xt_4d.reshape(B, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn
        else:
            flat_dim = K * self.num_poses * self.traj_dim
            x_T = torch.randn(B, flat_dim, device=device)

        # DPM-Solver++ sampling — model outputs (cls, reg) natively
        cls_pred, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": context_tokens, "status_emb": status_emb},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )

        # Reshape and strip anchor
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]  # strip anchor
        traj_3d = self._convert_6d_to_3d(x_0)  # [B, K, num_poses, 3]

        # Confidences from classification head (K-way softmax for K-mode competition)
        if cls_pred is not None:
            confidences = F.softmax(cls_pred, dim=-1)  # [B, K]
        else:
            confidences = torch.ones(B, K, device=device) / K

        return {"trajectories": traj_3d, "confidences": confidences}

    @staticmethod
    def _convert_nd_to_3d(traj_nd: torch.Tensor) -> torch.Tensor:
        """
        Convert N-dim trajectory to 3-dim (x, y, yaw).

        Supports 4D (x, y, cos_yaw, sin_yaw) and 6D (x, y, vx, vy, cos_yaw, sin_yaw).
        cos/sin yaw are always the last two dimensions.

        Args:
            traj_nd: [..., 4] or [..., 6]

        Returns:
            [..., 3] with (x, y, yaw)
        """
        x = traj_nd[..., 0]
        y = traj_nd[..., 1]
        td = traj_nd.shape[-1]
        cos_yaw = traj_nd[..., td - 2]
        sin_yaw = traj_nd[..., td - 1]
        yaw = torch.atan2(sin_yaw, cos_yaw)
        return torch.stack([x, y, yaw], dim=-1)

    # Backward-compatible alias
    @staticmethod
    def _convert_6d_to_3d(traj_6d: torch.Tensor) -> torch.Tensor:
        return DiffusionPlanner._convert_nd_to_3d(traj_6d)

    @staticmethod
    def convert_3d_to_nd(traj_3d: torch.Tensor, dt: float = 0.2, traj_dim: int = 6) -> torch.Tensor:
        """
        Convert 3-dim trajectory (x, y, yaw) to N-dim format.

        traj_dim=4: (x, y, cos_yaw, sin_yaw)
        traj_dim=6: (x, y, vx, vy, cos_yaw, sin_yaw)

        Args:
            traj_3d: [B, T, 3] with (x, y, yaw)
            dt: time interval between frames (default 0.2s = 5fps)
            traj_dim: output dimension (4 or 6)

        Returns:
            [B, T, traj_dim]
        """
        x = traj_3d[..., 0]
        y = traj_3d[..., 1]
        yaw = traj_3d[..., 2]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        if traj_dim == 4:
            return torch.stack([x, y, cos_yaw, sin_yaw], dim=-1)

        # 6D: compute velocities via finite differences
        vx = torch.zeros_like(x)
        vy = torch.zeros_like(y)
        vx[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt
        vy[:, 1:] = (y[:, 1:] - y[:, :-1]) / dt
        if x.shape[1] > 0:
            vx[:, 0] = x[:, 0] / dt
            vy[:, 0] = y[:, 0] / dt

        return torch.stack([x, y, vx, vy, cos_yaw, sin_yaw], dim=-1)

    # Backward-compatible alias
    @staticmethod
    def convert_3d_to_6d(traj_3d: torch.Tensor, dt: float = 0.2) -> torch.Tensor:
        return DiffusionPlanner.convert_3d_to_nd(traj_3d, dt=dt, traj_dim=6)
