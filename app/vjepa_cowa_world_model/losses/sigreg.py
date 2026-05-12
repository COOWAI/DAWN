"""
SIGReg — Sketch Isotropic Gaussian Regularizer

Ported from le-wm (Maes et al., 2026).
Forces embedding distribution to be isotropic Gaussian via characteristic-function-based
statistic, preventing representation collapse without EMA or stop-gradient.

Reference: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        Args:
            proj: (T, B, D) — time-first embedding sequence
        Returns:
            scalar loss
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def pool_tokens_per_frame(z_proj, tokens_per_frame):
    """Pool multi-token-per-frame projected embeddings for SIGReg.

    Args:
        z_proj: (B, T*P, embed_dim) — projected encoder output with P tokens per frame
        tokens_per_frame: int — number of tokens per video frame
    Returns:
        (T, B, embed_dim) — mean-pooled per-frame embeddings, time-first for SIGReg
    """
    B, TP, D = z_proj.shape
    T = TP // tokens_per_frame
    z_pooled = z_proj.reshape(B, T, tokens_per_frame, D).mean(dim=2)  # (B, T, D)
    return z_pooled.transpose(0, 1)  # (T, B, D)
