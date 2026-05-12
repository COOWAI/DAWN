# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Prefix-conditioned diffusion planner variant.

Keeps the base DiffusionPlanner implementation unchanged and only alters the
training conditioning path: during training it randomly truncates future z_ar
tokens so the denoiser learns to improve as predictor rollout grows.
"""

from typing import Dict, Optional

import torch

from .diffusion_planner import DiffusionPlanner


class PrefixConditionedDiffusionPlanner(DiffusionPlanner):
    """Diffusion planner variant with random rollout-prefix conditioning."""

    def __init__(
        self,
        *args,
        train_min_prefix_frames: int = 1,
        train_full_prefix_prob: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_min_prefix_frames = max(1, int(train_min_prefix_frames))
        self.train_full_prefix_prob = float(train_full_prefix_prob)

    def _maybe_apply_training_prefix_conditioning(self, z_ar: torch.Tensor) -> torch.Tensor:
        """Randomly truncate future predictor tokens during training."""
        if z_ar.size(1) % self.tokens_per_frame != 0:
            raise ValueError(
                f"z_ar token length {z_ar.size(1)} is not divisible by tokens_per_frame={self.tokens_per_frame}"
            )

        total_future_frames = z_ar.size(1) // self.tokens_per_frame
        if total_future_frames <= 1:
            return z_ar

        if torch.rand(1, device=z_ar.device).item() < self.train_full_prefix_prob:
            prefix_frames = total_future_frames
        else:
            min_prefix = min(self.train_min_prefix_frames, total_future_frames - 1)
            prefix_frames = int(torch.randint(min_prefix, total_future_frames, size=(1,), device=z_ar.device).item())

        return z_ar[:, : prefix_frames * self.tokens_per_frame]

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
        if gt_trajectory is not None and self.training:
            z_ar = self._maybe_apply_training_prefix_conditioning(z_ar)

        return super().forward(
            z_ar,
            status_feature,
            z_context=z_context,
            z_observed=z_observed,
            action_history=action_history,
            gt_trajectory=gt_trajectory,
            anchor_state=anchor_state,
        )
