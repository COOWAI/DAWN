"""Helper utilities for planner-side training and validation."""

from typing import Optional, Sequence

import torch


def resolve_validation_timestep_sec(
    fps: Optional[float] = None,
    diff_dt: Optional[float] = None,
    default: float = 0.5,
) -> float:
    """Resolve validation timestep seconds from config values.

    Prefer dataset fps because validation metrics are indexed by real trajectory
    timestamps. Fall back to planner.diff_dt, then to the provided default.
    """
    if fps is not None and fps > 0:
        return 1.0 / float(fps)
    if diff_dt is not None and diff_dt > 0:
        return float(diff_dt)
    return float(default)


def horizon_seconds_to_step_index(seconds: float, timestep_sec: float) -> int:
    """Map a future horizon in seconds to a 0-indexed trajectory step."""
    if timestep_sec <= 0:
        raise ValueError(f"timestep_sec must be positive, got {timestep_sec}")
    return int(float(seconds) / float(timestep_sec)) - 1


def build_horizon_regression_timestep_weights(
    num_poses: int,
    timestep_sec: float,
    horizon_seconds: Sequence[float],
    horizon_weights: Sequence[float],
    normalize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Build optional per-timestep regression weights for horizon-focused loss.

    Parameters
    ----------
    num_poses       : number of future trajectory poses.
    timestep_sec    : seconds represented by one future trajectory step.
    horizon_seconds : horizons to reweight, e.g. [2.0, 3.0].
    horizon_weights : raw weights for each horizon, e.g. [2.0, 2.0].
    normalize       : if True, scale the final vector to mean 1.

    Returns
    -------
    Optional[torch.Tensor]
        [num_poses] weights, or None when disabled or no horizons are in range.
    """
    if len(horizon_seconds) != len(horizon_weights):
        raise ValueError(
            "horizon_seconds and horizon_weights must have the same length, "
            f"got {len(horizon_seconds)} and {len(horizon_weights)}"
        )
    if num_poses <= 0:
        raise ValueError(f"num_poses must be positive, got {num_poses}")
    if not horizon_seconds:
        return None

    weights = torch.ones(num_poses, device=device, dtype=dtype)
    applied = False
    for seconds, raw_weight in zip(horizon_seconds, horizon_weights):
        raw_weight = float(raw_weight)
        if raw_weight < 0:
            raise ValueError(f"horizon regression weight must be non-negative, got {raw_weight}")
        step_idx = horizon_seconds_to_step_index(float(seconds), timestep_sec=timestep_sec)
        if 0 <= step_idx < num_poses:
            weights[step_idx] = raw_weight
            applied = True

    if not applied:
        return None
    if normalize:
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("horizon regression weights must sum to a positive value")
        weights = weights * (weights.numel() / weight_sum)
    return weights
