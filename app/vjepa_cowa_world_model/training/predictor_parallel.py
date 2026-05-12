"""Future-query parallel predictor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.models import (
    build_predictor_input_with_future_queries,
    register_predictor_future_query_tokens,
)
from app.vjepa_cowa_world_model.utils import mask_future_actions, prepare_inference_consistent_states


@dataclass(frozen=True)
class ParallelPredictorOutput:
    """Outputs from a single future-query predictor forward."""

    z_pred: torch.Tensor
    z_future: torch.Tensor
    z_ar: Optional[torch.Tensor]


def use_parallel_predictor(config) -> bool:
    """Return whether future-query parallel predictor mode is enabled."""
    return bool(getattr(config.train, "use_parallel_predictor", False))


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def has_parallel_predictor_tokens(predictor: torch.nn.Module) -> bool:
    predictor_core = unwrap_module(predictor)
    return getattr(predictor_core, "future_query_tokens", None) is not None


def maybe_register_parallel_predictor_tokens(
    predictor: torch.nn.Module,
    config,
    embed_dim: int,
    future_steps: int,
    tokens_per_frame: int,
    device: torch.device,
) -> None:
    """Register future query tokens when parallel predictor mode is enabled."""
    if not use_parallel_predictor(config):
        return
    register_predictor_future_query_tokens(
        predictor,
        embed_dim=embed_dim,
        future_tubelets=int(future_steps),
        tokens_per_frame=int(tokens_per_frame),
        device=device,
    )


def build_parallel_predictor_input(
    predictor: torch.nn.Module,
    observed_tokens: torch.Tensor,
    num_observed_steps: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Build observed-token prefix plus learnable future query tokens."""
    num_observed_tokens = int(num_observed_steps) * int(tokens_per_frame)
    if observed_tokens.size(1) < num_observed_tokens:
        raise ValueError(
            "Observed token sequence is shorter than the requested observed prefix: "
            f"tokens={observed_tokens.size(1)}, requested={num_observed_tokens}"
        )
    observed_prefix = observed_tokens[:, :num_observed_tokens]
    return build_predictor_input_with_future_queries(predictor, observed_prefix)


def forward_parallel_predictor(
    predictor: torch.nn.Module,
    observed_tokens: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    config,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    num_observed_steps: int,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    predictor_no_aux_input: Optional[bool] = None,
) -> ParallelPredictorOutput:
    """Run predictor once with future query tokens and return future tokens."""
    z_input = build_parallel_predictor_input(
        predictor,
        observed_tokens=observed_tokens,
        num_observed_steps=num_observed_steps,
        tokens_per_frame=tokens_per_frame,
    )
    if z_input.size(1) % tokens_per_frame != 0:
        raise ValueError(
            f"Parallel predictor input tokens ({z_input.size(1)}) must be divisible by tokens_per_frame "
            f"({tokens_per_frame})"
        )
    num_steps = z_input.size(1) // tokens_per_frame
    if actions.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor actions length mismatch: {actions.shape[1]} vs {num_steps}")
    if states.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor states length mismatch: {states.shape[1]} vs {num_steps}")
    if extrinsics.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor extrinsics length mismatch: {extrinsics.shape[1]} vs {num_steps}")

    no_aux_input = bool(getattr(config.train, "predictor_no_aux_input", False))
    if predictor_no_aux_input is not None:
        no_aux_input = bool(predictor_no_aux_input)

    action_mask = None
    actions_input = actions
    extrinsics_input = extrinsics
    if no_aux_input:
        actions_input = torch.zeros_like(actions)
        states_input = torch.zeros_like(states)
        extrinsics_input = torch.zeros_like(extrinsics)
    elif bool(getattr(config.train, "predictor_inference_consistent", False)):
        num_known = int(num_observed_steps) - 1
        actions_input, action_mask = mask_future_actions(actions, num_known)
        states_input = prepare_inference_consistent_states(
            states,
            num_observed=int(num_observed_steps),
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            state_dim=int(getattr(config.train, "state_dim", states.shape[-1])),
            use_drive_command=bool(getattr(config.train, "use_drive_command", True)),
        )
    elif bool(getattr(config.train, "use_states_for_predictor", True)):
        states_input = states
    else:
        states_input = torch.zeros_like(states)

    z_pred = predictor(z_input, actions_input, states_input, extrinsics_input, action_mask=action_mask)
    if runtime_normalize_reps:
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

    future_start = int(num_observed_steps) * int(tokens_per_frame)
    z_future = z_pred[:, future_start:]
    return ParallelPredictorOutput(z_pred=z_pred, z_future=z_future, z_ar=z_future)
