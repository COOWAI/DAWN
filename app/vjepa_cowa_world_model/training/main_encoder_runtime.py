"""Main encoder runtime helpers for V-JEPA and Drive-JEPA backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  # kept for callers that mirror predictor runtime imports

from app.vjepa_cowa_world_model.training.config import (
    is_drive_jepa_main_encoder_config,
    resolve_main_encoder_frame_stride,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.models import prepare_runtime_tokens


@dataclass(frozen=True)
class MainEncoderTimeline:
    """Resolved main-encoder temporal contract."""

    raw_num_frames: int
    frame_stride: int
    num_time_steps: int
    num_observed_steps: int
    num_future_steps: int
    tokens_per_frame: int


@dataclass(frozen=True)
class PredictorTimelineInputs:
    """Predictor inputs after aligning raw frame tensors to main-encoder steps."""

    raw_num_frames: int
    frame_stride: int
    num_time_steps: int
    num_observed_steps: int
    num_future_steps: int
    tokens_per_frame: int
    actions: torch.Tensor
    states: torch.Tensor
    extrinsics: torch.Tensor
    driving_command: Optional[torch.Tensor]
    ego_dynamics: Optional[torch.Tensor]


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def is_drive_jepa_main_encoder(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    core = unwrap_module(module)
    return bool(getattr(core, "is_drive_jepa_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "DriveJEPAImgEncoderAdapter"
    )


def resolve_main_timeline(config, encoder: Optional[torch.nn.Module], num_raw_frames: int) -> MainEncoderTimeline:
    frame_stride = resolve_main_encoder_frame_stride(config, encoder)
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames, encoder=encoder)
    num_observed_steps = resolve_main_encoder_num_observed_steps(config, encoder)
    if num_observed_steps > num_time_steps:
        raise ValueError(
            f"Observed predictor steps ({num_observed_steps}) exceed total predictor steps ({num_time_steps})"
        )
    return MainEncoderTimeline(
        raw_num_frames=int(num_raw_frames),
        frame_stride=frame_stride,
        num_time_steps=num_time_steps,
        num_observed_steps=num_observed_steps,
        num_future_steps=num_time_steps - num_observed_steps,
        tokens_per_frame=resolve_main_encoder_tokens_per_frame(config, encoder),
    )


def encode_main_context_tokens(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
) -> tuple[torch.Tensor, int]:
    """Encode a raw clip window and return flattened per-step tokens plus step count."""
    if context_clips.ndim != 5:
        raise ValueError(f"Expected context_clips shape [B, C, T, H, W], got ndim={context_clips.ndim}")

    batch_size, _, num_raw_frames, _, _ = context_clips.shape
    if is_drive_jepa_main_encoder_config(config) or is_drive_jepa_main_encoder(encoder):
        raw_tokens = encoder(context_clips, num_observed_frames=num_raw_frames)
        num_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames, encoder=encoder)
        return raw_tokens, num_steps

    encoder_input = context_clips
    if bool(getattr(config.data, "use_tubelet_repeat", True)):
        encoder_input = context_clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    encoded = encoder([encoder_input])[0]
    if bool(getattr(config.data, "use_tubelet_repeat", True)):
        encoded = encoded.view(batch_size, num_raw_frames, -1, encoded.size(-1)).flatten(1, 2)
    return encoded, num_raw_frames


def forward_main_context(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    raw_tokens, num_steps = encode_main_context_tokens(encoder, context_clips, config)
    return prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=runtime_normalize_reps,
        token_ae=token_ae,
    )


def forward_main_context_dual(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    predictor_normalize_reps: bool,
    proposal_normalize_reps: bool,
    predictor_token_ae: Optional[torch.nn.Module] = None,
    proposal_token_ae: Optional[torch.nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_tokens, num_steps = encode_main_context_tokens(encoder, context_clips, config)
    predictor_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=predictor_normalize_reps,
        token_ae=predictor_token_ae,
    )
    proposal_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=proposal_normalize_reps,
        token_ae=proposal_token_ae,
    )
    return predictor_context, proposal_context


def forward_main_target(
    target_encoder: torch.nn.Module,
    target_clips: torch.Tensor,
    config,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return forward_main_context(
            encoder=target_encoder,
            context_clips=target_clips,
            config=config,
            runtime_normalize_reps=runtime_normalize_reps,
            token_ae=token_ae,
        )


def should_reuse_context_as_target(config, encoder: Optional[torch.nn.Module] = None) -> bool:
    """Return whether frozen main-encoder tokens can be reused as predictor targets."""
    train_cfg = getattr(config, "train", None)
    if train_cfg is None:
        return False
    if not bool(getattr(train_cfg, "reuse_context_as_target_when_frozen", False)):
        return False
    if bool(getattr(train_cfg, "encoder_train", False)) or bool(getattr(train_cfg, "encoder_ema", False)):
        return False
    return bool(is_drive_jepa_main_encoder_config(config) or is_drive_jepa_main_encoder(encoder))


def _anchor_indices(num_raw_frames: int, frame_stride: int, device: torch.device) -> torch.Tensor:
    if frame_stride <= 1:
        return torch.arange(num_raw_frames, device=device, dtype=torch.long)
    return torch.arange(frame_stride - 1, num_raw_frames, frame_stride, device=device, dtype=torch.long)


def _index_temporal(tensor: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim < 3:
        return tensor
    if tensor.shape[1] < int(indices[-1].item()) + 1:
        raise ValueError(
            f"Temporal tensor length {tensor.shape[1]} is too short for anchor index {int(indices[-1].item())}"
        )
    return tensor.index_select(1, indices.to(tensor.device))


def _first_frame_indices(num_raw_frames: int, frame_stride: int, device: torch.device) -> torch.Tensor:
    if frame_stride <= 1:
        return torch.arange(num_raw_frames, device=device, dtype=torch.long)
    return torch.arange(0, num_raw_frames, frame_stride, device=device, dtype=torch.long)


def _pad_or_trim_temporal(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[1] == length:
        return tensor
    if tensor.shape[1] > length:
        return tensor[:, :length]
    pad_shape = list(tensor.shape)
    pad_shape[1] = length - tensor.shape[1]
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)


def build_ego_actions_between_states(states: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Build ego-frame actions between consecutive SE(2) states."""
    if states.shape[1] < 2:
        return states.new_zeros(states.shape[0], 0, action_dim)

    current = states[:, :-1]
    nxt = states[:, 1:]
    dx_global = nxt[..., 0] - current[..., 0]
    dy_global = nxt[..., 1] - current[..., 1]
    yaw = current[..., 5]
    cos_h = torch.cos(-yaw)
    sin_h = torch.sin(-yaw)
    dx_ego = cos_h * dx_global - sin_h * dy_global
    dy_ego = sin_h * dx_global + cos_h * dy_global
    d_yaw = nxt[..., 5] - current[..., 5]
    d_yaw = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))

    actions = states.new_zeros(states.shape[0], states.shape[1] - 1, action_dim)
    if action_dim > 0:
        actions[..., 0] = dx_ego
    if action_dim > 1:
        actions[..., 1] = dy_ego
    if action_dim > 2:
        actions[..., 2] = d_yaw
    return actions


def build_predictor_timeline_inputs(
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config,
    encoder: Optional[torch.nn.Module],
    dt: float,
) -> PredictorTimelineInputs:
    """Align predictor side inputs to the main encoder's temporal steps."""
    del dt  # State deltas already encode displacement in the sampled timeline.
    num_raw_frames = int(states.shape[1])
    timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=num_raw_frames)

    if timeline.frame_stride == 1:
        return PredictorTimelineInputs(
            raw_num_frames=timeline.raw_num_frames,
            frame_stride=timeline.frame_stride,
            num_time_steps=timeline.num_time_steps,
            num_observed_steps=timeline.num_observed_steps,
            num_future_steps=timeline.num_future_steps,
            tokens_per_frame=timeline.tokens_per_frame,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )

    indices = _anchor_indices(num_raw_frames, timeline.frame_stride, states.device)
    chunk_states = states.index_select(1, indices)
    chunk_actions = build_ego_actions_between_states(
        chunk_states, int(getattr(config.train, "action_dim", actions.shape[-1]))
    )
    chunk_extrinsics = _index_temporal(extrinsics, indices)
    if chunk_extrinsics is None:
        chunk_extrinsics = chunk_actions.new_zeros(
            chunk_actions.shape[0],
            chunk_states.shape[1],
            max(int(getattr(config.train, "action_dim", chunk_actions.shape[-1])) - 1, 1),
        )
    chunk_driving_command = _index_temporal(driving_command, indices)
    chunk_ego_dynamics = _index_temporal(ego_dynamics, indices)

    return PredictorTimelineInputs(
        raw_num_frames=timeline.raw_num_frames,
        frame_stride=timeline.frame_stride,
        num_time_steps=timeline.num_time_steps,
        num_observed_steps=timeline.num_observed_steps,
        num_future_steps=timeline.num_future_steps,
        tokens_per_frame=timeline.tokens_per_frame,
        actions=chunk_actions,
        states=chunk_states,
        extrinsics=chunk_extrinsics,
        driving_command=chunk_driving_command,
        ego_dynamics=chunk_ego_dynamics,
    )


def build_parallel_predictor_timeline_inputs(
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config,
    encoder: Optional[torch.nn.Module],
    dt: float,
) -> PredictorTimelineInputs:
    """Align side inputs for full-step future-query predictor forwarding."""
    del dt
    num_raw_frames = int(states.shape[1])
    timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=num_raw_frames)

    if timeline.frame_stride == 1:
        full_actions = _pad_or_trim_temporal(actions, timeline.num_time_steps)
        return PredictorTimelineInputs(
            raw_num_frames=timeline.raw_num_frames,
            frame_stride=timeline.frame_stride,
            num_time_steps=timeline.num_time_steps,
            num_observed_steps=timeline.num_observed_steps,
            num_future_steps=timeline.num_future_steps,
            tokens_per_frame=timeline.tokens_per_frame,
            actions=full_actions,
            states=_pad_or_trim_temporal(states, timeline.num_time_steps),
            extrinsics=_pad_or_trim_temporal(extrinsics, timeline.num_time_steps),
            driving_command=(
                _pad_or_trim_temporal(driving_command, timeline.num_time_steps)
                if driving_command is not None and driving_command.ndim >= 3
                else driving_command
            ),
            ego_dynamics=(
                _pad_or_trim_temporal(ego_dynamics, timeline.num_time_steps)
                if ego_dynamics is not None and ego_dynamics.ndim >= 3
                else ego_dynamics
            ),
        )

    indices = _first_frame_indices(num_raw_frames, timeline.frame_stride, states.device)[: timeline.num_time_steps]
    action_indices = _first_frame_indices(actions.shape[1], timeline.frame_stride, actions.device)[
        : timeline.num_time_steps
    ]
    chunk_actions = (
        actions.index_select(1, action_indices.to(actions.device)) if action_indices.numel() else actions[:, :0]
    )
    chunk_actions = _pad_or_trim_temporal(chunk_actions, timeline.num_time_steps)
    chunk_states = _pad_or_trim_temporal(states.index_select(1, indices), timeline.num_time_steps)
    chunk_extrinsics = _index_temporal(extrinsics, indices)
    if chunk_extrinsics is None:
        chunk_extrinsics = chunk_actions.new_zeros(
            chunk_actions.shape[0],
            timeline.num_time_steps,
            max(int(getattr(config.train, "action_dim", chunk_actions.shape[-1])) - 1, 1),
        )
    chunk_extrinsics = _pad_or_trim_temporal(chunk_extrinsics, timeline.num_time_steps)
    chunk_driving_command = _index_temporal(driving_command, indices)
    chunk_ego_dynamics = _index_temporal(ego_dynamics, indices)

    return PredictorTimelineInputs(
        raw_num_frames=timeline.raw_num_frames,
        frame_stride=timeline.frame_stride,
        num_time_steps=timeline.num_time_steps,
        num_observed_steps=timeline.num_observed_steps,
        num_future_steps=timeline.num_future_steps,
        tokens_per_frame=timeline.tokens_per_frame,
        actions=chunk_actions,
        states=chunk_states,
        extrinsics=chunk_extrinsics,
        driving_command=chunk_driving_command,
        ego_dynamics=chunk_ego_dynamics,
    )
