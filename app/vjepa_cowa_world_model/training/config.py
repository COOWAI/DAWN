# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
配置解析模块

提供结构化的配置数据类，用于解析训练配置 YAML 文件。
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

_PROPOSAL_PRETRAIN_YAML_NAMES = ("params-pretrain.yaml", "params-pretrain.yml")


def _parse_hw_resolution(value: Any, label: str) -> Tuple[int, int]:
    try:
        resolution = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must contain [height, width]") from exc
    if len(resolution) != 2:
        raise ValueError(f"{label} must contain [height, width]")
    return int(resolution[0]), int(resolution[1])


def _get_nested_value(config: Any, *keys: str, default=None):
    current = config
    for key in keys:
        if current is None:
            return default
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default


def normalize_image_size(image_size: Any) -> Tuple[int, int]:
    if isinstance(image_size, int):
        size = int(image_size)
        return size, size
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"image size must be an int or a 2-element sequence, got {image_size!r}")


def compute_tokens_per_frame(image_size: Any, patch_size: int) -> int:
    height, width = normalize_image_size(image_size)
    return int((height // patch_size) * (width // patch_size))


@lru_cache(maxsize=128)
def _load_proposal_pretrain_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path).expanduser()
    for name in _PROPOSAL_PRETRAIN_YAML_NAMES:
        yaml_path = ckpt_path.parent / name
        if not yaml_path.is_file():
            continue
        with open(yaml_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _get_proposal_pretrain_value(config: Any, *keys: str, default=None):
    checkpoint = _get_nested_value(config, "proposal", "checkpoint", default=None)
    if not checkpoint:
        return default
    return _get_adjacent_pretrain_value(checkpoint, *keys, default=default)


def _get_adjacent_pretrain_config(checkpoint: str) -> Dict[str, Any]:
    try:
        return _load_proposal_pretrain_config(str(checkpoint))
    except (OSError, yaml.YAMLError):
        return {}


def _get_adjacent_pretrain_value(checkpoint: str, *keys: str, default=None):
    pretrain_config = _get_adjacent_pretrain_config(checkpoint)
    if not pretrain_config:
        return default
    return _get_nested_value(pretrain_config, *keys, default=default)


def resolve_predictor_runtime_normalize_reps(config: Any) -> bool:
    """Resolve predictor/refiner representation normalization from predictor checkpoint metadata."""
    predictor_checkpoint = _get_nested_value(config, "meta", "predictor_checkpoint", default=None)
    if predictor_checkpoint:
        pretrain_value = _get_adjacent_pretrain_value(
            str(predictor_checkpoint),
            "loss",
            "normalize_reps",
            default=None,
        )
        if pretrain_value is not None:
            return bool(pretrain_value)
    return bool(_get_nested_value(config, "loss", "normalize_reps", default=True))


def resolve_effective_tokens_per_frame(config: Any) -> int:
    token_ae_enabled = bool(_get_nested_value(config, "token_ae", "enabled", default=False))
    if token_ae_enabled:
        num_latent_tokens = int(_get_nested_value(config, "token_ae", "num_latent_tokens", default=0))
        if num_latent_tokens <= 0:
            raise ValueError("token_ae.num_latent_tokens must be > 0 when token_ae.enabled=True")
        return num_latent_tokens

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)

    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_proposal_runtime_normalize_reps(config: Any) -> bool:
    """Resolve the representation normalization used by the frozen proposal branch."""
    explicit = _get_nested_value(config, "proposal", "runtime_normalize_reps", default=None)
    if explicit is not None:
        return bool(explicit)
    pretrain_value = _get_proposal_pretrain_value(config, "loss", "normalize_reps", default=None)
    if pretrain_value is not None:
        return bool(pretrain_value)
    return bool(_get_nested_value(config, "loss", "normalize_reps", default=True))


def resolve_proposal_use_token_ae(config: Any) -> bool:
    """Resolve whether the frozen proposal branch consumes TokenAE-compressed tokens."""
    explicit = _get_nested_value(config, "proposal", "use_token_ae", default=None)
    if explicit is not None:
        return bool(explicit)
    return False


def resolve_proposal_encoder_backbone(config: Any) -> str:
    """Resolve the independent proposal encoder backbone."""
    explicit = _get_nested_value(config, "proposal", "encoder_backbone", default=None)
    if explicit:
        return str(explicit)
    return str(_get_nested_value(config, "model", "backbone", default="vjepa2"))


def _is_drive_jepa_proposal_encoder_config(config: Any) -> bool:
    return resolve_proposal_encoder_backbone(config) == "drive_jepa_img_encoder"


def _get_encoder_static_attr(encoder: Optional[Any], name: str) -> Optional[int]:
    if encoder is None:
        return None
    core = encoder.module if hasattr(encoder, "module") else encoder
    value = getattr(core, name, None)
    return None if value is None else int(value)


def resolve_proposal_tokens_per_frame(config: Any, proposal_encoder: Optional[Any] = None) -> int:
    """Resolve tokens/frame for the frozen proposal branch, independent of predictor runtime."""
    if resolve_proposal_use_token_ae(config):
        return resolve_effective_tokens_per_frame(config)

    encoder_tokens_per_frame = _get_encoder_static_attr(proposal_encoder, "tokens_per_frame")
    if encoder_tokens_per_frame is not None:
        return encoder_tokens_per_frame

    if _is_drive_jepa_proposal_encoder_config(config):
        height, width = _get_nested_value(config, "proposal", "drive_jepa_resolution", default=None)
        patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
        return int((int(height) // patch_size) * (int(width) // patch_size))

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)
    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_proposal_num_time_steps(config: Any, proposal_encoder: Optional[Any] = None) -> int:
    """Resolve proposal encoder temporal token steps."""
    encoder_steps = _get_encoder_static_attr(proposal_encoder, "num_time_steps")
    if encoder_steps is not None:
        return encoder_steps

    num_observed = int(_get_nested_value(config, "train", "num_observed_frames", default=1))
    if _is_drive_jepa_proposal_encoder_config(config):
        num_frames = int(_get_nested_value(config, "proposal", "drive_jepa_num_frames", default=2))
        if num_frames <= 0:
            raise ValueError("proposal.drive_jepa_num_frames must be positive")
        if num_observed % num_frames != 0:
            raise ValueError(
                f"train.num_observed_frames ({num_observed}) must be divisible by "
                f"proposal.drive_jepa_num_frames ({num_frames})"
            )
        return num_observed // num_frames
    return num_observed


def is_drive_jepa_main_encoder_config(config: Any) -> bool:
    """Return whether the main encoder backbone is the Drive-JEPA image encoder."""
    return str(_get_nested_value(config, "model", "backbone", default="vjepa2")) == "drive_jepa_img_encoder"


def _validate_drive_jepa_main_token_ae(config: Any) -> None:
    if not is_drive_jepa_main_encoder_config(config) or not bool(
        _get_nested_value(config, "token_ae", "enabled", default=False)
    ):
        return

    height, width = _parse_hw_resolution(
        _get_nested_value(config, "model", "drive_jepa_resolution", default=(256, 512)),
        "model.drive_jepa_resolution",
    )
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("model.drive_jepa_resolution must be divisible by data.patch_size")

    num_latent_tokens = int(_get_nested_value(config, "token_ae", "num_latent_tokens", default=0))
    if num_latent_tokens <= 0:
        raise ValueError("token_ae.num_latent_tokens must be > 0 when Drive-JEPA main encoder uses TokenAE")

    input_grid_size = _get_nested_value(config, "token_ae", "input_grid_size", default=None)
    if input_grid_size is not None:
        input_grid = _parse_hw_resolution(input_grid_size, "token_ae.input_grid_size")
        expected_grid = (height // patch_size, width // patch_size)
        if input_grid != expected_grid:
            raise ValueError(
                "token_ae.input_grid_size must match Drive-JEPA raw token grid " f"{expected_grid}, got {input_grid}"
            )


def resolve_main_encoder_raw_tokens_per_frame(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve raw tokens per main-encoder predictor step before TokenAE compression."""
    if is_drive_jepa_main_encoder_config(config):
        _validate_drive_jepa_main_token_ae(config)
        encoder_tokens = _get_encoder_static_attr(encoder, "tokens_per_frame")
        if encoder_tokens is not None:
            return encoder_tokens
        height, width = _parse_hw_resolution(
            _get_nested_value(config, "model", "drive_jepa_resolution", default=(256, 512)),
            "model.drive_jepa_resolution",
        )
        patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("model.drive_jepa_resolution must be divisible by data.patch_size")
        return int((height // patch_size) * (width // patch_size))

    data_tokens = _get_nested_value(config, "data", "tokens_per_frame", default=None)
    if data_tokens is not None:
        return int(data_tokens)
    crop_size = _get_nested_value(config, "data", "crop_size", default=256)
    patch_size = int(_get_nested_value(config, "data", "patch_size", default=16))
    return compute_tokens_per_frame(crop_size, patch_size)


def resolve_main_encoder_frame_stride(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve raw-frame stride represented by one main-encoder predictor step."""
    if not is_drive_jepa_main_encoder_config(config):
        return 1

    _validate_drive_jepa_main_token_ae(config)
    encoder_stride = _get_encoder_static_attr(encoder, "num_frames")
    stride = (
        encoder_stride
        if encoder_stride is not None
        else int(_get_nested_value(config, "model", "drive_jepa_num_frames", default=2))
    )
    if stride <= 0:
        raise ValueError(f"model.drive_jepa_num_frames must be positive, got {stride}")

    for label, value in (
        ("data.num_target_frames", _get_nested_value(config, "data", "num_target_frames", default=None)),
        ("train.num_observed_frames", _get_nested_value(config, "train", "num_observed_frames", default=None)),
    ):
        if value is not None and int(value) % stride != 0:
            raise ValueError(f"{label} ({int(value)}) must be divisible by Drive-JEPA stride ({stride})")

    total = _get_nested_value(config, "data", "num_target_frames", default=None)
    observed = _get_nested_value(config, "train", "num_observed_frames", default=None)
    if total is not None and observed is not None:
        future = int(total) - int(observed)
        if future < 0:
            raise ValueError(
                f"data.num_target_frames ({int(total)}) must be >= train.num_observed_frames ({int(observed)})"
            )
        if future % stride != 0:
            raise ValueError(f"future frame count ({future}) must be divisible by Drive-JEPA stride ({stride})")

    return stride


def resolve_main_encoder_tokens_per_frame(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve tokens per main-encoder predictor step."""
    if is_drive_jepa_main_encoder_config(config):
        _validate_drive_jepa_main_token_ae(config)
        if bool(_get_nested_value(config, "token_ae", "enabled", default=False)):
            return resolve_effective_tokens_per_frame(config)
        return resolve_main_encoder_raw_tokens_per_frame(config, encoder)
    return resolve_effective_tokens_per_frame(config)


def resolve_main_encoder_num_time_steps(config: Any, num_raw_frames: int, encoder: Optional[Any] = None) -> int:
    """Resolve predictor time steps for a raw temporal window."""
    raw_frames = int(num_raw_frames)
    if raw_frames <= 0:
        raise ValueError(f"num_raw_frames must be positive, got {raw_frames}")
    stride = resolve_main_encoder_frame_stride(config, encoder)
    if raw_frames % stride != 0:
        raise ValueError(f"num_raw_frames ({raw_frames}) must be divisible by main encoder stride ({stride})")
    return raw_frames // stride


def resolve_main_encoder_num_observed_steps(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve observed predictor steps for the main encoder."""
    num_observed = int(_get_nested_value(config, "train", "num_observed_frames", default=1))
    return resolve_main_encoder_num_time_steps(config, num_raw_frames=num_observed, encoder=encoder)


def resolve_main_encoder_predictor_img_size(config: Any, encoder: Optional[Any] = None):
    """Resolve predictor image/grid size for the main encoder."""
    if is_drive_jepa_main_encoder_config(config):
        _validate_drive_jepa_main_token_ae(config)
        core = encoder.module if hasattr(encoder, "module") else encoder
        resolution = getattr(core, "resolution", None) if core is not None else None
        if resolution is None:
            resolution = _get_nested_value(config, "model", "drive_jepa_resolution", default=(256, 512))
        return _parse_hw_resolution(resolution, "model.drive_jepa_resolution")
    return int(_get_nested_value(config, "data", "crop_size", default=256))


@dataclass
class MetaConfig:
    """元配置：训练相关的通用设置"""

    folder: str = ""
    seed: int = 0
    dtype: str = "bfloat16"
    resume_checkpoint: Optional[str] = None
    pretrain_checkpoint: Optional[str] = None
    pretrain_repo: Optional[str] = None
    pretrain_checkpoint_full: Optional[str] = None
    predictor_checkpoint: Optional[str] = (
        None  # predictor 独立 checkpoint，优先于 pretrain_checkpoint_full 中的 predictor
    )
    ae_checkpoint: Optional[str] = None
    load_encoder: bool = True
    load_predictor: bool = False
    load_seg: bool = True
    load_planner: bool = True
    context_encoder_key: str = "encoder"
    target_encoder_key: str = "target_encoder"
    save_every_freq: int = -1
    skip_batches: int = -1
    use_sdpa: bool = False
    sync_gc: bool = False
    val_freq: int = 5  # 验证频率
    resume_broadcast: bool = False  # resume 时是否通过 broadcast 分发 checkpoint（False 则每个 rank 独立读盘）
    resume_model_only: bool = False  # 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）
    auto_resume_latest: bool = False


@dataclass
class ModelConfig:
    """模型配置：模型架构相关设置"""

    model_name: str = ""
    backbone: str = "vjepa2"  # "vjepa2" or "vjepa2.1" or "drive_jepa_img_encoder"
    drive_jepa_resolution: Tuple[int, int] = (256, 512)
    drive_jepa_crop_top_bottom: int = 28
    drive_jepa_num_frames: int = 2
    drive_jepa_checkpoint_key: str = "target_encoder"
    drive_jepa_use_grid_mask: bool = True
    drive_jepa_use_causal_attention: bool = True
    patch_size: int = 16
    pred_depth: int = 12
    pred_num_heads: Optional[int] = None
    pred_embed_dim: int = 384
    pred_is_frame_causal: bool = True
    uniform_power: bool = False
    use_rope: bool = False
    use_silu: bool = False
    use_pred_silu: bool = False
    wide_silu: bool = True
    use_extrinsics: bool = False
    use_mask_tokens: bool = False
    zero_init_mask_tokens: bool = True
    compile_model: bool = False
    use_activation_checkpointing: bool = False


@dataclass
class TrainConfig:
    """训练配置：训练策略相关设置"""

    encoder_train: bool = False
    seg_head: bool = True
    encoder_ema: bool = False
    perceiver_ema: bool = True
    predictor_train: bool = True
    use_states_for_predictor: bool = True
    action_dim: int = 7
    state_dim: int = 7  # IC状态维度: 7=drive_command_7, 8=drive_command_8
    command_dim: int = 0  # >0: 拆分 state[0:command_dim] 为 command token，剩余为 kinematics token
    use_drive_command: bool = True  # False 时从 status 向量去掉 drive_command 的 4 维 one-hot
    predictor_inference_consistent: bool = False
    use_parallel_predictor: bool = False
    predictor_use_z_ar_supervision: bool = True
    reuse_context_as_target_when_frozen: bool = False
    predictor_no_aux_input: bool = False
    # Raw-frame semantics for diffusion / world-model generation tasks:
    # - num_encoder_frames: how many observed image frames are exposed to the frozen encoder
    # - num_predictor_frames: how many future image frames the frozen predictor should autoregressively generate
    # If num_predictor_frames is None, downstream code derives it as
    # `data.num_target_frames - num_encoder_frames`.
    #
    # `num_observed_frames` is kept for backward compatibility with older training
    # scripts that still use the previous name.
    num_encoder_frames: int = 2
    num_predictor_frames: Optional[int] = None
    num_observed_frames: int = 2


@dataclass
class EMAConfig:
    """EMA 配置：指数移动平均相关设置"""

    ema_start: float = 0.996
    ema_end: float = 0.999

    @property
    def ema_range(self) -> Tuple[float, float]:
        return (self.ema_start, self.ema_end)


@dataclass
class SegmentationConfig:
    """分割配置：分割模块相关设置"""

    use_segmentation: bool = True
    seg_loss_weight: float = 1.0
    seg_data_root: str = ""


@dataclass
class PlannerConfig:
    """Planner 配置：轨迹预测模块相关设置"""

    use_planner: bool = True
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    planner_loss_weight: float = 1.0
    use_spatial_tokens: bool = False
    use_temporal: bool = False
    temporal_alignment: bool = True
    z_ar_mode: str = "full"
    planner_input_source: str = "z_ar"  # "z_ar" (自回归展开) 或 "z_tf" (teacher forcing)
    time_aligned_bias_scope: str = "all_tokens"
    num_modes: int = 6
    num_context_frames: int = 1
    conf_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    horizon_reg_loss_seconds: List[float] = field(default_factory=list)
    horizon_reg_loss_weights: List[float] = field(default_factory=list)
    horizon_reg_loss_normalize: bool = True
    states_mode: str = "first"
    use_status_for_planner: bool = True
    use_states_for_planner: bool = True
    use_z_context: bool = False
    # 观测 tokens 配置
    use_observed_tokens: bool = False  # 是否将观测帧 encoder tokens 与预测 tokens 拼接后输入 planner
    use_action_history_for_planner: bool = False
    action_history_dim: int = 3
    enable_rl_actor_critic: bool = False
    rl_action_dim: int = 2
    # WTA 损失配置
    wta_loss_version: str = "v1"
    wta_temperature: float = 1.0
    cover_loss_weight: float = 0.1
    # aWTA (v3) 专用参数
    awta_init_temperature: float = 8.0
    awta_exp_base: float = 0.984
    awta_min_temperature: float = 0.1
    # Planner 类型选择
    planner_type: str = "transformer"  # "transformer" (原始) 或 "diffusion" (扩散)
    refinement_core_type: Optional[str] = None  # Stage-3 refinement core; None 时继承 planner_type
    # Diffusion planner 专用参数
    diff_hidden_dim: int = 256
    diff_num_layers: int = 4
    diff_num_heads: int = 8
    diff_dropout: float = 0.0
    diff_mlp_ratio: float = 4.0
    diff_sde_beta_min: float = 0.1
    diff_sde_beta_max: float = 20.0
    diff_inference_steps: int = 2  # DPM-Solver++ 采样步数
    diff_num_samples: int = 6  # K 个噪声样本用于多模态
    diff_traj_dim: int = 6  # 轨迹维度 (x, y, vx, vy, cos_yaw, sin_yaw)
    diff_dt: float = 0.2  # 帧间时间间隔 (秒), 用于速度计算
    diff_trajectory_token_mode: str = "single_token"  # "single_token" 或 "per_pose_token"
    diff_adaln_version: str = (
        "legacy"  # "legacy"=旧 6-param adaLN, "v2"=9-param full adaLN, "v3"=legacy 结构但去掉 cross/mlp2 残差
    )
    diff_use_last_frame_only: bool = True  # True: 仅用 z_ar 最后一帧做 cross-attn (AR causal → 最后帧含全部信息)
    diff_interleave_predictor_sampling: bool = False  # 推理时 predictor 每产出一帧 prefix，就推进若干 diffusion step
    diff_train_prefix_conditioning: bool = False
    diff_train_min_prefix_frames: int = 1
    diff_train_full_prefix_prob: float = 0.25
    diff_num_modes: int = 1
    diff_independent_modes: bool = False  # True: B->B*K independent processing (anti-collapse); False: joint XTR
    # Architectural anti-collapse: expand K modes into the DiT sequence dimension
    # (with learnable mode embeddings) so every block can diversify modes via
    # self-attention.  Only meaningful for per_pose_token + num_modes>1; ignored
    # otherwise.  Default False preserves checkpoint compatibility.
    diff_mode_token_expansion: bool = False
    diff_use_anchor_frame: bool = False
    # Diffusion seed initialization (inference x_T/x_t init)
    # "gaussian": pure noise (legacy behavior)
    # "kinematic": constant-velocity prior + optional mode spread + noise
    diff_init_traj_strategy: str = "gaussian"
    diff_init_traj_noise_scale: float = 1.0
    diff_init_traj_yaw_span_deg: float = 30.0
    diff_init_traj_speed_scale_span: float = 0.2
    diff_cls_loss_weight: float = 1.0
    diff_reg_loss_weight: float = 1.0
    diff_vel_loss_weight: float = 0.5
    diff_yaw_loss_weight: float = 0.5
    # Hybrid WTA (aWTA reg + XTR gated soft-CE) 专用超参
    diff_conf_temperature: float = 1.5
    diff_cls_th: float = 2.0
    diff_cls_ignore: float = 0.2
    # Status 拆分嵌入：将分类（导航指令 one-hot）与连续（运动学）分量独立嵌入
    split_status_embedding: bool = True
    # Planner 侧是否保留 drive_command 4 维；None 时继承 train.use_drive_command
    use_drive_command: Optional[bool] = None
    # Planner status 维度（与 predictor.state_dim 解耦）：
    #   0  = 继承 train.state_dim（向后兼容，默认）
    #   12 = cmd(4) + dyn(4) + pose(4, x_local/y_local/sin_yaw/cos_yaw)
    status_dim: int = 0


@dataclass
class ProposalConfig:
    """独立 proposal provider 配置。"""

    enabled: bool = False
    provider_type: str = "transformer"  # transformer | diffusion | history_kinematic
    checkpoint: Optional[str] = None
    use_separate_encoder: bool = False
    encoder_backbone: Optional[str] = None
    encoder_model_name: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    encoder_checkpoint_key: str = "encoder"
    encoder_freeze: bool = True
    drive_jepa_resolution: Tuple[int, int] = (256, 512)
    drive_jepa_crop_top_bottom: int = 28
    drive_jepa_num_frames: int = 2
    drive_jepa_checkpoint_key: Optional[str] = None
    drive_jepa_use_grid_mask: bool = True
    drive_jepa_use_causal_attention: bool = True
    freeze: bool = True
    num_modes: int = 6
    provider_num_modes: Optional[int] = None
    log_metrics_only: bool = True
    use_z_context: bool = True
    temporal_alignment: bool = True
    runtime_normalize_reps: Optional[bool] = None
    use_token_ae: Optional[bool] = None
    history_temperature: float = 1.0
    hidden_dim: int = 256
    manual_mode_expansion: bool = False
    manual_lateral_offsets: Optional[List[float]] = None
    manual_yaw_offsets_deg: Optional[List[float]] = None
    manual_speed_scales: Optional[List[float]] = None
    manual_ramp_power: float = 1.5
    manual_confidence_temperature: float = 1.0


@dataclass
class NavSimConfig:
    """NavSim 数据配置"""

    enabled: bool = False
    data_path: str = ""
    sensor_blobs_path: str = ""
    val_data_path: Optional[str] = None
    val_sensor_blobs_path: Optional[str] = None
    camera_name: str = "CAM_F0"
    num_history_frames: Optional[int] = None
    num_future_frames: Optional[int] = None
    max_scenes: Optional[int] = None
    max_val_scenes: Optional[int] = None
    index_cache: bool = True  # 缓存场景索引到磁盘，避免每次启动重复扫描 pkl 文件
    window_stride: int = 1  # 训练集滑窗步长（帧），1=最大重叠，等于 frames_per_clip=无重叠
    val_window_stride: Optional[int] = None  # 验证集独立步长，None 时回退到 window_stride
    max_frame_gap: int = 3  # 窗口内相邻 valid 帧之间允许的最大原始帧索引差，超出则丢弃该窗口


@dataclass
class MongoRawConfig:
    """Mongo + raw clip 在线数据配置"""

    enabled: bool = False
    mongo_uri: Optional[str] = None
    mongo_uri_env: Optional[str] = None
    database: str = "e2e-data-platform-prod"
    collection: str = "clip"
    vehicle_type: Optional[str] = None
    vehicle_types: List[str] = field(default_factory=list)
    require_latest_available_revision: bool = True
    query_filter: Dict[str, Any] = field(default_factory=dict)
    start_index: int = 0
    end_index: Optional[int] = None
    max_clips: Optional[int] = None
    max_val_clips: Optional[int] = None
    val_ratio: float = 0.05
    split_seed: int = 0
    source_fps: int = 10
    base_fps: int = 5
    main_topic: str = "/main/ruby/lidar_points"
    pose_topic: str = "/pose/odom"
    match_topic: str = "/match"
    camera_topics: List[str] = field(default_factory=list)
    default_storage_root: str = ""
    e2e_storage_root: str = ""
    clipdata_storage_root: str = ""
    cache_size: int = 8
    max_retries: int = 5
    extra_camera_mappings: Dict[str, str] = field(default_factory=dict)
    record_cache_dir: Optional[str] = None  # 缓存目录，None 则不缓存
    record_cache_ttl: int = 604800  # 缓存过期时间（秒），默认 7 天
    blacklist_path: Optional[str] = None  # 坏 clip ID 黑名单 JSON 路径，None 则不使用


@dataclass
class DataConfig:
    """数据配置：数据集和加载器相关设置"""

    datasets: List[str] = field(default_factory=list)
    val_datasets: Optional[List[str]] = None
    dataset_fpcs: List[int] = field(default_factory=list)
    batch_size: int = 4
    tubelet_size: int = 2
    use_tubelet_repeat: bool = True
    fps: int = 5
    crop_size: Tuple[int, int] = (256, 256)
    patch_size: int = 16
    num_target_frames: int = 16
    pin_mem: bool = False
    num_workers: int = 1
    persistent_workers: bool = True
    camera_frame: bool = False
    camera_views: List[str] = field(default_factory=lambda: ["left_mp4_path"])
    stereo_view: bool = False
    navsim: Optional[NavSimConfig] = None
    mongo_raw: Optional[MongoRawConfig] = None

    @property
    def dataset_path(self) -> Optional[str]:
        return self.datasets[0] if self.datasets else None

    @property
    def val_dataset_path(self) -> Optional[str]:
        return self.val_datasets[0] if self.val_datasets else None

    @property
    def max_num_frames(self) -> int:
        return max(self.dataset_fpcs) if self.dataset_fpcs else 16

    @property
    def crop_height(self) -> int:
        return self.crop_size[0]

    @property
    def crop_width(self) -> int:
        return self.crop_size[1]

    @property
    def tokens_per_frame(self) -> int:
        return compute_tokens_per_frame(self.crop_size, self.patch_size)


@dataclass
class DataAugConfig:
    """数据增强配置"""

    horizontal_flip: bool = False
    random_resize_aspect_ratio: List[float] = field(default_factory=lambda: [3 / 4, 4 / 3])
    random_resize_scale: List[float] = field(default_factory=lambda: [0.3, 1.0])
    motion_shift: bool = False
    reprob: float = 0.0
    auto_augment: bool = False


@dataclass
class TokenAEConfig:
    """Token AE 配置"""

    enabled: bool = False
    num_latent_tokens: int = 64
    num_heads: int = 16
    encoder_depth: int = 4
    decoder_depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    encoder_mode: str = "parallel"
    loss_type: str = "smooth_l1"
    cos_loss_weight: float = 0.25
    latent_reg_weight: float = 0.0
    pos_embed_type: str = "sincos"
    input_grid_size: Optional[Tuple[int, int]] = None
    latent_grid_size: Optional[Tuple[int, int]] = None
    temporal_depth: int = 0
    temporal_num_heads: Optional[int] = None
    temporal_mlp_ratio: Optional[float] = None
    temporal_causal: bool = True
    temporal_mode: str = "index"
    temporal_pos_embed_type: str = "none"
    input_frame_mode: str = "all_frames"
    temporal_loss_weight: float = 0.0


@dataclass
class LeWMConfig:
    """le-wm JEPA pipeline 配置"""

    enabled: bool = False
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    projector_hidden_dim: int = 2048
    embed_dim: int = 192


@dataclass
class StageLossWeightsConfig:
    """Stage-2 planner 损失权重。"""

    prop: float = 1.0
    refine: float = 1.0
    anchor: float = 0.0
    div: float = 0.0


@dataclass
class Stage2Config:
    """LeWM Stage-2 proposal/refinement 配置。"""

    num_modes: int = 6
    inference_num_rounds: int = 1
    predictor_rollout_seconds: Optional[float] = None
    detach_zfut: bool = True
    refine_stop_grad_to_proposer: bool = True
    refine_use_random_predictor_latent: bool = False
    refine_keep_initial_actions: bool = False
    predictor_finetune: bool = False
    checkpoint: Optional[str] = None
    lambdas: StageLossWeightsConfig = field(default_factory=StageLossWeightsConfig)


@dataclass
class Stage3Config:
    """LeWM Stage-3 iterative planner 配置。"""

    num_rounds: int = 2
    round_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    predictor_rollout_seconds: Optional[float] = None
    grad_checkpoint: bool = True
    predictor_finetune: bool = False
    checkpoint: Optional[str] = None
    fut_consistency_weight: float = 0.0
    refine_use_z_context: bool = True
    refine_use_status_feature: bool = True
    refine_use_proposal_traj: bool = True
    refine_use_proposal_logits: bool = True
    refine_use_proposal_features: bool = True
    refine_use_predictor_rollout: bool = True
    refine_use_random_predictor_latent: bool = False
    refine_keep_initial_actions: bool = False
    use_multimodal_final: bool = False


@dataclass
class LossConfig:
    """损失函数配置"""

    loss_exp: float = 2.0
    normalize_reps: bool = True
    auto_steps: int = 1


@dataclass
class RLConfig:
    """闭环 RL 配置"""

    enabled: bool = False
    algo: str = "ppo"
    status_mode: str = "current_only"
    hugsim_repo_root: Optional[str] = None
    scenario_path: Optional[str] = None
    scenario_manifest: Optional[str] = None
    base_path: Optional[str] = None
    camera_path: Optional[str] = None
    kinematic_path: Optional[str] = None
    camera_name: str = "CAM_FRONT"
    output_subdir: str = "hugsim_rl"
    eval_checkpoint: Optional[str] = None
    rollout_steps: int = 128
    max_episode_steps: int = 400
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    reward_scale: float = 1.0
    rl_loss_weight: float = 1.0
    supervised_loss_weight: float = 0.0
    supervised_warmup_epochs: int = 0
    supervised_batches_per_epoch: int = 0
    normalize_advantage: bool = True
    deterministic_eval: bool = True
    eval_episodes: int = 1
    wheel_base: float = 2.7
    kinematic_dt: float = 0.25


@dataclass
class OptimizationConfig:
    """优化器配置"""

    ipe: Optional[int] = None
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    epochs: int = 100
    anneal: int = 1
    warmup: int = 10
    start_lr: float = 0.0001
    lr: float = 0.0001
    final_lr: float = 0.0
    enc_lr_scale: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    is_anneal: bool = False
    anneal_ckpt: Optional[str] = None
    resume_anneal: bool = False
    ipe_scale: float = 1.0


@dataclass
class TrainingConfig:
    """完整的训练配置"""

    meta: MetaConfig = field(default_factory=MetaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    data_aug: DataAugConfig = field(default_factory=DataAugConfig)
    token_ae: TokenAEConfig = field(default_factory=TokenAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    lewm: LeWMConfig = field(default_factory=LeWMConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    rl: RLConfig = field(default_factory=RLConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def dtype(self) -> torch.dtype:
        """获取 PyTorch dtype"""
        dtype_str = self.meta.dtype.lower()
        if dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float16":
            return torch.float16
        else:
            return torch.float32

    @property
    def mixed_precision(self) -> bool:
        """是否使用混合精度"""
        return self.meta.dtype.lower() in ("bfloat16", "float16")

    @property
    def effective_tokens_per_frame(self) -> int:
        """运行时每帧有效 token 数，Token AE 开启时返回压缩后的 latent token 数。"""
        return resolve_effective_tokens_per_frame(self)


def parse_training_config(args: Dict[str, Any]) -> TrainingConfig:
    """
    从原始 args 字典解析配置

    Args:
        args: 从 YAML 文件加载的配置字典

    Returns:
        TrainingConfig: 结构化的配置对象
    """
    # 解析 meta 配置
    cfgs_meta = args.get("meta", {})
    meta = MetaConfig(
        folder=args.get("folder", ""),
        seed=cfgs_meta.get("seed", 0),
        dtype=cfgs_meta.get("dtype", "bfloat16"),
        resume_checkpoint=cfgs_meta.get("resume_checkpoint"),
        pretrain_checkpoint=cfgs_meta.get("pretrain_checkpoint"),
        pretrain_repo=cfgs_meta.get("pretrain_repo"),
        pretrain_checkpoint_full=cfgs_meta.get("pretrain_checkpoint_full"),
        predictor_checkpoint=cfgs_meta.get("predictor_checkpoint"),
        ae_checkpoint=cfgs_meta.get("ae_checkpoint"),
        load_encoder=cfgs_meta.get("load_encoder", True),
        load_predictor=cfgs_meta.get("load_predictor", False),
        load_seg=cfgs_meta.get("load_seg", True),
        load_planner=cfgs_meta.get("load_planner", True),
        context_encoder_key=cfgs_meta.get("context_encoder_key", "encoder"),
        target_encoder_key=cfgs_meta.get("target_encoder_key", "target_encoder"),
        save_every_freq=cfgs_meta.get("save_every_freq", -1),
        skip_batches=cfgs_meta.get("skip_batches", -1),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        sync_gc=cfgs_meta.get("sync_gc", False),
        val_freq=cfgs_meta.get("val_freq", 5),
        resume_broadcast=cfgs_meta.get("resume_broadcast", False),
        resume_model_only=cfgs_meta.get("resume_model_only", False),
        auto_resume_latest=cfgs_meta.get("auto_resume_latest", False),
    )

    # 解析 model 配置
    cfgs_model = args.get("model", {})
    drive_jepa_resolution = _parse_hw_resolution(
        cfgs_model.get("drive_jepa_resolution", (256, 512)),
        "model.drive_jepa_resolution",
    )
    model = ModelConfig(
        model_name=cfgs_model.get("model_name", ""),
        backbone=cfgs_model.get("backbone", "vjepa2"),
        drive_jepa_resolution=drive_jepa_resolution,
        drive_jepa_crop_top_bottom=cfgs_model.get("drive_jepa_crop_top_bottom", 28),
        drive_jepa_num_frames=cfgs_model.get("drive_jepa_num_frames", 2),
        drive_jepa_checkpoint_key=cfgs_model.get("drive_jepa_checkpoint_key", "target_encoder"),
        drive_jepa_use_grid_mask=cfgs_model.get("drive_jepa_use_grid_mask", True),
        drive_jepa_use_causal_attention=cfgs_model.get("drive_jepa_use_causal_attention", True),
        patch_size=cfgs_model.get("patch_size", 16),
        pred_depth=cfgs_model.get("pred_depth", 12),
        pred_num_heads=cfgs_model.get("pred_num_heads"),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        pred_is_frame_causal=cfgs_model.get("pred_is_frame_causal", True),
        uniform_power=cfgs_model.get("uniform_power", False),
        use_rope=cfgs_model.get("use_rope", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        use_extrinsics=cfgs_model.get("use_extrinsics", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        compile_model=cfgs_model.get("compile_model", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
    )

    cfgs_lewm = args.get("lewm", {})

    # 解析 train 配置
    cfgs_train = args.get("train", {})
    observed_frames = cfgs_train.get("num_encoder_frames", cfgs_train.get("num_observed_frames", 2))
    predictor_use_z_ar_supervision_default = not bool(cfgs_lewm.get("enabled", False))
    train = TrainConfig(
        encoder_train=cfgs_train.get("encoder_train", False),
        seg_head=cfgs_train.get("seg_head", True),
        encoder_ema=cfgs_train.get("encoder_ema", False),
        perceiver_ema=cfgs_train.get("perceiver_ema", True),
        predictor_train=cfgs_train.get("predictor_train", True),
        use_states_for_predictor=cfgs_train.get("use_states_for_predictor", True),
        action_dim=cfgs_train.get("action_dim", 7),
        state_dim=cfgs_train.get("state_dim", 7),
        command_dim=cfgs_train.get("command_dim", 0),
        use_drive_command=cfgs_train.get("use_drive_command", True),
        predictor_inference_consistent=cfgs_train.get("predictor_inference_consistent", False),
        use_parallel_predictor=cfgs_train.get("use_parallel_predictor", False),
        predictor_use_z_ar_supervision=cfgs_train.get(
            "predictor_use_z_ar_supervision",
            predictor_use_z_ar_supervision_default,
        ),
        reuse_context_as_target_when_frozen=cfgs_train.get("reuse_context_as_target_when_frozen", False),
        predictor_no_aux_input=cfgs_train.get("predictor_no_aux_input", False),
        num_encoder_frames=observed_frames,
        num_predictor_frames=cfgs_train.get("num_predictor_frames"),
        num_observed_frames=observed_frames,
    )

    # 解析 EMA 配置
    cfgs_ema = args.get("ema", {})
    ema = EMAConfig(
        ema_start=cfgs_ema.get("ema_start", 0.996),
        ema_end=cfgs_ema.get("ema_end", 0.999),
    )

    # 解析 segmentation 配置
    cfgs_seg = args.get("segmentation", {})
    segmentation = SegmentationConfig(
        use_segmentation=cfgs_seg.get("use_segmentation", True),
        seg_loss_weight=cfgs_seg.get("seg_loss_weight", 1.0),
        seg_data_root=cfgs_seg.get("seg_data_root", ""),
    )

    # 解析 planner 配置
    cfgs_planner = args.get("planner", {})
    planner = PlannerConfig(
        use_planner=cfgs_planner.get("use_planner", True),
        tf_d_model=cfgs_planner.get("tf_d_model", 256),
        tf_d_ffn=cfgs_planner.get("tf_d_ffn", 1024),
        tf_num_layers=cfgs_planner.get("tf_num_layers", 3),
        tf_num_head=cfgs_planner.get("tf_num_head", 8),
        tf_dropout=cfgs_planner.get("tf_dropout", 0.0),
        planner_loss_weight=cfgs_planner.get("planner_loss_weight", 1.0),
        use_spatial_tokens=cfgs_planner.get("use_spatial_tokens", False),
        use_temporal=cfgs_planner.get("use_temporal", False),
        temporal_alignment=cfgs_planner.get("temporal_alignment", True),
        z_ar_mode=cfgs_planner.get("z_ar_mode", "full"),
        planner_input_source=cfgs_planner.get("planner_input_source", "z_ar"),
        time_aligned_bias_scope=cfgs_planner.get("time_aligned_bias_scope", "all_tokens"),
        num_modes=cfgs_planner.get("num_modes", 6),
        num_context_frames=cfgs_planner.get("num_context_frames", 1),
        conf_loss_weight=cfgs_planner.get("conf_loss_weight", 1.0),
        reg_loss_weight=cfgs_planner.get("reg_loss_weight", 1.0),
        horizon_reg_loss_seconds=cfgs_planner.get("horizon_reg_loss_seconds", []),
        horizon_reg_loss_weights=cfgs_planner.get("horizon_reg_loss_weights", []),
        horizon_reg_loss_normalize=cfgs_planner.get("horizon_reg_loss_normalize", True),
        states_mode=cfgs_planner.get("states_mode", cfgs_planner.get("status_mode", "first")),
        use_status_for_planner=cfgs_planner.get("use_status_for_planner", cfgs_planner.get("use_status", True)),
        use_states_for_planner=cfgs_planner.get("use_states_for_planner", True),
        use_z_context=cfgs_planner.get("use_z_context", False),
        use_observed_tokens=cfgs_planner.get("use_observed_tokens", False),
        use_action_history_for_planner=cfgs_planner.get("use_action_history_for_planner", False),
        action_history_dim=cfgs_planner.get("action_history_dim", 3),
        enable_rl_actor_critic=cfgs_planner.get("enable_rl_actor_critic", False),
        rl_action_dim=cfgs_planner.get("rl_action_dim", 2),
        wta_loss_version=cfgs_planner.get("wta_loss_version", "v1"),
        wta_temperature=cfgs_planner.get("wta_temperature", 1.0),
        cover_loss_weight=cfgs_planner.get("cover_loss_weight", 0.1),
        awta_init_temperature=cfgs_planner.get("awta_init_temperature", 8.0),
        awta_exp_base=cfgs_planner.get("awta_exp_base", 0.984),
        awta_min_temperature=cfgs_planner.get("awta_min_temperature", 0.1),
        planner_type=cfgs_planner.get("planner_type", "transformer"),
        refinement_core_type=cfgs_planner.get("refinement_core_type"),
        diff_hidden_dim=cfgs_planner.get("diff_hidden_dim", 256),
        diff_num_layers=cfgs_planner.get("diff_num_layers", 4),
        diff_num_heads=cfgs_planner.get("diff_num_heads", 8),
        diff_dropout=cfgs_planner.get("diff_dropout", 0.0),
        diff_mlp_ratio=cfgs_planner.get("diff_mlp_ratio", 4.0),
        diff_sde_beta_min=cfgs_planner.get("diff_sde_beta_min", 0.1),
        diff_sde_beta_max=cfgs_planner.get("diff_sde_beta_max", 20.0),
        diff_inference_steps=cfgs_planner.get("diff_inference_steps", 2),
        diff_num_samples=cfgs_planner.get("diff_num_samples", 6),
        diff_traj_dim=cfgs_planner.get("diff_traj_dim", 6),
        diff_dt=cfgs_planner.get("diff_dt", 0.2),
        diff_trajectory_token_mode=cfgs_planner.get("diff_trajectory_token_mode", "single_token"),
        diff_adaln_version=cfgs_planner.get("diff_adaln_version", "legacy"),
        diff_use_last_frame_only=cfgs_planner.get("diff_use_last_frame_only", True),
        diff_interleave_predictor_sampling=cfgs_planner.get("diff_interleave_predictor_sampling", False),
        diff_train_prefix_conditioning=cfgs_planner.get("diff_train_prefix_conditioning", False),
        diff_train_min_prefix_frames=cfgs_planner.get("diff_train_min_prefix_frames", 1),
        diff_train_full_prefix_prob=cfgs_planner.get("diff_train_full_prefix_prob", 0.25),
        diff_num_modes=cfgs_planner.get("diff_num_modes", 1),
        diff_independent_modes=cfgs_planner.get("diff_independent_modes", False),
        diff_mode_token_expansion=cfgs_planner.get("diff_mode_token_expansion", False),
        diff_use_anchor_frame=cfgs_planner.get("diff_use_anchor_frame", False),
        diff_init_traj_strategy=cfgs_planner.get("diff_init_traj_strategy", "gaussian"),
        diff_init_traj_noise_scale=cfgs_planner.get("diff_init_traj_noise_scale", 1.0),
        diff_init_traj_yaw_span_deg=cfgs_planner.get("diff_init_traj_yaw_span_deg", 30.0),
        diff_init_traj_speed_scale_span=cfgs_planner.get("diff_init_traj_speed_scale_span", 0.2),
        diff_cls_loss_weight=cfgs_planner.get("diff_cls_loss_weight", 1.0),
        diff_reg_loss_weight=cfgs_planner.get("diff_reg_loss_weight", 1.0),
        diff_vel_loss_weight=cfgs_planner.get("diff_vel_loss_weight", 0.5),
        diff_yaw_loss_weight=cfgs_planner.get("diff_yaw_loss_weight", 0.5),
        diff_conf_temperature=cfgs_planner.get("diff_conf_temperature", 1.5),
        diff_cls_th=cfgs_planner.get("diff_cls_th", 2.0),
        diff_cls_ignore=cfgs_planner.get("diff_cls_ignore", 0.2),
        split_status_embedding=cfgs_planner.get("split_status_embedding", True),
        use_drive_command=cfgs_planner.get("use_drive_command"),
        status_dim=cfgs_planner.get("status_dim", 0),
    )

    # 解析独立 proposal 配置
    cfgs_proposal = args.get("proposal", {}) or {}
    proposal_drive_jepa_resolution = _parse_hw_resolution(
        cfgs_proposal.get("drive_jepa_resolution", model.drive_jepa_resolution),
        "proposal.drive_jepa_resolution",
    )
    proposal = ProposalConfig(
        enabled=bool(cfgs_proposal.get("enabled", False)),
        provider_type=cfgs_proposal.get("provider_type", cfgs_planner.get("planner_type", "transformer")),
        checkpoint=cfgs_proposal.get("checkpoint"),
        use_separate_encoder=bool(cfgs_proposal.get("use_separate_encoder", False)),
        encoder_backbone=cfgs_proposal.get("encoder_backbone"),
        encoder_model_name=cfgs_proposal.get("encoder_model_name", model.model_name),
        encoder_checkpoint=cfgs_proposal.get("encoder_checkpoint"),
        encoder_checkpoint_key=cfgs_proposal.get("encoder_checkpoint_key", "encoder"),
        encoder_freeze=bool(cfgs_proposal.get("encoder_freeze", True)),
        drive_jepa_resolution=proposal_drive_jepa_resolution,
        drive_jepa_crop_top_bottom=cfgs_proposal.get("drive_jepa_crop_top_bottom", model.drive_jepa_crop_top_bottom),
        drive_jepa_num_frames=cfgs_proposal.get("drive_jepa_num_frames", model.drive_jepa_num_frames),
        drive_jepa_checkpoint_key=cfgs_proposal.get("drive_jepa_checkpoint_key"),
        drive_jepa_use_grid_mask=cfgs_proposal.get("drive_jepa_use_grid_mask", model.drive_jepa_use_grid_mask),
        drive_jepa_use_causal_attention=cfgs_proposal.get(
            "drive_jepa_use_causal_attention",
            model.drive_jepa_use_causal_attention,
        ),
        freeze=bool(cfgs_proposal.get("freeze", True)),
        num_modes=int(cfgs_proposal.get("num_modes", cfgs_planner.get("num_modes", 6))),
        provider_num_modes=(
            int(cfgs_proposal["provider_num_modes"]) if cfgs_proposal.get("provider_num_modes") is not None else None
        ),
        log_metrics_only=bool(cfgs_proposal.get("log_metrics_only", True)),
        use_z_context=bool(cfgs_proposal.get("use_z_context", cfgs_planner.get("use_z_context", True))),
        temporal_alignment=bool(cfgs_proposal.get("temporal_alignment", cfgs_planner.get("temporal_alignment", True))),
        runtime_normalize_reps=cfgs_proposal.get("runtime_normalize_reps"),
        use_token_ae=cfgs_proposal.get("use_token_ae"),
        history_temperature=float(cfgs_proposal.get("history_temperature", 1.0)),
        hidden_dim=int(cfgs_proposal.get("hidden_dim", cfgs_planner.get("tf_d_model", 256))),
        manual_mode_expansion=bool(cfgs_proposal.get("manual_mode_expansion", False)),
        manual_lateral_offsets=cfgs_proposal.get("manual_lateral_offsets"),
        manual_yaw_offsets_deg=cfgs_proposal.get("manual_yaw_offsets_deg"),
        manual_speed_scales=cfgs_proposal.get("manual_speed_scales"),
        manual_ramp_power=float(cfgs_proposal.get("manual_ramp_power", 1.5)),
        manual_confidence_temperature=float(cfgs_proposal.get("manual_confidence_temperature", 1.0)),
    )

    # 解析 data 配置
    cfgs_data = args.get("data", {})
    cfgs_navsim = cfgs_data.get("navsim")
    cfgs_mongo_raw = cfgs_data.get("mongo_raw")

    navsim = None
    if isinstance(cfgs_navsim, dict):
        front_only = cfgs_navsim.get("front_only", True)
        default_camera_name = "CAM_F0" if front_only else "CAM_F0"
        navsim = NavSimConfig(
            enabled=cfgs_navsim.get("enabled", True),
            data_path=cfgs_navsim.get("data_path", ""),
            sensor_blobs_path=cfgs_navsim.get("sensor_blobs_path", ""),
            val_data_path=cfgs_navsim.get("val_data_path"),
            val_sensor_blobs_path=cfgs_navsim.get("val_sensor_blobs_path"),
            camera_name=cfgs_navsim.get("camera_name", default_camera_name),
            num_history_frames=cfgs_navsim.get("num_history_frames"),
            num_future_frames=cfgs_navsim.get("num_future_frames"),
            max_scenes=cfgs_navsim.get("max_scenes"),
            max_val_scenes=cfgs_navsim.get("max_val_scenes"),
            index_cache=cfgs_navsim.get("index_cache", True),
            window_stride=cfgs_navsim.get("window_stride", 1),
            val_window_stride=cfgs_navsim.get("val_window_stride", None),
            max_frame_gap=cfgs_navsim.get("max_frame_gap", 3),
        )

    mongo_raw = None
    if isinstance(cfgs_mongo_raw, dict):
        mongo_raw = MongoRawConfig(
            enabled=cfgs_mongo_raw.get("enabled", True),
            mongo_uri=cfgs_mongo_raw.get("mongo_uri"),
            mongo_uri_env=cfgs_mongo_raw.get("mongo_uri_env"),
            database=cfgs_mongo_raw.get("database", "e2e-data-platform-prod"),
            collection=cfgs_mongo_raw.get("collection", "clip"),
            vehicle_type=cfgs_mongo_raw.get("vehicle_type"),
            vehicle_types=list(cfgs_mongo_raw.get("vehicle_types", []) or []),
            require_latest_available_revision=cfgs_mongo_raw.get("require_latest_available_revision", True),
            query_filter=cfgs_mongo_raw.get("query_filter", {}) or {},
            start_index=cfgs_mongo_raw.get("start_index", 0),
            end_index=cfgs_mongo_raw.get("end_index"),
            max_clips=cfgs_mongo_raw.get("max_clips"),
            max_val_clips=cfgs_mongo_raw.get("max_val_clips"),
            val_ratio=cfgs_mongo_raw.get("val_ratio", 0.05),
            split_seed=cfgs_mongo_raw.get("split_seed", 0),
            source_fps=cfgs_mongo_raw.get("source_fps", 10),
            base_fps=cfgs_mongo_raw.get("base_fps", 5),
            main_topic=cfgs_mongo_raw.get("main_topic", "/main/ruby/lidar_points"),
            pose_topic=cfgs_mongo_raw.get("pose_topic", "/pose/odom"),
            match_topic=cfgs_mongo_raw.get("match_topic", "/match"),
            camera_topics=cfgs_mongo_raw.get("camera_topics", []),
            default_storage_root=cfgs_mongo_raw.get("default_storage_root", ""),
            e2e_storage_root=cfgs_mongo_raw.get("e2e_storage_root", ""),
            clipdata_storage_root=cfgs_mongo_raw.get("clipdata_storage_root", ""),
            cache_size=cfgs_mongo_raw.get("cache_size", 8),
            max_retries=cfgs_mongo_raw.get("max_retries", 5),
            extra_camera_mappings=cfgs_mongo_raw.get("extra_camera_mappings", {}) or {},
            record_cache_dir=cfgs_mongo_raw.get("record_cache_dir"),
            record_cache_ttl=cfgs_mongo_raw.get("record_cache_ttl", 604800),
            blacklist_path=cfgs_mongo_raw.get("blacklist_path"),
        )

    data = DataConfig(
        datasets=cfgs_data.get("datasets", []),
        val_datasets=cfgs_data.get("val_datasets"),
        dataset_fpcs=cfgs_data.get("dataset_fpcs", []),
        batch_size=cfgs_data.get("batch_size", 4),
        tubelet_size=cfgs_data.get("tubelet_size", 2),
        use_tubelet_repeat=cfgs_data.get("use_tubelet_repeat", True),
        fps=cfgs_data.get("fps", 5),
        crop_size=normalize_image_size(cfgs_data.get("crop_size", 256)),
        patch_size=cfgs_data.get("patch_size", 16),
        num_target_frames=cfgs_data.get("num_target_frames", 16),
        pin_mem=cfgs_data.get("pin_mem", False),
        num_workers=cfgs_data.get("num_workers", 1),
        persistent_workers=cfgs_data.get("persistent_workers", True),
        camera_frame=cfgs_data.get("camera_frame", False),
        camera_views=cfgs_data.get("camera_views", ["left_mp4_path"]),
        stereo_view=cfgs_data.get("stereo_view", False),
        navsim=navsim,
        mongo_raw=mongo_raw,
    )

    # 解析 data_aug 配置
    cfgs_data_aug = args.get("data_aug", {})
    data_aug = DataAugConfig(
        horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]),
        motion_shift=cfgs_data_aug.get("motion_shift", False),
        reprob=cfgs_data_aug.get("reprob", 0.0),
        auto_augment=cfgs_data_aug.get("auto_augment", False),
    )

    # 解析 token_ae 配置
    cfgs_token_ae = args.get("token_ae", {})
    token_ae = TokenAEConfig(
        enabled=cfgs_token_ae.get("enabled", False),
        num_latent_tokens=cfgs_token_ae.get("num_latent_tokens", 64),
        num_heads=cfgs_token_ae.get("num_heads", 16),
        encoder_depth=cfgs_token_ae.get("encoder_depth", 4),
        decoder_depth=cfgs_token_ae.get("decoder_depth", 4),
        mlp_ratio=cfgs_token_ae.get("mlp_ratio", 4.0),
        dropout=cfgs_token_ae.get("dropout", 0.0),
        encoder_mode=cfgs_token_ae.get("encoder_mode", "parallel"),
        loss_type=cfgs_token_ae.get("loss_type", "smooth_l1"),
        cos_loss_weight=cfgs_token_ae.get("cos_loss_weight", 0.25),
        latent_reg_weight=cfgs_token_ae.get("latent_reg_weight", 0.0),
        pos_embed_type=cfgs_token_ae.get("pos_embed_type", "sincos"),
        input_grid_size=cfgs_token_ae.get("input_grid_size"),
        latent_grid_size=cfgs_token_ae.get("latent_grid_size"),
        temporal_depth=cfgs_token_ae.get("temporal_depth", 0),
        temporal_num_heads=cfgs_token_ae.get("temporal_num_heads"),
        temporal_mlp_ratio=cfgs_token_ae.get("temporal_mlp_ratio"),
        temporal_causal=cfgs_token_ae.get("temporal_causal", True),
        temporal_mode=cfgs_token_ae.get("temporal_mode", "index"),
        temporal_pos_embed_type=cfgs_token_ae.get("temporal_pos_embed_type", "none"),
        input_frame_mode=cfgs_token_ae.get("input_frame_mode", "all_frames"),
        temporal_loss_weight=cfgs_token_ae.get("temporal_loss_weight", 0.0),
    )

    # 解析 loss 配置
    cfgs_loss = args.get("loss", {})
    loss = LossConfig(
        loss_exp=cfgs_loss.get("loss_exp", 2.0),
        normalize_reps=cfgs_loss.get("normalize_reps", True),
        auto_steps=min(cfgs_loss.get("auto_steps", 1), data.max_num_frames),
    )

    # 解析 RL 配置
    cfgs_rl = args.get("rl", {})
    rl = RLConfig(
        enabled=cfgs_rl.get("enabled", False),
        algo=cfgs_rl.get("algo", "ppo"),
        status_mode=cfgs_rl.get("status_mode", "current_only"),
        hugsim_repo_root=cfgs_rl.get("hugsim_repo_root"),
        scenario_path=cfgs_rl.get("scenario_path"),
        scenario_manifest=cfgs_rl.get("scenario_manifest"),
        base_path=cfgs_rl.get("base_path"),
        camera_path=cfgs_rl.get("camera_path"),
        kinematic_path=cfgs_rl.get("kinematic_path"),
        camera_name=cfgs_rl.get("camera_name", "CAM_FRONT"),
        output_subdir=cfgs_rl.get("output_subdir", "hugsim_rl"),
        eval_checkpoint=cfgs_rl.get("eval_checkpoint"),
        rollout_steps=int(cfgs_rl.get("rollout_steps", 128)),
        max_episode_steps=int(cfgs_rl.get("max_episode_steps", 400)),
        ppo_epochs=int(cfgs_rl.get("ppo_epochs", 4)),
        mini_batch_size=int(cfgs_rl.get("mini_batch_size", 32)),
        gamma=float(cfgs_rl.get("gamma", 0.99)),
        gae_lambda=float(cfgs_rl.get("gae_lambda", 0.95)),
        clip_eps=float(cfgs_rl.get("clip_eps", 0.2)),
        value_clip_eps=float(cfgs_rl.get("value_clip_eps", 0.2)),
        vf_coef=float(cfgs_rl.get("vf_coef", 0.5)),
        ent_coef=float(cfgs_rl.get("ent_coef", 0.01)),
        lr=float(cfgs_rl.get("lr", 3e-4)),
        weight_decay=float(cfgs_rl.get("weight_decay", 0.01)),
        max_grad_norm=float(cfgs_rl.get("max_grad_norm", 1.0)),
        reward_scale=float(cfgs_rl.get("reward_scale", 1.0)),
        rl_loss_weight=float(cfgs_rl.get("rl_loss_weight", 1.0)),
        supervised_loss_weight=float(cfgs_rl.get("supervised_loss_weight", 0.0)),
        supervised_warmup_epochs=int(cfgs_rl.get("supervised_warmup_epochs", 0)),
        supervised_batches_per_epoch=int(cfgs_rl.get("supervised_batches_per_epoch", 0)),
        normalize_advantage=bool(cfgs_rl.get("normalize_advantage", True)),
        deterministic_eval=bool(cfgs_rl.get("deterministic_eval", True)),
        eval_episodes=int(cfgs_rl.get("eval_episodes", 1)),
        wheel_base=float(cfgs_rl.get("wheel_base", 2.7)),
        kinematic_dt=float(cfgs_rl.get("kinematic_dt", 0.25)),
    )

    # 解析 optimization 配置
    cfgs_opt = args.get("optimization", {})
    optimization = OptimizationConfig(
        ipe=cfgs_opt.get("ipe"),
        weight_decay=float(cfgs_opt.get("weight_decay", 0.04)),
        final_weight_decay=float(cfgs_opt.get("final_weight_decay", 0.4)),
        epochs=cfgs_opt.get("epochs", 100),
        anneal=cfgs_opt.get("anneal", 1),
        warmup=cfgs_opt.get("warmup", 10),
        start_lr=cfgs_opt.get("start_lr", 0.0001),
        lr=cfgs_opt.get("lr", 0.0001),
        final_lr=cfgs_opt.get("final_lr", 0.0),
        enc_lr_scale=cfgs_opt.get("enc_lr_scale", 1.0),
        betas=cfgs_opt.get("betas", (0.9, 0.999)),
        eps=cfgs_opt.get("eps", 1e-8),
        is_anneal=cfgs_opt.get("is_anneal", False),
        anneal_ckpt=cfgs_opt.get("anneal_ckpt", None),
        resume_anneal=cfgs_opt.get("resume_anneal", False),
        ipe_scale=cfgs_opt.get("ipe_scale", 1.0),
    )

    # 解析 lewm 配置
    lewm = LeWMConfig(
        enabled=bool(cfgs_lewm.get("enabled", False)),
        sigreg_weight=float(cfgs_lewm.get("sigreg_weight", 0.09)),
        sigreg_knots=int(cfgs_lewm.get("sigreg_knots", 17)),
        sigreg_num_proj=int(cfgs_lewm.get("sigreg_num_proj", 1024)),
        projector_hidden_dim=int(cfgs_lewm.get("projector_hidden_dim", 2048)),
        embed_dim=int(cfgs_lewm.get("embed_dim", 192)),
    )

    # 解析 Stage-2 / Stage-3 配置
    cfgs_stage2 = args.get("stage2", {})
    cfgs_stage2_lambdas = cfgs_stage2.get("lambdas", {}) or {}
    stage2 = Stage2Config(
        num_modes=int(cfgs_stage2.get("num_modes", planner.num_modes)),
        inference_num_rounds=int(cfgs_stage2.get("inference_num_rounds", 1)),
        predictor_rollout_seconds=(
            None
            if cfgs_stage2.get("predictor_rollout_seconds") is None
            else float(cfgs_stage2.get("predictor_rollout_seconds"))
        ),
        detach_zfut=bool(cfgs_stage2.get("detach_zfut", True)),
        refine_stop_grad_to_proposer=bool(cfgs_stage2.get("refine_stop_grad_to_proposer", True)),
        refine_use_random_predictor_latent=bool(cfgs_stage2.get("refine_use_random_predictor_latent", False)),
        refine_keep_initial_actions=bool(cfgs_stage2.get("refine_keep_initial_actions", False)),
        predictor_finetune=bool(cfgs_stage2.get("predictor_finetune", False)),
        checkpoint=cfgs_stage2.get("checkpoint"),
        lambdas=StageLossWeightsConfig(
            prop=float(cfgs_stage2_lambdas.get("prop", 1.0)),
            refine=float(cfgs_stage2_lambdas.get("refine", 1.0)),
            anchor=float(cfgs_stage2_lambdas.get("anchor", 0.0)),
            div=float(cfgs_stage2_lambdas.get("div", 0.0)),
        ),
    )

    cfgs_stage3 = args.get("stage3", {})
    stage3_num_rounds = int(cfgs_stage3.get("num_rounds", 2))
    stage3 = Stage3Config(
        num_rounds=stage3_num_rounds,
        round_weights=list(cfgs_stage3.get("round_weights", [1.0] * stage3_num_rounds)),
        predictor_rollout_seconds=(
            None
            if cfgs_stage3.get("predictor_rollout_seconds") is None
            else float(cfgs_stage3.get("predictor_rollout_seconds"))
        ),
        grad_checkpoint=bool(cfgs_stage3.get("grad_checkpoint", True)),
        predictor_finetune=bool(cfgs_stage3.get("predictor_finetune", False)),
        checkpoint=cfgs_stage3.get("checkpoint"),
        fut_consistency_weight=float(cfgs_stage3.get("fut_consistency_weight", 0.0)),
        refine_use_z_context=bool(cfgs_stage3.get("refine_use_z_context", True)),
        refine_use_status_feature=bool(cfgs_stage3.get("refine_use_status_feature", True)),
        refine_use_proposal_traj=bool(cfgs_stage3.get("refine_use_proposal_traj", True)),
        refine_use_proposal_logits=bool(cfgs_stage3.get("refine_use_proposal_logits", True)),
        refine_use_proposal_features=bool(cfgs_stage3.get("refine_use_proposal_features", True)),
        refine_use_predictor_rollout=bool(cfgs_stage3.get("refine_use_predictor_rollout", True)),
        refine_use_random_predictor_latent=bool(cfgs_stage3.get("refine_use_random_predictor_latent", False)),
        refine_keep_initial_actions=bool(cfgs_stage3.get("refine_keep_initial_actions", False)),
        use_multimodal_final=bool(cfgs_stage3.get("use_multimodal_final", False)),
    )

    return TrainingConfig(
        meta=meta,
        model=model,
        train=train,
        ema=ema,
        segmentation=segmentation,
        planner=planner,
        proposal=proposal,
        data=data,
        data_aug=data_aug,
        token_ae=token_ae,
        loss=loss,
        lewm=lewm,
        stage2=stage2,
        stage3=stage3,
        rl=rl,
        optimization=optimization,
    )
