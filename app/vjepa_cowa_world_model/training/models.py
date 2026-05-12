# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
模型初始化模块

提供各种模型的初始化函数。
"""

import copy
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa.utils import init_video_model as init_video_model_vjepa
from app.vjepa_cowa_world_model.utils import get_status_dim, resolve_planner_use_drive_command
from src.utils.logging import get_logger

from .config import (
    TrainingConfig,
    is_drive_jepa_main_encoder_config,
    resolve_effective_tokens_per_frame,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_predictor_img_size,
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_main_encoder_tokens_per_frame,
    resolve_predictor_runtime_normalize_reps,
    resolve_proposal_encoder_backbone,
)

logger = get_logger(__name__)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _init_encoder_vjepa2_1(
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """
    初始化 V-JEPA 2.1 encoder (inference mode only)

    使用 app/vjepa_2_1 的 VisionTransformer 和 MultiSeqWrapper。
    当 training=False 时，输出形状为 [B, N, embed_dim]，与 V-JEPA 2 一致。

    Args:
        config: 训练配置
        device: 设备

    Returns:
        nn.Module: encoder (MultiSeqWrapper wrapped)
    """
    import app.vjepa_2_1.models.vision_transformer as vjepa21_vit
    from app.vjepa_2_1.wrappers import MultiSeqWrapper as MultiSeqWrapper21

    encoder = vjepa21_vit.__dict__[config.model.model_name](
        img_size=config.data.crop_size,
        patch_size=config.data.patch_size,
        num_frames=512,
        tubelet_size=config.data.tubelet_size,
        uniform_power=config.model.uniform_power,
        use_sdpa=config.meta.use_sdpa,
        use_silu=config.model.use_silu,
        wide_silu=config.model.wide_silu,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_rope=config.model.use_rope,
        # V-JEPA 2.1 专用参数
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder = MultiSeqWrapper21(encoder)
    encoder.to(device)

    return encoder


def init_encoder(
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """
    初始化 encoder 和 target_encoder

    根据 config.model.backbone 选择 V-JEPA 2 或 V-JEPA 2.1 编码器。

    Args:
        config: 训练配置
        device: 设备

    Returns:
        Tuple[nn.Module, nn.Module]: (encoder, target_encoder)
    """
    encoder = init_context_encoder(config, device)

    target_encoder = copy.deepcopy(encoder)
    logger.info("end init encoder")

    # 打印参数量（仅主进程）
    encoder_params = sum(p.numel() for p in encoder.parameters())
    target_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    if _is_main_process():
        logger.info(f"init encoder_params: {encoder_params / 1e6:>8.2f}M")
        logger.info(f"init target_encoder_params: {target_encoder_params / 1e6:>8.2f}M")

    return encoder, target_encoder


def init_context_encoder(config: TrainingConfig, device: torch.device) -> nn.Module:
    """初始化单个上下文 encoder，不额外创建 target_encoder。"""
    backbone = config.model.backbone
    logger.info(f"begin init context encoder (backbone={backbone}):")

    if backbone == "drive_jepa_img_encoder":
        from app.vjepa_cowa_world_model.models.drive_jepa_img_encoder import DriveJEPAImgEncoderAdapter

        checkpoint_path = config.meta.pretrain_checkpoint_full if config.meta.load_encoder else None
        encoder = DriveJEPAImgEncoderAdapter(
            checkpoint_path=checkpoint_path,
            resolution=config.model.drive_jepa_resolution,
            num_frames=config.model.drive_jepa_num_frames,
            max_num_observed_frames=config.data.num_target_frames,
            checkpoint_key=config.model.drive_jepa_checkpoint_key,
            model_name=config.model.model_name,
            patch_size=config.data.patch_size,
            tubelet_size=config.data.tubelet_size,
            uniform_power=config.model.uniform_power,
            use_rope=config.model.use_rope,
            use_sdpa=config.meta.use_sdpa,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
            use_grid_mask=config.model.drive_jepa_use_grid_mask,
            use_causal_attention=getattr(config.model, "drive_jepa_use_causal_attention", True),
        ).to(device)
    elif backbone == "vjepa2.1":
        encoder = _init_encoder_vjepa2_1(config, device)
    elif backbone == "vjepa2":
        encoder, _ = init_video_model_vjepa(
            uniform_power=config.model.uniform_power,
            use_mask_tokens=config.model.use_mask_tokens,
            num_mask_tokens=10,
            zero_init_mask_tokens=config.model.zero_init_mask_tokens,
            device=device,
            patch_size=config.data.patch_size,
            max_num_frames=512,
            tubelet_size=config.data.tubelet_size,
            model_name=config.model.model_name,
            crop_size=config.data.crop_size,
            pred_depth=config.model.pred_depth,
            pred_num_heads=config.model.pred_num_heads,
            pred_embed_dim=config.model.pred_embed_dim,
            use_sdpa=config.meta.use_sdpa,
            use_silu=config.model.use_silu,
            use_pred_silu=config.model.use_pred_silu,
            wide_silu=config.model.wide_silu,
            use_rope=config.model.use_rope,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Must be 'vjepa2', 'vjepa2.1', or 'drive_jepa_img_encoder'.")

    encoder_params = sum(p.numel() for p in encoder.parameters())
    if _is_main_process():
        logger.info(f"init context_encoder_params: {encoder_params / 1e6:>8.2f}M")
    return encoder


def init_proposal_encoder(config: TrainingConfig, device: torch.device) -> nn.Module:
    """初始化独立 proposal encoder，允许与主 encoder 使用不同 backbone。"""
    backbone = resolve_proposal_encoder_backbone(config)
    logger.info(f"begin init proposal encoder (backbone={backbone}):")

    if backbone == "drive_jepa_img_encoder":
        from app.vjepa_cowa_world_model.models.drive_jepa_img_encoder import DriveJEPAImgEncoderAdapter

        checkpoint_key = config.proposal.drive_jepa_checkpoint_key or config.proposal.encoder_checkpoint_key
        encoder = DriveJEPAImgEncoderAdapter(
            checkpoint_path=None,
            resolution=config.proposal.drive_jepa_resolution,
            num_frames=config.proposal.drive_jepa_num_frames,
            max_num_observed_frames=config.train.num_observed_frames,
            checkpoint_key=checkpoint_key,
            model_name=config.proposal.encoder_model_name or config.model.model_name,
            patch_size=config.data.patch_size,
            tubelet_size=config.data.tubelet_size,
            uniform_power=config.model.uniform_power,
            use_rope=config.model.use_rope,
            use_sdpa=config.meta.use_sdpa,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
            use_grid_mask=config.proposal.drive_jepa_use_grid_mask,
            use_causal_attention=getattr(config.proposal, "drive_jepa_use_causal_attention", True),
        ).to(device)
    elif backbone == config.model.backbone:
        encoder = init_context_encoder(config, device)
    else:
        raise ValueError(
            "Unsupported proposal.encoder_backbone=%r with model.backbone=%r. "
            "Currently heterogeneous proposal encoders support only 'drive_jepa_img_encoder'."
            % (backbone, config.model.backbone)
        )

    encoder_params = sum(p.numel() for p in encoder.parameters())
    if _is_main_process():
        logger.info(f"init proposal_encoder_params: {encoder_params / 1e6:>8.2f}M")
    return encoder


def init_predictor(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    predictor_img_size_override=None,
) -> nn.Module:
    """
    初始化 action-conditioned predictor

    Args:
        config: 训练配置
        device: 设备
        encoder_embed_dim: encoder 的嵌入维度

    Returns:
        nn.Module: predictor 模型
    """
    logger.info("begin init action-conditioned predictor:")

    # use_drive_command=False 时：去掉 4 维 cmd，command_dim 强制 0
    _use_cmd = getattr(config.train, "use_drive_command", True)
    _state_dim = config.train.state_dim
    _command_dim = config.train.command_dim
    if not _use_cmd:
        _state_dim = _state_dim - 4
        _command_dim = 0
        logger.info(
            f"use_drive_command=False: predictor state_dim {_state_dim + 4} -> {_state_dim}, command_dim forced to 0"
        )

    # 断言：predictor command_dim > 0 要求 split_status_embedding 开启
    if (
        _command_dim > 0
        and not config.planner.split_status_embedding
        and not config.train.predictor_inference_consistent
    ):
        raise ValueError(
            f"predictor command_dim={_command_dim} > 0 but planner.split_status_embedding=False. "
            "Enable split_status_embedding or set command_dim=0."
        )

    from app.vjepa_droid.utils import init_predictor_model

    # 计算 state_embed_dim: 当 state_dim != action_dim 时使用独立维度
    _state_embed_dim = None
    if _state_dim != config.train.action_dim:
        _state_embed_dim = _state_dim

    predictor_img_size = (
        predictor_img_size_override if predictor_img_size_override is not None else config.data.crop_size
    )

    predictor = init_predictor_model(
        uniform_power=config.model.uniform_power,
        device=device,
        patch_size=config.data.patch_size,
        max_num_frames=512,
        tubelet_size=config.data.tubelet_size,
        model_name=config.model.model_name,
        crop_size=predictor_img_size,
        pred_depth=config.model.pred_depth,
        pred_num_heads=config.model.pred_num_heads,
        pred_embed_dim=config.model.pred_embed_dim,
        embed_dim=encoder_embed_dim,
        action_embed_dim=config.train.action_dim,
        state_embed_dim=_state_embed_dim,
        command_dim=_command_dim,
        pred_is_frame_causal=config.model.pred_is_frame_causal,
        use_extrinsics=config.model.use_extrinsics,
        use_sdpa=config.meta.use_sdpa,
        use_silu=config.model.use_silu,
        use_pred_silu=config.model.use_pred_silu,
        wide_silu=config.model.wide_silu,
        use_rope=config.model.use_rope,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_perceiver_ema=config.train.perceiver_ema,
        target_shape=None,
    )

    logger.info(
        f"end init predictor (action_embed_dim={config.train.action_dim}, "
        f"state_embed_dim={_state_embed_dim}, command_dim={_command_dim}, "
        f"use_drive_command={_use_cmd}, predictor_img_size={predictor_img_size})"
    )

    # 打印参数量（仅主进程）
    predictor_params = sum(p.numel() for p in predictor.parameters())
    if _is_main_process():
        logger.info(f"init predictor_params: {predictor_params / 1e6:>8.2f}M")

    return predictor


def init_predictor_for_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    num_latent_tokens: int,
    embed_dim_override: Optional[int] = None,
    latent_grid_size: Optional[Tuple[int, int]] = None,
) -> nn.Module:
    """
    初始化适配 Token AE 压缩 token 的 predictor。

    Token AE 将 encoder 的 256 tokens/frame 压缩到 num_latent_tokens (如 64)。
    原始 predictor 假设 tokens_per_frame = (crop_size / patch_size)²，
    内部用 grid_height × grid_width 做帧级 reshape 和 frame-causal attention mask。

    适配方法：用 virtual crop_size 使 (crop_height / patch_size) × (crop_width / patch_size)
    = num_latent_tokens，例如 num_latent_tokens=32 且 latent_grid_size=(4, 8)
    → virtual crop_size = (64, 128)。

    注意：
    - predictor 的所有 Linear 权重与 grid 尺寸无关，预训练权重可直接加载
    - 仅 attention mask 和 RoPE 根据 grid 尺寸动态计算（非学习参数）
    - 由于空间假设变了，predictor 需要 fine-tune (建议 predictor_train=True)

    Args:
        config: 训练配置
        device: 设备
        encoder_embed_dim: encoder 嵌入维度 (如 1408)
        num_latent_tokens: AE 压缩后每帧 token 数 (如 64)
        embed_dim_override: predictor I/O 维度覆盖值；默认沿用 encoder_embed_dim
        latent_grid_size: AE latent token 的空间网格 (H, W)；为空时兼容旧的方形推导

    Returns:
        nn.Module: 适配后的 predictor
    """
    patch_size = config.data.patch_size  # typically 16
    if latent_grid_size is None and getattr(config, "token_ae", None) is not None:
        latent_grid_size = config.token_ae.latent_grid_size
    if latent_grid_size is None:
        import math

        virtual_grid = int(math.sqrt(num_latent_tokens))
        if virtual_grid * virtual_grid != num_latent_tokens:
            raise ValueError(
                f"num_latent_tokens={num_latent_tokens} 必须是完全平方数，或显式配置 token_ae.latent_grid_size；"
                f"sqrt={math.sqrt(num_latent_tokens):.2f}"
            )
        virtual_grid_height = virtual_grid
        virtual_grid_width = virtual_grid
    else:
        if not isinstance(latent_grid_size, (list, tuple)) or len(latent_grid_size) != 2:
            raise ValueError(f"token_ae.latent_grid_size must be a 2-element sequence, got {latent_grid_size!r}")
        virtual_grid_height = int(latent_grid_size[0])
        virtual_grid_width = int(latent_grid_size[1])
        if virtual_grid_height <= 0 or virtual_grid_width <= 0:
            raise ValueError(f"token_ae.latent_grid_size must be positive, got {latent_grid_size!r}")
        if virtual_grid_height * virtual_grid_width != int(num_latent_tokens):
            raise ValueError(
                "token_ae.latent_grid_size does not match num_latent_tokens: "
                f"latent_grid_size={latent_grid_size!r}, num_latent_tokens={num_latent_tokens}"
            )
    virtual_crop_size = (
        virtual_grid_height * patch_size
        if virtual_grid_height == virtual_grid_width
        else (virtual_grid_height * patch_size, virtual_grid_width * patch_size)
    )

    logger.info(
        "init predictor for Token AE: num_latent_tokens=%d → virtual_grid=%d×%d → "
        "virtual_crop_size=%s (original crop_size=%s)",
        num_latent_tokens,
        virtual_grid_height,
        virtual_grid_width,
        virtual_crop_size,
        config.data.crop_size,
    )

    # use_drive_command=False 时：去掉 4 维 cmd，command_dim 强制 0
    _use_cmd = getattr(config.train, "use_drive_command", True)
    _state_dim = config.train.state_dim
    _command_dim = config.train.command_dim
    if not _use_cmd:
        _state_dim = _state_dim - 4
        _command_dim = 0
        logger.info(
            f"use_drive_command=False: predictor state_dim {_state_dim + 4} -> {_state_dim}, command_dim forced to 0"
        )

    # 计算 state_embed_dim: 当 state_dim != action_dim 时使用独立维度
    _state_embed_dim = None
    if _state_dim != config.train.action_dim:
        _state_embed_dim = _state_dim

    predictor_io_dim = int(embed_dim_override) if embed_dim_override is not None else int(encoder_embed_dim)

    from app.vjepa_droid.utils import init_predictor_model

    predictor = init_predictor_model(
        uniform_power=config.model.uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=config.data.tubelet_size,
        model_name=config.model.model_name,
        crop_size=virtual_crop_size,  # <-- 关键：用虚拟 crop_size 使 grid 匹配 AE token 数
        pred_depth=config.model.pred_depth,
        pred_num_heads=config.model.pred_num_heads,
        pred_embed_dim=config.model.pred_embed_dim,
        embed_dim=predictor_io_dim,
        action_embed_dim=config.train.action_dim,
        state_embed_dim=_state_embed_dim,
        command_dim=_command_dim,
        pred_is_frame_causal=config.model.pred_is_frame_causal,
        use_extrinsics=config.model.use_extrinsics,
        use_sdpa=config.meta.use_sdpa,
        use_silu=config.model.use_silu,
        use_pred_silu=config.model.use_pred_silu,
        wide_silu=config.model.wide_silu,
        use_rope=config.model.use_rope,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_perceiver_ema=config.train.perceiver_ema,
        target_shape=None,
    )

    predictor_params = sum(p.numel() for p in predictor.parameters())
    logger.info(
        "predictor for AE: params=%.2fM, virtual_grid=%d×%d, tokens_per_frame=%d, io_dim=%d, "
        "action_embed_dim=%d, state_embed_dim=%s, command_dim=%d, use_drive_command=%s",
        predictor_params / 1e6,
        virtual_grid_height,
        virtual_grid_width,
        num_latent_tokens,
        predictor_io_dim,
        config.train.action_dim,
        _state_embed_dim,
        _command_dim,
        _use_cmd,
    )

    # Enable gradient checkpointing to reduce memory usage during training
    # This is especially beneficial with LoRA adapters: reduces activation memory ~30-40%
    if hasattr(predictor, "gradient_checkpointing_enable"):
        try:
            predictor.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for predictor (memory ~30-40% reduction)")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing for predictor: %s", e)

    return predictor


def load_frozen_token_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    tokens_per_frame: int,
    normalize_reps: bool,
    dtype: Optional[torch.dtype] = None,
):
    """Load a frozen TokenAE from checkpoint for training or validation runtime use."""
    if not getattr(config, "token_ae", None) or not config.token_ae.enabled:
        return None, normalize_reps

    ae_checkpoint = getattr(config.meta, "ae_checkpoint", None)
    if not ae_checkpoint:
        raise ValueError("meta.ae_checkpoint must be provided when token_ae.enabled=True")

    from app.vjepa_cowa_world_model.models.token_ae import TokenAE

    token_ae_ckpt = torch.load(ae_checkpoint, map_location="cpu")
    token_ae_cfg = token_ae_ckpt.get("config", {})
    token_ae_embed_dim = int(token_ae_cfg.get("embed_dim", encoder_embed_dim))
    token_ae_tokens_per_frame = int(token_ae_cfg.get("tokens_per_frame", tokens_per_frame))
    token_ae_normalize_reps = bool(token_ae_cfg.get("normalize_reps", normalize_reps))

    if token_ae_embed_dim != encoder_embed_dim:
        raise ValueError(
            f"Token AE embed_dim={token_ae_embed_dim} does not match encoder embed_dim={encoder_embed_dim}"
        )
    if token_ae_tokens_per_frame != tokens_per_frame:
        raise ValueError(
            "Token AE tokens_per_frame="
            f"{token_ae_tokens_per_frame} does not match current data setting {tokens_per_frame}"
        )

    configured_num_latent_tokens = int(config.token_ae.num_latent_tokens)
    checkpoint_num_latent_tokens = int(token_ae_cfg.get("num_latent_tokens", configured_num_latent_tokens))
    if checkpoint_num_latent_tokens != configured_num_latent_tokens:
        raise ValueError(
            "Token AE num_latent_tokens mismatch: "
            f"checkpoint={checkpoint_num_latent_tokens}, config={configured_num_latent_tokens}"
        )

    token_ae = TokenAE(
        embed_dim=token_ae_embed_dim,
        tokens_per_frame=token_ae_tokens_per_frame,
        num_latent_tokens=checkpoint_num_latent_tokens,
        num_heads=int(token_ae_cfg.get("num_heads", config.token_ae.num_heads)),
        encoder_depth=int(token_ae_cfg.get("encoder_depth", config.token_ae.encoder_depth)),
        decoder_depth=int(token_ae_cfg.get("decoder_depth", config.token_ae.decoder_depth)),
        mlp_ratio=float(token_ae_cfg.get("mlp_ratio", config.token_ae.mlp_ratio)),
        dropout=float(token_ae_cfg.get("dropout", config.token_ae.dropout)),
        encoder_mode=token_ae_cfg.get("encoder_mode", config.token_ae.encoder_mode),
        loss_type=token_ae_cfg.get("loss_type", config.token_ae.loss_type),
        cos_loss_weight=float(token_ae_cfg.get("cos_loss_weight", config.token_ae.cos_loss_weight)),
        latent_reg_weight=float(token_ae_cfg.get("latent_reg_weight", config.token_ae.latent_reg_weight)),
        pos_embed_type=token_ae_cfg.get("pos_embed_type", config.token_ae.pos_embed_type),
        input_grid_size=token_ae_cfg.get("input_grid_size", config.token_ae.input_grid_size),
        latent_grid_size=token_ae_cfg.get("latent_grid_size", config.token_ae.latent_grid_size),
        temporal_depth=int(token_ae_cfg.get("temporal_depth", config.token_ae.temporal_depth)),
        temporal_num_heads=token_ae_cfg.get("temporal_num_heads", config.token_ae.temporal_num_heads),
        temporal_mlp_ratio=token_ae_cfg.get("temporal_mlp_ratio", config.token_ae.temporal_mlp_ratio),
        temporal_causal=bool(token_ae_cfg.get("temporal_causal", config.token_ae.temporal_causal)),
        temporal_mode=token_ae_cfg.get("temporal_mode", config.token_ae.temporal_mode),
        temporal_pos_embed_type=token_ae_cfg.get("temporal_pos_embed_type", config.token_ae.temporal_pos_embed_type),
        temporal_loss_weight=float(token_ae_cfg.get("temporal_loss_weight", config.token_ae.temporal_loss_weight)),
    )
    if dtype is None:
        token_ae = token_ae.to(device=device)
    else:
        token_ae = token_ae.to(device=device, dtype=dtype)

    token_ae_state = token_ae_ckpt["token_ae"] if "token_ae" in token_ae_ckpt else token_ae_ckpt
    token_ae.load_state_dict(token_ae_state, strict=True)

    for parameter in token_ae.parameters():
        parameter.requires_grad = False
    token_ae.eval()

    return token_ae, token_ae_normalize_reps


def prepare_runtime_tokens(
    tokens: torch.Tensor,
    num_frames: int,
    normalize_reps: bool,
    token_ae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Compress per-frame tokens with TokenAE if present, then apply runtime normalization."""
    input_dtype = tokens.dtype
    if token_ae is not None:
        ae_tokens_per_frame = int(getattr(token_ae, "tokens_per_frame"))
        expected_tokens = int(num_frames) * ae_tokens_per_frame
        if tokens.size(1) != expected_tokens:
            if tokens.size(1) % ae_tokens_per_frame != 0:
                raise ValueError(
                    "Cannot infer TokenAE frame count: "
                    f"tokens={tokens.size(1)}, num_frames={num_frames}, "
                    f"ae_tokens_per_frame={ae_tokens_per_frame}"
                )
            num_frames = tokens.size(1) // ae_tokens_per_frame
        token_ae_parameters = getattr(token_ae, "parameters", None)
        ae_param = next(token_ae_parameters(), None) if callable(token_ae_parameters) else None
        if ae_param is not None and tokens.is_floating_point() and tokens.dtype != ae_param.dtype:
            tokens = tokens.to(dtype=ae_param.dtype)
        tokens = token_ae.encode(tokens, num_frames=num_frames)
        if tokens.is_floating_point() and tokens.dtype != input_dtype:
            tokens = tokens.to(dtype=input_dtype)
    if normalize_reps:
        tokens = F.layer_norm(tokens, (tokens.size(-1),))
    return tokens


def register_predictor_future_query_tokens(
    predictor: nn.Module,
    embed_dim: int,
    future_tubelets: int,
    tokens_per_frame: int,
    device: torch.device,
    init_std: float = 0.02,
) -> None:
    """Register learnable predictor query tokens for future tubelets."""
    if future_tubelets <= 0:
        return

    predictor_core = predictor.module if hasattr(predictor, "module") else predictor
    num_future_tokens = future_tubelets * tokens_per_frame
    expected_shape = (1, num_future_tokens, embed_dim)
    existing = getattr(predictor_core, "future_query_tokens", None)
    if existing is not None:
        if tuple(existing.shape) != expected_shape:
            raise ValueError(
                "Existing predictor.future_query_tokens shape mismatch: "
                f"got {tuple(existing.shape)}, expected {expected_shape}"
            )
        return

    future_query_tokens = nn.Parameter(torch.empty(expected_shape, device=device))
    nn.init.trunc_normal_(future_query_tokens, std=init_std)
    predictor_core.register_parameter("future_query_tokens", future_query_tokens)
    logger.info(
        "Registered predictor future query tokens: future_tubelets=%d, tokens=%d, embed_dim=%d",
        future_tubelets,
        num_future_tokens,
        embed_dim,
    )


def build_predictor_input_with_future_queries(predictor: nn.Module, observed_tokens: torch.Tensor) -> torch.Tensor:
    """Append learnable future query tokens after observed predictor input tokens."""
    predictor_core = predictor.module if hasattr(predictor, "module") else predictor
    future_query_tokens = getattr(predictor_core, "future_query_tokens", None)
    if future_query_tokens is None:
        return observed_tokens
    return torch.cat([observed_tokens, future_query_tokens.expand(observed_tokens.size(0), -1, -1)], dim=1)


def init_predictor_runtime_with_token_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    raw_tokens_per_frame_override: Optional[int] = None,
    predictor_img_size_override=None,
):
    """Initialize predictor plus optional frozen TokenAE runtime state for training scripts."""
    raw_tokens_per_frame = (
        int(raw_tokens_per_frame_override)
        if raw_tokens_per_frame_override is not None
        else config.data.tokens_per_frame
    )
    token_ae_enabled = bool(getattr(config, "token_ae", None) and config.token_ae.enabled)
    if token_ae_enabled:
        effective_tokens_per_frame = resolve_effective_tokens_per_frame(config)
    else:
        effective_tokens_per_frame = (
            int(raw_tokens_per_frame_override)
            if raw_tokens_per_frame_override is not None
            else resolve_effective_tokens_per_frame(config)
        )
    runtime_normalize_reps = resolve_predictor_runtime_normalize_reps(config)
    token_ae = None

    if token_ae_enabled:
        token_ae, runtime_normalize_reps = load_frozen_token_ae(
            config,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            tokens_per_frame=raw_tokens_per_frame,
            normalize_reps=runtime_normalize_reps,
            dtype=config.dtype,
        )
        predictor = init_predictor_for_ae(
            config,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            num_latent_tokens=effective_tokens_per_frame,
            latent_grid_size=getattr(token_ae, "latent_grid_size", config.token_ae.latent_grid_size),
        )
    else:
        predictor = init_predictor(
            config,
            device,
            encoder_embed_dim,
            predictor_img_size_override=predictor_img_size_override,
        )

    return predictor, token_ae, effective_tokens_per_frame, runtime_normalize_reps


def resolve_main_predictor_runtime_overrides(config: TrainingConfig, encoder: Optional[nn.Module] = None):
    """Return tokens/grid overrides needed by non-default main encoders."""
    if not is_drive_jepa_main_encoder_config(config):
        return None, None
    return resolve_main_encoder_raw_tokens_per_frame(config, encoder), resolve_main_encoder_predictor_img_size(
        config, encoder
    )


def is_drive_jepa_encoder(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    core = module.module if hasattr(module, "module") else module
    return bool(getattr(core, "is_drive_jepa_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "DriveJEPAImgEncoderAdapter"
    )


def configure_drive_jepa_encoder_trainability(
    encoder: Optional[nn.Module],
    config: TrainingConfig,
    trainable: Optional[bool] = None,
) -> None:
    """Apply the main Drive-JEPA freeze/eval policy after init or DDP wrapping."""
    if not is_drive_jepa_encoder(encoder):
        return
    core = encoder.module if hasattr(encoder, "module") else encoder
    should_train = bool(config.train.encoder_train if trainable is None else trainable)
    if should_train:
        core.train()
        for parameter in core.parameters():
            parameter.requires_grad = True
        return
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad = False


def should_save_main_encoder(config: TrainingConfig) -> bool:
    """Return whether checkpoints should include the main encoder state."""
    return bool(config.train.encoder_train or is_drive_jepa_main_encoder_config(config))


def init_segmentation_modules(
    device: torch.device, use_segmentation: bool = True
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    初始化 seg_neck 和 seg_head

    Args:
        device: 设备
        use_segmentation: 是否使用分割模块

    Returns:
        Tuple[Optional[nn.Module], Optional[nn.Module]]: (seg_neck, seg_head)
    """
    if not use_segmentation:
        return None, None

    # Lazy import to avoid requiring torchcv when segmentation is disabled
    from app.vjepa_cowa.co_detr_decoder import CoDetrDecoder
    from app.vjepa_cowa.seg_head2 import SimpleSemanticSegHead
    from app.vjepa_cowa.seg_neck2 import SFP

    seg_neck = SFP(input_channels=[1408], out_channels=256, use_p2=True, use_act_checkpoint=False)

    seg_head = SimpleSemanticSegHead(
        input_strides=[4, 8, 16, 32, 64],
        num_classes=2,  # 占位
        decoder=CoDetrDecoder(
            num_proposals=1500,
            embed_dims=256,
            num_heads=8,
            num_levels=5,
            dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            two_stage=False,
            num_co_heads=0,
            with_coord_feat=False,
            with_pos_coord=False,
        ),
        embed_dims=256,
        out_mask_dim=256,  # 占位
        loss_weights={
            "loss_seg": 2.0,
            "loss_dice": 5.0,
        },
        subcat_num=0,
    )

    seg_neck = seg_neck.to(device)
    seg_head = seg_head.to(device)

    return seg_neck, seg_head


def init_planner(
    config: TrainingConfig,
    encoder_dim: int,
    device: torch.device,
    num_poses: Optional[int] = None,
    tokens_per_frame_override: Optional[int] = None,
) -> Optional[nn.Module]:
    """
    初始化 planner

    Args:
        config: 训练配置
        encoder_dim: encoder 的嵌入维度
        device: 设备
        num_poses: 可选，自定义 num_poses 值。如果为 None，则根据模式自动计算：
                   - predictor_inference_consistent=True: total_frames - num_encoder_frames
                   - 否则: total_frames - 1
        tokens_per_frame_override: 可选，覆盖 planner 使用的每帧 token 数

    Returns:
        Optional[nn.Module]: planner 模型，如果未启用则返回 None
    """
    if not config.planner.use_planner:
        logger.info("use_planner=False, planner is disabled")
        return None

    # 计算 num_poses
    # total_frames = config.data.num_target_frames // config.data.tubelet_size
    total_frames = (
        config.data.num_target_frames
    )  # should be num_target_frames, not divided by tubelet_size, because planner operates at frame level now
    if num_poses is None:
        # 默认计算逻辑
        if config.train.predictor_inference_consistent:
            num_poses = total_frames - config.train.num_encoder_frames
        else:
            num_poses = total_frames - 1

    # Planner trajectory heads still predict raw future poses (num_poses), but the temporal
    # memory may be produced at a coarser main-encoder step.  Drive-JEPA main encodes one
    # non-overlapping 2-frame chunk as one predictor/planner token step.
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_poses)
    num_context_frames = max(1, int(config.planner.num_context_frames))
    tokens_per_frame = (
        int(tokens_per_frame_override) if tokens_per_frame_override is not None else config.data.tokens_per_frame
    )
    enable_rl_actor_critic = config.planner.enable_rl_actor_critic or config.rl.enabled
    use_action_history = bool(getattr(config.planner, "use_action_history_for_planner", False))
    action_history_dim = int(getattr(config.planner, "action_history_dim", 3))
    if is_drive_jepa_main_encoder_config(config):
        num_observed_frames = resolve_main_encoder_num_observed_steps(config)
    else:
        num_observed_frames = int(
            getattr(config.train, "num_encoder_frames", getattr(config.train, "num_observed_frames", 1))
        )

    # 根据 predictor_inference_consistent 和 use_states_for_planner 决定 status_dim
    # IC 格式: 7(drive_command_7) 或 8(drive_command_8)
    # 根据 predictor_inference_consistent、use_drive_command_for_predictor 和 use_states_for_planner 决定 status_dim
    _use_cmd = resolve_planner_use_drive_command(config)
    if config.rl.enabled:
        planner_status_dim = get_status_dim(
            config.rl.status_mode,
            num_context_frames=num_context_frames,
        )
    elif config.train.predictor_inference_consistent:
        # planner.status_dim > 0 时优先使用（解耦 predictor.state_dim 与 planner.status_dim）
        planner_status_dim = config.planner.status_dim if config.planner.status_dim > 0 else config.train.state_dim
    elif config.planner.use_states_for_planner:
        planner_status_dim = 7  # 原始 states 维度
    else:
        planner_status_dim = 8  # 提取的特征维度

    # use_drive_command=False 时：planner 维度减 4，command_dim 强制 0
    if not _use_cmd:
        planner_status_dim = planner_status_dim - 4
        logger.info(f"use_drive_command=False: planner status_dim -> {planner_status_dim}, command_dim forced to 0")

    # 防止未知维度静默走到 Linear 层产生难以定位的 shape error
    _valid_dims = (3, 4, 7, 8, 12) if not _use_cmd else (7, 8, 12)
    assert planner_status_dim in _valid_dims, (
        f"Unsupported planner_status_dim={planner_status_dim}; expected one of {_valid_dims}. "
        f"Check train.state_dim and planner.status_dim in your config."
    )

    # 计算 command_dim：IC 模式下 status = [cmd(4) | kinematics(N)]，拆分嵌入
    if _use_cmd and config.planner.split_status_embedding and config.train.predictor_inference_consistent:
        planner_command_dim = 4  # drive_command one-hot 维度
    else:
        planner_command_dim = 0  # 旧行为，不拆分（或 use_drive_command=False）

    diff_adaln_version = getattr(config.planner, "diff_adaln_version", "legacy")
    diff_init_traj_strategy = str(getattr(config.planner, "diff_init_traj_strategy", "gaussian")).lower()
    diff_init_traj_noise_scale = float(getattr(config.planner, "diff_init_traj_noise_scale", 1.0))
    diff_init_traj_yaw_span_deg = float(getattr(config.planner, "diff_init_traj_yaw_span_deg", 30.0))
    diff_init_traj_speed_scale_span = float(getattr(config.planner, "diff_init_traj_speed_scale_span", 0.2))
    diff_dt = float(getattr(config.planner, "diff_dt", 0.2))
    use_seeded_diff_init = not (
        diff_init_traj_strategy == "gaussian"
        and diff_init_traj_noise_scale == 1.0
        and diff_init_traj_yaw_span_deg == 30.0
        and diff_init_traj_speed_scale_span == 0.2
    )

    # ── Status dimension summary（所有 planner 类型共享）──
    _status_layouts = {
        (3, False): "[velocity, acceleration, yaw_rate]",
        (4, False): "[vx, vy, ax, ay]",
        (7, True): "[cmd(4), velocity, acceleration, yaw_rate]",
        (8, True): "[cmd(4), vx, vy, ax, ay]",
        (8, False): "[vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
        (12, True): "[cmd(4), vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
    }
    _layout = _status_layouts.get(
        (planner_status_dim, _use_cmd),
        f"custom({planner_status_dim}d)",
    )
    logger.info(
        f"[Status Summary] planner_status_dim={planner_status_dim}, "
        f"command_dim={planner_command_dim}, "
        f"use_drive_command={_use_cmd}, "
        f"split_status_embedding={config.planner.split_status_embedding} | layout: {_layout}"
    )

    # 根据 planner_type 选择 planner 实现
    if config.planner.planner_type == "diffusion":
        if config.planner.diff_train_prefix_conditioning:
            if use_seeded_diff_init:
                from app.vjepa_cowa_world_model.models import PrefixConditionedSeededDiffusionPlanner as PlannerCls
            else:
                from app.vjepa_cowa_world_model.models import PrefixConditionedDiffusionPlanner as PlannerCls

            planner_kwargs = dict(
                encoder_dim=encoder_dim,
                num_poses=num_poses,
                status_dim=planner_status_dim,
                hidden_dim=config.planner.diff_hidden_dim,
                depth=config.planner.diff_num_layers,
                heads=config.planner.diff_num_heads,
                dropout=config.planner.diff_dropout,
                mlp_ratio=config.planner.diff_mlp_ratio,
                traj_dim=config.planner.diff_traj_dim,
                sde_beta_min=config.planner.diff_sde_beta_min,
                sde_beta_max=config.planner.diff_sde_beta_max,
                num_samples=config.planner.diff_num_samples,
                inference_steps=config.planner.diff_inference_steps,
                use_z_context=config.planner.use_z_context,
                tokens_per_frame=tokens_per_frame,
                trajectory_token_mode=config.planner.diff_trajectory_token_mode,
                use_last_frame_only=config.planner.diff_use_last_frame_only,
                use_action_history=use_action_history,
                action_history_dim=action_history_dim,
                num_observed_frames=num_observed_frames,
                train_min_prefix_frames=config.planner.diff_train_min_prefix_frames,
                train_full_prefix_prob=config.planner.diff_train_full_prefix_prob,
                num_modes=getattr(config.planner, "diff_num_modes", 1),
                use_anchor_frame=getattr(config.planner, "diff_use_anchor_frame", False),
                independent_modes=getattr(config.planner, "diff_independent_modes", False),
                cls_loss_weight=getattr(config.planner, "diff_cls_loss_weight", 1.0),
                reg_loss_weight=getattr(config.planner, "diff_reg_loss_weight", 1.0),
                vel_loss_weight=getattr(config.planner, "diff_vel_loss_weight", 0.5),
                yaw_loss_weight=getattr(config.planner, "diff_yaw_loss_weight", 0.5),
                awta_init_temperature=getattr(config.planner, "awta_init_temperature", 8.0),
                awta_min_temperature=getattr(config.planner, "awta_min_temperature", 0.1),
                conf_temperature=getattr(config.planner, "diff_conf_temperature", 1.5),
                cls_th=getattr(config.planner, "diff_cls_th", 2.0),
                cls_ignore=getattr(config.planner, "diff_cls_ignore", 0.2),
                command_dim=planner_command_dim,
                adaln_version=diff_adaln_version,
                mode_token_expansion=getattr(config.planner, "diff_mode_token_expansion", False),
            )
            if use_seeded_diff_init:
                planner_kwargs.update(
                    init_traj_strategy=diff_init_traj_strategy,
                    init_traj_noise_scale=diff_init_traj_noise_scale,
                    init_traj_yaw_span_deg=diff_init_traj_yaw_span_deg,
                    init_traj_speed_scale_span=diff_init_traj_speed_scale_span,
                    dt=diff_dt,
                )

            planner = PlannerCls(**planner_kwargs).to(device)
            planner_impl_name = PlannerCls.__name__
        else:
            if use_seeded_diff_init:
                from app.vjepa_cowa_world_model.models import SeededDiffusionPlanner as PlannerCls
            else:
                from app.vjepa_cowa_world_model.models import DiffusionPlanner as PlannerCls

            planner_kwargs = dict(
                encoder_dim=encoder_dim,
                num_poses=num_poses,
                status_dim=planner_status_dim,
                hidden_dim=config.planner.diff_hidden_dim,
                depth=config.planner.diff_num_layers,
                heads=config.planner.diff_num_heads,
                dropout=config.planner.diff_dropout,
                mlp_ratio=config.planner.diff_mlp_ratio,
                traj_dim=config.planner.diff_traj_dim,
                sde_beta_min=config.planner.diff_sde_beta_min,
                sde_beta_max=config.planner.diff_sde_beta_max,
                num_samples=config.planner.diff_num_samples,
                inference_steps=config.planner.diff_inference_steps,
                use_z_context=config.planner.use_z_context,
                tokens_per_frame=tokens_per_frame,
                trajectory_token_mode=config.planner.diff_trajectory_token_mode,
                use_last_frame_only=config.planner.diff_use_last_frame_only,
                use_action_history=use_action_history,
                action_history_dim=action_history_dim,
                num_observed_frames=num_observed_frames,
                num_modes=getattr(config.planner, "diff_num_modes", 1),
                use_anchor_frame=getattr(config.planner, "diff_use_anchor_frame", False),
                independent_modes=getattr(config.planner, "diff_independent_modes", False),
                cls_loss_weight=getattr(config.planner, "diff_cls_loss_weight", 1.0),
                reg_loss_weight=getattr(config.planner, "diff_reg_loss_weight", 1.0),
                vel_loss_weight=getattr(config.planner, "diff_vel_loss_weight", 0.5),
                yaw_loss_weight=getattr(config.planner, "diff_yaw_loss_weight", 0.5),
                awta_init_temperature=getattr(config.planner, "awta_init_temperature", 8.0),
                awta_min_temperature=getattr(config.planner, "awta_min_temperature", 0.1),
                conf_temperature=getattr(config.planner, "diff_conf_temperature", 1.5),
                cls_th=getattr(config.planner, "diff_cls_th", 2.0),
                cls_ignore=getattr(config.planner, "diff_cls_ignore", 0.2),
                command_dim=planner_command_dim,
                adaln_version=diff_adaln_version,
                mode_token_expansion=getattr(config.planner, "diff_mode_token_expansion", False),
            )
            if use_seeded_diff_init:
                planner_kwargs.update(
                    init_traj_strategy=diff_init_traj_strategy,
                    init_traj_noise_scale=diff_init_traj_noise_scale,
                    init_traj_yaw_span_deg=diff_init_traj_yaw_span_deg,
                    init_traj_speed_scale_span=diff_init_traj_speed_scale_span,
                    dt=diff_dt,
                )

            planner = PlannerCls(**planner_kwargs).to(device)
            planner_impl_name = PlannerCls.__name__

        planner_params = sum(p.numel() for p in planner.parameters())
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"({planner_impl_name}, type=DiT, hidden_dim={config.planner.diff_hidden_dim}, "
            f"depth={config.planner.diff_num_layers}, heads={config.planner.diff_num_heads}, "
            f"traj_dim={config.planner.diff_traj_dim}, num_poses={num_poses}, "
            f"trajectory_token_mode={config.planner.diff_trajectory_token_mode}, "
            f"interleave_predictor_sampling={config.planner.diff_interleave_predictor_sampling}, "
            f"sde_beta=[{config.planner.diff_sde_beta_min}, {config.planner.diff_sde_beta_max}], "
            f"inference_steps={config.planner.diff_inference_steps}, "
            f"num_samples={config.planner.diff_num_samples}, "
            f"num_modes={getattr(config.planner, 'diff_num_modes', 1)}, "
            f"seed_init={diff_init_traj_strategy}, "
            f"status_dim={planner_status_dim}, command_dim={planner_command_dim})"
        )
        return planner

    # 默认: transformer planner (原始实现)
    from app.vjepa_cowa_world_model.models import MultiModalTemporalPlanner

    planner = MultiModalTemporalPlanner(
        encoder_dim=encoder_dim,
        tf_d_model=config.planner.tf_d_model,
        tf_d_ffn=config.planner.tf_d_ffn,
        tf_num_layers=config.planner.tf_num_layers,
        tf_num_head=config.planner.tf_num_head,
        tf_dropout=config.planner.tf_dropout,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=num_time_steps,
        num_context_frames=num_context_frames,
        status_dim=planner_status_dim,
        use_spatial_tokens=config.planner.use_spatial_tokens,
        num_modes=config.planner.num_modes,
        use_temporal=config.planner.use_temporal,
        use_time_aligned_bias=config.planner.temporal_alignment,
        use_z_context=config.planner.use_z_context,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_observed_tokens=config.planner.use_observed_tokens,
        use_action_history=use_action_history,
        action_history_dim=action_history_dim,
        enable_rl_actor_critic=enable_rl_actor_critic,
        rl_action_dim=config.planner.rl_action_dim,
        num_observed_frames=num_observed_frames,
        command_dim=planner_command_dim,
    ).to(device)

    # 打印参数量和配置信息
    planner_params = sum(p.numel() for p in planner.parameters())

    if config.planner.use_z_context:
        input_src = "z_context (first-frame encoder output)"
    elif config.planner.use_observed_tokens:
        input_src = (
            f"z_observed+z_ar (observed {getattr(config.train, 'num_observed_frames', 1)} frames + predictor output)"
        )
    else:
        input_src = "z_ar (predictor output)"

    if config.train.predictor_inference_consistent:
        status_info = f"status_dim={planner_status_dim} (inference_consistent, command_dim={planner_command_dim})"
    elif config.rl.enabled:
        status_info = f"status_dim={planner_status_dim} (rl:{config.rl.status_mode})"
    elif config.planner.use_states_for_planner:
        status_info = f"status_dim={planner_status_dim} (raw_states)"
    else:
        status_info = f"status_dim={planner_status_dim} (extracted)"

    rl_info = (
        f", actor_critic={enable_rl_actor_critic}, rl_action_dim={config.planner.rl_action_dim}"
        if enable_rl_actor_critic
        else ""
    )

    if config.planner.use_temporal and config.planner.use_z_context:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(TemporalPlanner, input={input_src}, num_context_frames={num_context_frames}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, {status_info}{rl_info})"
        )
    elif config.planner.use_temporal:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(TemporalPlanner, input={input_src}, num_time_steps={num_time_steps}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, "
            f"temporal_alignment={config.planner.temporal_alignment}, {status_info}{rl_info})"
        )
    else:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(SingleFramePlanner, input={input_src}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, {status_info}{rl_info})"
        )

    return planner


def compile_models(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_head: Optional[nn.Module] = None,
    compile_model: bool = False,
) -> None:
    """
    编译模型 (torch.compile)

    Args:
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_head: seg_head 模型 (可选)
        compile_model: 是否编译模型
    """
    if not compile_model:
        return

    logger.info("Compiling encoder, target_encoder, and predictor.")
    torch._dynamo.config.optimize_ddp = False
    encoder.compile()
    target_encoder.compile()
    predictor.compile()
    if seg_head is not None:
        seg_head.compile()


def get_encoder_embed_dim(encoder: nn.Module) -> int:
    """
    获取 encoder 的嵌入维度

    Args:
        encoder: encoder 模型

    Returns:
        int: 嵌入维度
    """
    core = encoder.module if hasattr(encoder, "module") else encoder
    if hasattr(core, "embed_dim"):
        return int(core.embed_dim)
    return int(core.backbone.embed_dim)
