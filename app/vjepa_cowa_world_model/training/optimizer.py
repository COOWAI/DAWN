# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
优化器和 DDP 包装模块

提供优化器创建、DDP 包装和参数冻结功能。
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_droid.utils import init_opt_no_resample_world_model
from src.utils.logging import get_logger
from src.utils.schedulers import LinearDecaySchedule

from .config import TrainingConfig

logger = get_logger(__name__)


def create_optimizer_and_scheduler(
    config: TrainingConfig,
    encoder: nn.Module,
    predictor: nn.Module,
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    ipe: int,
) -> Tuple[torch.optim.Optimizer, Optional[GradScaler], Any, Any]:
    """
    创建优化器、调度器和 scaler

    Args:
        config: 训练配置
        encoder: encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        ipe: 每个 epoch 的迭代次数

    Returns:
        Tuple: (optimizer, scaler, scheduler, wd_scheduler)
    """
    opt_config = config.optimization

    optimizer, scaler, scheduler, wd_scheduler = init_opt_no_resample_world_model(
        encoder=encoder,
        predictor=predictor,
        seg_head=seg_head if config.segmentation.use_segmentation else None,
        seg_neck=seg_neck,
        wd=opt_config.weight_decay,
        final_wd=opt_config.final_weight_decay,
        start_lr=opt_config.start_lr,
        ref_lr=opt_config.lr,
        final_lr=opt_config.final_lr,
        enc_lr_scale=opt_config.enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=opt_config.anneal,
        warmup=opt_config.warmup,
        num_epochs=opt_config.epochs,
        mixed_precision=config.mixed_precision,
        betas=opt_config.betas,
        eps=opt_config.eps,
    )

    # -- Cooldown / anneal: replace scheduler with LinearDecaySchedule
    if opt_config.is_anneal:
        T_max = int(opt_config.ipe_scale * opt_config.epochs * ipe)
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=opt_config.lr,
            final_lr=opt_config.final_lr,
            T_max=T_max,
        )
        logger.info(
            f"[Anneal] Using LinearDecaySchedule: ref_lr={opt_config.lr}, "
            f"final_lr={opt_config.final_lr}, T_max={T_max}"
        )

    return optimizer, scaler, scheduler, wd_scheduler


def add_planner_param_groups(optimizer: torch.optim.Optimizer, planner: nn.Module) -> None:
    """
    将 planner 参数添加到优化器

    Args:
        optimizer: 优化器
        planner: planner 模型
    """
    optimizer.add_param_group(
        {
            "params": [p for n, p in planner.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)],
        }
    )
    optimizer.add_param_group(
        {
            "params": [p for n, p in planner.named_parameters() if ("bias" in n) or (len(p.shape) == 1)],
            "WD_exclude": True,
            "weight_decay": 0,
        }
    )


def add_encoder_param_groups(optimizer: torch.optim.Optimizer, encoder: nn.Module, enc_lr_scale: float = 1.0) -> None:
    """
    将 encoder 参数添加到优化器

    Args:
        optimizer: 优化器
        encoder: encoder 模型
        enc_lr_scale: encoder 学习率缩放因子
    """
    optimizer.add_param_group(
        {
            "params": [p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)],
            "lr_scale": enc_lr_scale,
        }
    )
    optimizer.add_param_group(
        {
            "params": [p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)],
            "WD_exclude": True,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        }
    )
    logger.info(f"Added encoder parameters to optimizer with lr_scale={enc_lr_scale}")


def planner_has_dynamic_unused_parameters(
    planner: Optional[nn.Module],
    use_status_for_planner: bool = True,
    use_temporal: bool = False,
    use_z_context: bool = False,
) -> bool:
    """Return whether planner DDP should tolerate dynamically unused parameters."""
    del planner
    return (not use_status_for_planner) or (use_temporal and use_z_context)


def wrap_ddp_models(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    encoder_train: bool = False,
    use_planner: bool = True,
    use_status_for_planner: bool = True,
    use_temporal: bool = False,
    use_z_context: bool = False,
) -> Dict[str, nn.Module]:
    """
    用 DDP 包装所有模型

    Args:
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        encoder_train: encoder 是否训练
        use_planner: 是否使用 planner
        use_status_for_planner: planner 是否使用 status 特征
        use_temporal: 是否使用时序预测
        use_z_context: 是否使用 z_context

    Returns:
        Dict[str, nn.Module]: 包装后的模型字典
    """
    # DDP 包装
    # DDP refuses to wrap a fully-frozen module ("not needed when no parameter requires grad"),
    # so skip the wrap when the encoder has zero trainable params (e.g. Drive-JEPA adapter
    # with frozen backbone and no internal projector).
    encoder_has_trainable = any(p.requires_grad for p in encoder.parameters())
    if encoder_has_trainable:
        encoder = DistributedDataParallel(
            encoder,
            static_graph=not encoder_train,
            find_unused_parameters=encoder_train,
        )
    else:
        logger.info(
            "[DDP] Skipping DDP wrap for encoder: no parameter requires grad "
            "(fully-frozen encoder; gradients do not need synchronization)."
        )
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    target_encoder_has_trainable = any(p.requires_grad for p in target_encoder.parameters())
    if target_encoder_has_trainable:
        target_encoder = DistributedDataParallel(target_encoder)
    else:
        logger.info("[DDP] Skipping DDP wrap for target_encoder: no parameter requires grad.")

    if seg_head is not None:
        seg_head = DistributedDataParallel(seg_head, find_unused_parameters=False)
    if seg_neck is not None:
        seg_neck = DistributedDataParallel(seg_neck, find_unused_parameters=False)

    if use_planner and planner is not None:
        has_unused = planner_has_dynamic_unused_parameters(
            planner,
            use_status_for_planner=use_status_for_planner,
            use_temporal=use_temporal,
            use_z_context=use_z_context,
        )
        planner = DistributedDataParallel(planner, find_unused_parameters=has_unused)

    return {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "seg_neck": seg_neck,
        "seg_head": seg_head,
        "planner": planner,
    }


def freeze_parameters(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    encoder_train: bool = False,
    predictor_train: bool = True,
    seg_head_train: bool = True,
) -> None:
    """
    根据训练标志冻结参数

    Args:
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        encoder_train: encoder 是否训练
        predictor_train: predictor 是否训练
        seg_head_train: seg_head 是否训练
    """
    # Encoder
    for p in encoder.parameters():
        p.requires_grad = encoder_train

    # Target encoder (始终冻结)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Predictor
    for p in predictor.parameters():
        p.requires_grad = predictor_train

    # Segmentation
    if seg_head is not None:
        for p in seg_head.parameters():
            p.requires_grad = seg_head_train
    if seg_neck is not None:
        for p in seg_neck.parameters():
            p.requires_grad = seg_head_train

    # Planner (始终训练，如果启用)
    # 注意: planner 参数默认 requires_grad=True


def log_trainable_parameters(
    encoder: nn.Module,
    predictor: nn.Module,
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    use_planner: bool = True,
    predictor_state_mode: str = "none",
) -> None:
    """
    打印可训练参数统计

    Args:
        encoder: encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        optimizer: 优化器
        use_planner: 是否使用 planner
        predictor_state_mode: predictor 的状态模式描述
    """
    # 统计所有参与梯度优化的参数量
    total_trainable_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                total_trainable_params += p.numel()

    # 分模块统计
    planner_params_total = (
        sum(p.numel() for p in planner.parameters() if p.requires_grad) if use_planner and planner is not None else 0
    )

    logger.info(f"{'='*50}")
    logger.info(f"Trainable Parameters Summary:")
    logger.info(f"  encoder:           {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    logger.info(
        f"  predictor:         {sum(p.numel() for p in predictor.parameters() if p.requires_grad) / 1e6:>8.2f}M (state_mode={predictor_state_mode})"
    )
    if seg_neck is not None:
        logger.info(
            f"  seg_neck:          {sum(p.numel() for p in seg_neck.parameters() if p.requires_grad) / 1e6:>8.2f}M"
        )
    if seg_head is not None:
        logger.info(
            f"  seg_head:          {sum(p.numel() for p in seg_head.parameters() if p.requires_grad) / 1e6:>8.2f}M"
        )
    if use_planner and planner is not None:
        logger.info(f"  planner:           {planner_params_total / 1e6:>8.2f}M")
    logger.info(f"{'─'*50}")
    logger.info(f"  ALL trainable:     {total_trainable_params / 1e6:>8.2f}M")
    logger.info(f"{'='*50}")
