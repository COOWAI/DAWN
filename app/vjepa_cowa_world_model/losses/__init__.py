# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
损失函数模块

包含:
- wta_loss, wta_loss_v2, wta_loss_v3: 多模态 WTA 损失
- single_model_loss: 单模型损失
- l1_length_normalized_loss: 长度归一化 L1 损失
- awta_temperature_schedule: 退火温度调度器
- get_loss_function: 损失函数工厂
- convert_trajectory_3d_to_6d: 3D→6D 轨迹转换 (扩散 planner 用, 兼容别名)
- convert_trajectory_3d_to_nd: 3D→ND 轨迹转换 (4D 或 6D)
- compute_diffusion_metrics: 扩散 planner 评估指标
"""

from .ppo_loss import (
    gaussian_log_prob,
    gaussian_entropy,
    ppo_loss,
)
from .diffusion_loss import compute_diffusion_metrics, convert_trajectory_3d_to_6d, convert_trajectory_3d_to_nd
from .single_model_loss import l1_length_normalized_loss, single_model_loss
from .token_ae_sem_loss import sem_bbox_loss
from .sigreg import SIGReg, pool_tokens_per_frame
from .wta_loss import awta_temperature_schedule, get_loss_function, wta_loss, wta_loss_v2, wta_loss_v3

__all__ = [
    "wta_loss",
    "wta_loss_v2",
    "wta_loss_v3",
    "awta_temperature_schedule",
    "get_loss_function",
    "single_model_loss",
    "l1_length_normalized_loss",
    "sem_bbox_loss",
    "convert_trajectory_3d_to_6d",
    "convert_trajectory_3d_to_nd",
    "compute_diffusion_metrics",
    "SIGReg",
    "pool_tokens_per_frame",
    "gaussian_log_prob",
    "gaussian_entropy",
    "ppo_loss",
]
