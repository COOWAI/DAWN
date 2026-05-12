# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
分布式训练初始化模块

提供分布式训练环境的初始化和设备设置功能。
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from src.utils.distributed import init_distributed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def setup_distributed(seed: int = 0) -> Tuple[int, int]:
    """
    初始化分布式训练环境

    Args:
        seed: 随机种子

    Returns:
        Tuple[int, int]: (world_size, rank)
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # 设置多进程启动方式
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 已经设置过，忽略
        pass

    # 初始化分布式后端
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # 打印 CUDA 信息
    logger.info(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"torch.cuda.current_device()={torch.cuda.current_device()}, "
        f"torch.cuda.device_count()={torch.cuda.device_count()}"
    )

    return world_size, rank


def setup_device(rank: int) -> torch.device:
    """
    设置训练设备

    Args:
        rank: 当前进程的 rank

    Returns:
        torch.device: 分配的设备
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")

    # 使用 LOCAL_RANK 分配不同的 GPU (torchrun 会自动设置 LOCAL_RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logger.info(f"Using device: cuda:{local_rank}")

    return device


def get_local_rank() -> int:
    """
    获取当前进程的本地 rank

    Returns:
        int: 本地 rank
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process(rank: int) -> bool:
    """
    判断是否为主进程

    Args:
        rank: 当前进程的 rank

    Returns:
        bool: 是否为主进程
    """
    return rank == 0
