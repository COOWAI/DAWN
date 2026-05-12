# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
数据加载模块

提供数据增强和数据加载器的创建功能。
"""

import os
from typing import Any, Optional, Tuple

from torch.utils.data import DataLoader, DistributedSampler

from app.vjepa_droid.transforms import make_transforms
from src.utils.logging import get_logger

from .config import TrainingConfig, resolve_proposal_encoder_backbone
from .drive_jepa_transforms import DriveJEPAImageTransform
from .navsim_data import init_navsim_data

logger = get_logger(__name__)


def _is_navsim_enabled(config: TrainingConfig) -> bool:
    navsim = config.data.navsim
    return navsim is not None and navsim.enabled


def _is_mongo_raw_enabled(config: TrainingConfig) -> bool:
    mongo_raw = config.data.mongo_raw
    return mongo_raw is not None and mongo_raw.enabled


def create_transforms(config: TrainingConfig) -> Any:
    """
    创建数据增强 transform

    Args:
        config: 训练配置

    Returns:
        Any: transform 对象
    """
    if config.model.backbone == "drive_jepa_img_encoder":
        return DriveJEPAImageTransform(
            resolution=config.model.drive_jepa_resolution,
            crop_top_bottom=config.model.drive_jepa_crop_top_bottom,
        )

    transform = make_transforms(
        random_horizontal_flip=config.data_aug.horizontal_flip,
        random_resize_aspect_ratio=config.data_aug.random_resize_aspect_ratio,
        random_resize_scale=config.data_aug.random_resize_scale,
        reprob=config.data_aug.reprob,
        auto_augment=config.data_aug.auto_augment,
        motion_shift=config.data_aug.motion_shift,
        crop_size=config.data.crop_size,
    )
    return transform


def create_proposal_transforms(config: TrainingConfig) -> Optional[Any]:
    """Create an optional second transform for an independent proposal encoder."""
    proposal_cfg = getattr(config, "proposal", None)
    if proposal_cfg is None or not proposal_cfg.enabled or not proposal_cfg.use_separate_encoder:
        return None
    if resolve_proposal_encoder_backbone(config) != "drive_jepa_img_encoder":
        return None
    return DriveJEPAImageTransform(
        resolution=proposal_cfg.drive_jepa_resolution,
        crop_top_bottom=proposal_cfg.drive_jepa_crop_top_bottom,
    )


def create_train_dataloader(
    config: TrainingConfig, rank: int, world_size: int, transform: Any = None
) -> Tuple[DataLoader, DistributedSampler]:
    """
    创建训练数据加载器

    Args:
        config: 训练配置
        rank: 当前进程的 rank
        world_size: 进程总数
        transform: 数据增强 transform (可选，如果为 None 则自动创建)

    Returns:
        Tuple[DataLoader, DistributedSampler]: (dataloader, sampler)
    """
    if transform is None:
        transform = create_transforms(config)
    proposal_transform = create_proposal_transforms(config)

    if _is_mongo_raw_enabled(config):
        from .mongo_raw_data import init_mongo_raw_data

        mongo_raw = config.data.mongo_raw
        if mongo_raw is None:
            raise ValueError("Unexpected mongo_raw config state")

        if config.segmentation.use_segmentation:
            raise ValueError("Mongo raw online loader does not support segmentation in the first version")

        logger.info(
            "Initializing Mongo raw training dataset from database=%s collection=%s vehicle_type=%s vehicle_types=%s",
            mongo_raw.database,
            mongo_raw.collection,
            mongo_raw.vehicle_type,
            mongo_raw.vehicle_types,
        )

        loader, sampler = init_mongo_raw_data(
            mongo_cfg=mongo_raw,
            split="train",
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            action_dim=config.train.action_dim,
            shuffle=True,
        )
    elif _is_navsim_enabled(config):
        navsim = config.data.navsim
        if navsim is None:
            raise ValueError("Unexpected navsim config state")

        if not navsim.data_path or not navsim.sensor_blobs_path:
            raise ValueError("data.navsim.data_path and data.navsim.sensor_blobs_path must be configured")

        logger.info(
            "Initializing NavSim training dataset from logs=%s, blobs=%s",
            navsim.data_path,
            navsim.sensor_blobs_path,
        )

        loader, sampler = init_navsim_data(
            data_path=navsim.data_path,
            sensor_blobs_path=navsim.sensor_blobs_path,
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=navsim.camera_name,
            max_scenes=navsim.max_scenes,
            action_dim=config.train.action_dim,
            shuffle=True,
            index_cache=navsim.index_cache,
            window_stride=navsim.window_stride,
            max_frame_gap=navsim.max_frame_gap,
        )
    else:
        dataset_path = config.data.dataset_path
        if dataset_path is None:
            raise ValueError("Training dataset path is not configured")

        logger.info(f"Initializing training dataset from: {dataset_path}")

        from app.vjepa_cowa.cowa import init_data_only_seg

        loader, sampler = init_data_only_seg(
            data_path=dataset_path,
            batch_size=config.data.batch_size,
            fps=config.data.fps,
            camera_views=config.data.camera_views,
            camera_frame=config.data.camera_frame,
            frames_per_clip=config.data.num_target_frames,
            stereo_view=config.data.stereo_view,
            tubelet_size=1,  # 训练时不使用 tubelet，保持与验证一致
            transform=transform,
            collator=None,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            load_segmentation=config.segmentation.use_segmentation,
            seg_data_root=config.segmentation.seg_data_root,
            crop_size=config.data.crop_size,
            action_dim=config.train.action_dim,
        )

    logger.info(f"Training dataset initialized with {len(loader)} batches")

    return loader, sampler


def create_val_dataloader(
    config: TrainingConfig, rank: int, world_size: int, transform: Any = None
) -> Tuple[Optional[DataLoader], Optional[DistributedSampler]]:
    """
    创建验证数据加载器

    Args:
        config: 训练配置
        rank: 当前进程的 rank
        world_size: 进程总数
        transform: 数据增强 transform (可选，如果为 None 则自动创建)

    Returns:
        Tuple[Optional[DataLoader], Optional[DistributedSampler]]: (dataloader, sampler)
    """
    if _is_mongo_raw_enabled(config):
        from .mongo_raw_data import init_mongo_raw_data

        mongo_raw = config.data.mongo_raw
        if mongo_raw is None:
            raise ValueError("Unexpected mongo_raw config state")

        if config.segmentation.use_segmentation:
            raise ValueError("Mongo raw online loader does not support segmentation in the first version")

        if transform is None:
            transform = create_transforms(config)

        logger.info(
            "Initializing Mongo raw validation dataset from database=%s collection=%s "
            "vehicle_type=%s vehicle_types=%s",
            mongo_raw.database,
            mongo_raw.collection,
            mongo_raw.vehicle_type,
            mongo_raw.vehicle_types,
        )

        try:
            loader, sampler = init_mongo_raw_data(
                mongo_cfg=mongo_raw,
                split="val",
                batch_size=config.data.batch_size,
                frames_per_clip=config.data.num_target_frames,
                fps=config.data.fps,
                tubelet_size=1,
                transform=transform,
                num_workers=config.data.num_workers,
                world_size=world_size,
                pin_mem=config.data.pin_mem,
                persistent_workers=config.data.persistent_workers,
                rank=rank,
                action_dim=config.train.action_dim,
                shuffle=False,
                drop_last=False,
            )
        except ValueError as exc:
            logger.warning("Mongo raw validation dataset unavailable. Validation will be skipped.")
            logger.warning(f"  Reason: {exc}")
            return None, None
    elif _is_navsim_enabled(config):
        navsim = config.data.navsim
        if navsim is None:
            raise ValueError("Unexpected navsim config state")

        val_data_path = navsim.val_data_path
        val_sensor_blobs_path = navsim.val_sensor_blobs_path

        if not val_data_path or not val_sensor_blobs_path:
            logger.warning(
                "NavSim val_data_path or val_sensor_blobs_path is not configured. "
                "Validation will be skipped. Set explicit val paths in the YAML config."
            )
            return None, None

        if not os.path.exists(val_data_path) or not os.path.exists(val_sensor_blobs_path):
            logger.warning("NavSim validation path does not exist. Validation will be skipped.")
            logger.warning(f"  logs: {val_data_path}")
            logger.warning(f"  sensor_blobs: {val_sensor_blobs_path}")
            return None, None

        if transform is None:
            transform = create_transforms(config)
        proposal_transform = create_proposal_transforms(config)

        logger.info(
            "Initializing NavSim validation dataset from logs=%s, blobs=%s",
            val_data_path,
            val_sensor_blobs_path,
        )

        loader, sampler = init_navsim_data(
            data_path=val_data_path,
            sensor_blobs_path=val_sensor_blobs_path,
            batch_size=config.data.batch_size,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            camera_name=navsim.camera_name,
            max_scenes=navsim.max_val_scenes,
            action_dim=config.train.action_dim,
            shuffle=False,
            index_cache=navsim.index_cache,
            window_stride=navsim.val_window_stride if navsim.val_window_stride is not None else navsim.window_stride,
            max_frame_gap=navsim.max_frame_gap,
            drop_last=False,
        )
    else:
        val_dataset_path = config.data.val_dataset_path

        if val_dataset_path is None or not os.path.exists(val_dataset_path):
            if val_dataset_path:
                logger.warning("Validation dataset not configured or path does not exist. Validation will be skipped.")
                logger.warning(f"  Provided path: {val_dataset_path}")
            return None, None

        if transform is None:
            transform = create_transforms(config)

        logger.info(f"Initializing validation dataset from: {val_dataset_path}")

        from app.vjepa_cowa.cowa import init_data_only_seg

        loader, sampler = init_data_only_seg(
            data_path=val_dataset_path,
            batch_size=config.data.batch_size,
            fps=config.data.fps,
            camera_views=config.data.camera_views,
            camera_frame=config.data.camera_frame,
            frames_per_clip=config.data.num_target_frames,
            stereo_view=config.data.stereo_view,
            tubelet_size=1,  # 验证时不使用 tubelet，保持与训练一致
            transform=transform,
            collator=None,
            num_workers=config.data.num_workers,
            world_size=world_size,
            pin_mem=config.data.pin_mem,
            persistent_workers=config.data.persistent_workers,
            rank=rank,
            load_segmentation=False,  # 验证时不需要分割标注
            seg_data_root="",
            crop_size=config.data.crop_size,
            action_dim=config.train.action_dim,
        )

    logger.info(f"Validation dataset initialized with {len(loader)} batches")

    return loader, sampler


def calculate_iterations_per_epoch(config: TrainingConfig, dataloader: DataLoader) -> int:
    """
    计算每个 epoch 的迭代次数。

    默认根据 dataloader 长度自动计算 ipe（推荐），无需在 YAML 中手动维护。
    当 batch_size 或 world_size（节点数/GPU数）变化时，ipe 会自动适配。

    若 YAML 中显式设置了 ipe（非 null），则作为手动覆盖使用，
    并在与 dataloader 长度不一致时输出警告日志。

    Args:
        config: 训练配置
        dataloader: 数据加载器

    Returns:
        int: 每个 epoch 的迭代次数
    """
    dataset_len = len(dataloader)
    config_ipe = config.optimization.ipe

    if config_ipe is not None:
        # 显式覆盖：使用配置值，但在不一致时发出警告
        if config_ipe != dataset_len:
            logger.warning(
                f"Config ipe ({config_ipe}) differs from dataloader length ({dataset_len}). "
                f"Using config override. Set ipe to null in YAML for auto-detection."
            )
        ipe = config_ipe
    else:
        # 自动计算（推荐）：ipe = len(dataloader) = ceil(dataset_size / world_size) / batch_size
        ipe = dataset_len
        logger.info(f"Auto-detected ipe from dataloader: {ipe}")

    logger.info(f"iterations per epoch: {ipe} (dataloader length: {dataset_len})")

    return ipe
