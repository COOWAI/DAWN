# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Checkpoint 保存/加载模块

提供模型权重的保存和加载功能。
"""

import copy
import inspect
import os
import threading
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 异步 Checkpoint 保存
# ---------------------------------------------------------------------------


def _deep_copy_to_cpu(obj: Any) -> Any:
    """
    递归地将嵌套 dict/list 中的 Tensor 拷贝到 CPU，非 Tensor 值做 deepcopy。

    这样后台线程在做 torch.save 时不会与 GPU 训练产生数据竞争。
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _deep_copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        copied = [_deep_copy_to_cpu(v) for v in obj]
        return type(obj)(copied)
    else:
        return copy.deepcopy(obj)


class _AsyncSaver:
    """
    管理后台 checkpoint 写入线程。

    使用方式::

        saver.save(save_dict, path)   # 提交异步保存（自动等待上次完成）
        saver.wait()                  # 显式等待当前保存完成

    线程安全：同一时刻最多只有一个后台写入线程。
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    def wait(self) -> None:
        """等待上一次异步保存完成。如果上一次保存失败，记录 warning。"""
        if self._thread is not None:
            self._thread.join()
            if self._error is not None:
                logger.warning(f"上一次异步 checkpoint 保存失败: {self._error}")
                self._error = None
            self._thread = None

    def save(self, save_dict: Dict[str, Any], path: str) -> None:
        """
        在后台线程中执行 torch.save。

        会先等待上一次保存完成，然后启动新的后台线程。
        save_dict 必须已经在 CPU 上（调用方负责 _deep_copy_to_cpu）。
        """
        self.wait()  # 确保上一次写完

        def _worker() -> None:
            try:
                t0 = time.monotonic()
                torch.save(save_dict, path)
                elapsed = time.monotonic() - t0
                logger.info(f"异步 checkpoint 写入完成: {path} (耗时 {elapsed:.1f}s)")
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()


# 模块级单例，所有 save_training_checkpoint 调用共享
_async_saver = _AsyncSaver()


def wait_for_checkpoint_save() -> None:
    """
    等待异步 checkpoint 保存完成。

    训练脚本应在以下时机调用：
    - 训练循环结束后、进程退出前
    - 需要确保 checkpoint 已落盘时（例如 evaluation 前加载 checkpoint）
    """
    _async_saver.wait()


def _load_checkpoint_broadcast(path: str, rank: int = 0, world_size: int = 1) -> Dict[str, Any]:
    """
    加载 checkpoint，多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank。

    避免所有 rank 同时从 NFS 读取同一个大文件（ViT-G ~4GB），减少 NFS 带宽压力。

    Parameters
    ----------
    path       : checkpoint 文件路径
    rank       : 当前进程 rank
    world_size : 总进程数

    Returns
    -------
    dict: checkpoint 状态字典
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return torch.load(path, map_location="cpu")

    # NOTE:
    # Broadcasting a very large Python checkpoint object can transiently duplicate
    # memory (pickle + object reconstruction) and cause rank crashes/OOM.
    # Default to local load on each rank for robustness; allow explicit broadcast
    # fallback via env var when NFS pressure is the bottleneck.
    load_mode = os.environ.get("VJEPA_CHECKPOINT_LOAD_MODE", "local").strip().lower()
    if load_mode in {"local", "per_rank", "rank_local"}:
        if rank == 0:
            logger.info("Loading checkpoint locally on all ranks (mode=%s): %s", load_mode, path)
        else:
            logger.debug("Rank %d local checkpoint load: %s", rank, path)
        return torch.load(path, map_location="cpu")
    if load_mode not in {"broadcast", "rank0_broadcast"}:
        if rank == 0:
            logger.warning(
                "Unknown VJEPA_CHECKPOINT_LOAD_MODE=%s, fallback to local load on all ranks",
                load_mode,
            )
        return torch.load(path, map_location="cpu")

    if rank == 0:
        logger.info(f"Rank 0 loading checkpoint from disk: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        broadcast_list = [checkpoint]
    else:
        broadcast_list = [None]

    if rank == 0:
        logger.info("Broadcasting checkpoint object to all ranks...")
    dist.broadcast_object_list(broadcast_list, src=0)
    if rank == 0:
        logger.info("Checkpoint broadcast completed")

    checkpoint = broadcast_list[0]
    if checkpoint is None:
        raise RuntimeError("Received empty checkpoint object from broadcast")
    return checkpoint


_TORCH_LOAD_PARAMS = inspect.signature(torch.load).parameters


def _torch_load_checkpoint(
    path: str,
    *,
    map_location: str = "cpu",
    weights_only: bool = False,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if "mmap" in _TORCH_LOAD_PARAMS:
        kwargs["mmap"] = True
    if "weights_only" in _TORCH_LOAD_PARAMS:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def load_state_dict_helper(model: nn.Module, state_dict: Dict[str, Any], name: str) -> None:
    """
    通用状态字典加载辅助函数，自动处理 DDP 的 module. 前缀

    Args:
        model: 模型
        state_dict: 状态字典
        name: 模型名称 (用于日志)
    """
    model_unwrapped = model.module if hasattr(model, "module") else model

    # 移除 'module.' 前缀
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if hasattr(model_unwrapped, "base_model") and hasattr(model_unwrapped.base_model, "model"):
        base_model = model_unwrapped.base_model.model
        # PEFT/LoRA 将 target module 的 weight/bias 移到 base_layer 下,
        # 例如 qkv.weight → qkv.base_layer.weight, 需要自动 remap
        model_keys = set(base_model.state_dict().keys())
        remapped = {}
        for k, v in new_state_dict.items():
            if k in model_keys:
                remapped[k] = v
            else:
                # 尝试插入 .base_layer
                for suffix in (".weight", ".bias"):
                    if k.endswith(suffix):
                        new_key = k[: -len(suffix)] + ".base_layer" + suffix
                        if new_key in model_keys:
                            remapped[new_key] = v
                            break
                else:
                    remapped[k] = v
        missing, unexpected = base_model.load_state_dict(remapped, strict=False)
    else:
        missing, unexpected = model_unwrapped.load_state_dict(new_state_dict, strict=False)

    # LoRA 新增的 adapter 权重（lora_A/lora_B/lora_embedding_A/B）不在旧 checkpoint 中是正常的
    _LORA_SUFFIXES = ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B.")
    missing = [k for k in missing if not any(k.endswith(s) or s in k for s in _LORA_SUFFIXES)]
    unexpected = [k for k in unexpected if not any(k.endswith(s) or s in k for s in _LORA_SUFFIXES)]

    if name == "predictor":
        predictor_query = getattr(model_unwrapped, "future_query_tokens", None)
        if predictor_query is not None:
            for query_key in ("future_query_tokens", "base_model.model.future_query_tokens"):
                query_value = new_state_dict.get(query_key)
                if query_value is None:
                    continue
                if tuple(query_value.shape) != tuple(predictor_query.shape):
                    raise RuntimeError(
                        "Checkpoint mismatch for predictor future_query_tokens: "
                        f"checkpoint_shape={tuple(query_value.shape)}, model_shape={tuple(predictor_query.shape)}"
                    )
                predictor_query.data.copy_(query_value.to(device=predictor_query.device, dtype=predictor_query.dtype))
                logger.info("Loaded predictor future_query_tokens from checkpoint key '%s'", query_key)
                break

        predictor_query_missing = [k for k in missing if k.endswith("future_query_tokens")]
        predictor_query_unexpected = [k for k in unexpected if k.endswith("future_query_tokens")]
        if predictor_query_missing or predictor_query_unexpected:
            logger.warning(
                "Loaded predictor with future_query_tokens checkpoint compatibility: " "missing=%s, unexpected=%s",
                predictor_query_missing,
                predictor_query_unexpected,
            )
        missing = [k for k in missing if not k.endswith("future_query_tokens")]
        unexpected = [k for k in unexpected if not k.endswith("future_query_tokens")]

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for '{name}': "
            f"missing={len(missing)} keys, unexpected={len(unexpected)} keys.\n"
            f"  missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"  unexpected: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    logger.info(f"Loaded {name}: all keys matched (missing=0, unexpected=0)")


def load_pretrained_checkpoint(
    path: str,
    encoder: Optional[nn.Module],
    target_encoder: Optional[nn.Module],
    predictor: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    load_encoder: bool = True,
    load_predictor: bool = False,
    load_seg: bool = True,
    load_planner: bool = True,
    context_encoder_key: str = "encoder",
    target_encoder_key: str = "target_encoder",
    rank: int = 0,
    world_size: int = 1,
    predictor_checkpoint: Optional[str] = None,
) -> None:
    """
    加载预训练 checkpoint

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank，
    避免所有 rank 同时从 NFS 读取同一个大文件。

    Args:
        path: checkpoint 路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        load_encoder: 是否加载 encoder 权重
        load_predictor: 是否加载 predictor 权重
        load_seg: 是否加载 seg 权重
        load_planner: 是否加载 planner 权重
        context_encoder_key: checkpoint 中 encoder 的 key 名称
            (默认 "encoder"，V-JEPA 2.1 可能是 "ema_encoder" 或 "target_encoder")
        target_encoder_key: checkpoint 中 target_encoder 的 key 名称
            (默认 "target_encoder")
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        predictor_checkpoint: predictor 独立 checkpoint 路径，设置后优先从该文件加载 predictor
    """
    if path is None or not os.path.exists(path):
        if path is not None:
            logger.warning(f"Full pretrained checkpoint not found: {path}")
        return

    logger.info(f"Loading full pretrained checkpoint from {path}")
    checkpoint = _load_checkpoint_broadcast(path, rank=rank, world_size=world_size)

    if load_encoder:
        # 支持多种 encoder key: encoder, ema_encoder, target_encoder
        encoder_state = None
        for key in [context_encoder_key, "encoder", "ema_encoder", "target_encoder"]:
            if key in checkpoint:
                encoder_state = checkpoint[key]
                logger.info(f"Found encoder weights under key '{key}'")
                break
        if encoder_state is not None:
            load_state_dict_helper(encoder, encoder_state, "encoder")
            target_encoder.load_state_dict(encoder.state_dict())
            logger.info("Synchronized target_encoder with encoder")
        else:
            logger.warning("No encoder weights found in checkpoint")

    if load_predictor:
        # 优先从 predictor_checkpoint 加载，否则回退到 pretrain_checkpoint_full
        if predictor_checkpoint and os.path.exists(predictor_checkpoint):
            logger.info(f"Loading predictor from separate checkpoint: {predictor_checkpoint}")
            pred_ckpt = _load_checkpoint_broadcast(predictor_checkpoint, rank=rank, world_size=world_size)
            if "predictor" in pred_ckpt:
                load_state_dict_helper(predictor, pred_ckpt["predictor"], "predictor")
            else:
                logger.warning(
                    f"No 'predictor' key in {predictor_checkpoint}, available keys: {list(pred_ckpt.keys())[:10]}"
                )
        elif "predictor" in checkpoint:
            load_state_dict_helper(predictor, checkpoint["predictor"], "predictor")

    if load_seg:
        if seg_neck is not None and "seg_neck" in checkpoint:
            load_state_dict_helper(seg_neck, checkpoint["seg_neck"], "seg_neck")
        if seg_head is not None and "seg_head" in checkpoint:
            load_state_dict_helper(seg_head, checkpoint["seg_head"], "seg_head")

    if load_planner and planner is not None and "planner" in checkpoint:
        load_state_dict_helper(planner, checkpoint["planner"], "planner")

    logger.info("Full pretrained checkpoint loaded successfully!")


def save_training_checkpoint(
    path: str,
    encoder: Optional[nn.Module],
    target_encoder: Optional[nn.Module],
    predictor: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    seg_head: Optional[nn.Module],
    planner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: Any,
    wd_scheduler: Any,
    epoch: int,
    loss: float,
    batch_size: int,
    world_size: int,
    lr: float,
    rank: int,
    use_planner: bool = True,
    encoder_train: bool = True,
    encoder_ema: bool = True,
    predictor_train: bool = True,
    seg_head_train: bool = True,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存训练 checkpoint（只保存训练中实际更新的模块）

    冻结且未通过 EMA 更新的模块不会被保存，以减小 checkpoint 体积。
    resume 时，未保存的模块会从 pretrained checkpoint 加载。

    Args:
        path: checkpoint 保存路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_neck: seg_neck 模型
        seg_head: seg_head 模型
        planner: planner 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        epoch: 当前 epoch
        loss: 当前损失
        batch_size: 批大小
        world_size: 进程数
        lr: 学习率
        rank: 当前进程 rank
        use_planner: 是否使用 planner
        encoder_train: encoder 是否参与训练（梯度更新）
        encoder_ema: target_encoder 是否通过 EMA 更新
        predictor_train: predictor 是否参与训练
        seg_head_train: seg_head/seg_neck 是否参与训练
    """
    if rank != 0:
        return

    save_dict = {
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "batch_size": batch_size,
        "world_size": world_size,
        "lr": lr,
    }

    # 记录保存了哪些模块（用于 debug / 日志）
    saved_modules = []
    skipped_modules = []

    # encoder: 仅在参与梯度训练时保存
    if encoder_train:
        save_dict["encoder"] = encoder.state_dict()
        saved_modules.append("encoder")
    else:
        skipped_modules.append("encoder")

    # target_encoder: 仅在 EMA 更新时保存（冻结且无 EMA 时与 pretrained 一致）
    if encoder_ema:
        save_dict["target_encoder"] = target_encoder.state_dict()
        saved_modules.append("target_encoder")
    else:
        skipped_modules.append("target_encoder")

    # predictor: 仅在参与训练时保存
    if predictor_train:
        save_dict["predictor"] = predictor.state_dict()
        saved_modules.append("predictor")
    else:
        skipped_modules.append("predictor")

    # seg_head / seg_neck: 仅在参与训练时保存
    if seg_head_train:
        if seg_head is not None:
            save_dict["seg_head"] = seg_head.state_dict()
            saved_modules.append("seg_head")
        if seg_neck is not None:
            save_dict["seg_neck"] = seg_neck.state_dict()
            saved_modules.append("seg_neck")
    else:
        if seg_head is not None:
            skipped_modules.append("seg_head")
        if seg_neck is not None:
            skipped_modules.append("seg_neck")

    # planner: 始终可训练，始终保存
    if use_planner and planner is not None:
        save_dict["planner"] = planner.state_dict()
        saved_modules.append("planner")
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
        saved_modules.append("scheduler")
    if wd_scheduler is not None:
        save_dict["wd_scheduler"] = wd_scheduler.state_dict()
        saved_modules.append("wd_scheduler")

    if extra_state is not None:
        save_dict.update(extra_state)

    # 保存元数据，方便 resume 时判断 checkpoint 内容
    save_dict["saved_modules"] = saved_modules

    # 异步保存：先将 state_dict 拷贝到 CPU，然后在后台线程中 torch.save
    save_dict_cpu = _deep_copy_to_cpu(save_dict)
    logger.info(f"checkpoint 异步保存已提交: {path}")
    logger.info(f"  保存的模块: {saved_modules}")
    if skipped_modules:
        logger.info(f"  跳过的冻结模块: {skipped_modules}（resume 时从 pretrained 加载）")
    _async_saver.save(save_dict_cpu, path)


def resume_from_checkpoint(
    resume_path: Optional[str],
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_head: Optional[nn.Module],
    seg_neck: Optional[nn.Module],
    planner: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: Any,
    wd_scheduler: Any,
    use_planner: bool = True,
    load_planner: Optional[bool] = None,
    rank: int = 0,
    world_size: int = 1,
    use_broadcast: bool = True,
    model_only: bool = False,
) -> int:
    """
    从 checkpoint 恢复训练

    多卡场景下仅 rank 0 从磁盘读取，通过 broadcast 分发给其他 rank。

    Args:
        resume_path: checkpoint 路径
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_head: seg_head 模型
        seg_neck: seg_neck 模型
        planner: planner 模型
        optimizer: 优化器
        scaler: GradScaler
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        use_planner: 是否使用 planner
        load_planner: 是否从 resume checkpoint 加载 planner；None 表示跟随 use_planner
        rank: 当前进程 rank（默认 0，向后兼容单卡场景）
        world_size: 总进程数（默认 1，向后兼容单卡场景）
        use_broadcast: 是否使用 broadcast 分发 checkpoint（False 则每个 rank 独立从磁盘读取）
        model_only: 仅加载模型权重，跳过 optimizer/scheduler/epoch（用于分阶段训练）

    Returns:
        int: 开始的 epoch（model_only=True 时始终返回 0）
    """
    if resume_path is None or not os.path.exists(resume_path):
        return 0

    if model_only:
        logger.info("resume_model_only=True: 仅加载模型权重，跳过 optimizer/scheduler/epoch")

    should_load_planner = use_planner if load_planner is None else use_planner and load_planner
    if use_planner and planner is not None and not should_load_planner:
        logger.info("Skipping planner resume load because load_planner=False")

    from app.vjepa_droid.utils import resume_training_checkpoint

    start_epoch = resume_training_checkpoint(
        resume_path=resume_path,
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        seg_head=seg_head,
        seg_neck=seg_neck,
        planner=planner if should_load_planner else None,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        logger=logger,
        rank=rank,
        world_size=world_size,
        use_broadcast=use_broadcast,
        model_only=model_only,
    )

    return start_epoch


def load_checkpoint(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a checkpoint dictionary without restoring it."""
    if path is None or not os.path.exists(path):
        return None
    return _torch_load_checkpoint(path, map_location="cpu", weights_only=False)


def setup_checkpoint_paths(folder: str, resume_checkpoint: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    设置 checkpoint 路径

    Args:
        folder: 保存文件夹
        resume_checkpoint: 指定的恢复 checkpoint 文件名

    Returns:
        Dict[str, Optional[str]]: 包含各种 checkpoint 路径的字典
    """
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best_ade.pt")

    if resume_checkpoint is not None:
        resume_path = os.path.join(folder, resume_checkpoint)
    else:
        resume_path = latest_path

    if not os.path.exists(resume_path):
        resume_path = None

    return {
        "latest": latest_path,
        "best": best_path,
        "resume": resume_path,
    }
