#!/usr/bin/env python3
"""Offline nuScenes/NavSim-PKL validation entrypoint for train_navsim_v2 planners."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


CHECKPOINT_CANDIDATES = ("best_open_loop.pt", "latest.pt")


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_relative_to_folder(folder: str | Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(folder).expanduser() / candidate
    return candidate.resolve()


def resolve_checkpoint_path(
    config_folder: str | Path,
    explicit_checkpoint: Optional[str],
    resume_checkpoint: Optional[str],
) -> Path:
    """Resolve the checkpoint used for evaluation."""
    if explicit_checkpoint:
        checkpoint_path = _expand_path(explicit_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    folder = _expand_path(config_folder)
    for filename in CHECKPOINT_CANDIDATES:
        candidate = folder / filename
        if candidate.exists():
            return candidate.resolve()

    if resume_checkpoint:
        candidate = _resolve_relative_to_folder(folder, resume_checkpoint)
        if candidate.exists():
            return candidate

    tried = [str(folder / filename) for filename in CHECKPOINT_CANDIDATES]
    if resume_checkpoint:
        tried.append(str(_resolve_relative_to_folder(folder, resume_checkpoint)))
    raise FileNotFoundError("No evaluation checkpoint found. Tried: " + ", ".join(tried))


def resolve_output_dir(config_folder: str | Path, checkpoint_path: str | Path, explicit_output_dir: Optional[str]) -> Path:
    """Resolve the directory where metrics and visualizations are written."""
    if explicit_output_dir:
        return _expand_path(explicit_output_dir)
    checkpoint_name = Path(checkpoint_path).stem
    return (_expand_path(config_folder) / "pure_eval" / checkpoint_name).resolve()


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def apply_cli_overrides(raw_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply small inference-time overrides before dataclass parsing."""
    config = dict(raw_config)
    data_config = dict(config.get("data", {}))
    navsim_config = dict(data_config.get("navsim", {}))

    if args.batch_size is not None:
        data_config["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        data_config["num_workers"] = int(args.num_workers)
    if args.val_data_path is not None:
        navsim_config["val_data_path"] = args.val_data_path
    if args.val_sensor_blobs_path is not None:
        navsim_config["val_sensor_blobs_path"] = args.val_sensor_blobs_path
    if args.max_val_scenes is not None:
        navsim_config["max_val_scenes"] = int(args.max_val_scenes)

    data_config["navsim"] = navsim_config
    config["data"] = data_config
    return config


def json_safe(value: Any) -> Any:
    """Convert tensors, numpy values, and paths into JSON-serializable objects."""
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def setup_runtime(seed: int) -> Tuple[Any, int, int]:
    """Set RNGs, initialize distributed mode when launched by torchrun, and choose device."""
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        from src.utils.distributed import init_distributed

        world_size, rank = init_distributed()
    else:
        world_size, rank = 1, 0

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, world_size, rank


def freeze_for_inference(*models: Any) -> None:
    for model in models:
        if model is None:
            continue
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False


def load_inference_weights(
    config: Any,
    checkpoint_path: Path,
    encoder: Any,
    target_encoder: Any,
    predictor: Any,
    seg_neck: Any,
    seg_head: Any,
    planner: Any,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Load base pretrained weights, then overlay the selected evaluation checkpoint."""
    from app.vjepa_cowa_world_model.training.checkpoint import (
        load_checkpoint,
        load_pretrained_checkpoint,
        load_state_dict_helper,
    )
    from app.vjepa_cowa_world_model.training.config import is_drive_jepa_main_encoder_config
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    if config.meta.pretrain_checkpoint_full:
        load_pretrained_checkpoint(
            config.meta.pretrain_checkpoint_full,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            load_encoder=bool(config.meta.load_encoder and not is_drive_jepa_main_encoder_config(config)),
            load_predictor=bool(config.meta.load_predictor),
            load_seg=bool(config.meta.load_seg),
            load_planner=bool(config.meta.load_planner),
            context_encoder_key=config.meta.context_encoder_key,
            target_encoder_key=config.meta.target_encoder_key,
            rank=rank,
            world_size=world_size,
            predictor_checkpoint=config.meta.predictor_checkpoint,
        )

    checkpoint = load_checkpoint(str(checkpoint_path))
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = []

    def _load_first_available(model: Any, keys: Tuple[str, ...], name: str, required: bool = False) -> bool:
        if model is None:
            return False
        for key in keys:
            if key in checkpoint:
                logger.info("Loading %s from checkpoint key '%s'", name, key)
                load_state_dict_helper(model, checkpoint[key], name)
                loaded.append(name)
                return True
        if required:
            raise KeyError(f"Required checkpoint key for {name!r} not found. Tried: {keys}")
        return False

    _load_first_available(
        encoder,
        (config.meta.context_encoder_key, "encoder", "ema_encoder", "target_encoder"),
        "encoder",
        required=False,
    )
    _load_first_available(
        target_encoder,
        (config.meta.target_encoder_key, "target_encoder", "ema_encoder", "encoder"),
        "target_encoder",
        required=False,
    )
    _load_first_available(predictor, ("predictor",), "predictor", required=False)
    _load_first_available(seg_neck, ("seg_neck",), "seg_neck", required=False)
    _load_first_available(seg_head, ("seg_head",), "seg_head", required=False)
    _load_first_available(planner, ("planner",), "planner", required=bool(config.planner.use_planner))

    logger.info("Loaded checkpoint modules from %s: %s", checkpoint_path, ", ".join(loaded) or "none")
    return checkpoint


def build_runtime(raw_config: Dict[str, Any], checkpoint_path: Path, output_dir: Path) -> Tuple[Any, ...]:
    """Build models, load weights, and return validation runtime objects."""
    import torch

    from app.vjepa_cowa_world_model.training.config import parse_training_config
    from app.vjepa_cowa_world_model.training.data import create_transforms, create_val_dataloader
    from app.vjepa_cowa_world_model.training.main_encoder_runtime import resolve_main_timeline
    from app.vjepa_cowa_world_model.training.models import (
        configure_drive_jepa_encoder_trainability,
        get_encoder_embed_dim,
        init_encoder,
        init_planner,
        init_predictor_runtime_with_token_ae,
        init_segmentation_modules,
        resolve_main_predictor_runtime_overrides,
    )
    from app.vjepa_cowa_world_model.training.predictor_parallel import maybe_register_parallel_predictor_tokens
    from src.utils.logging import get_logger

    config = parse_training_config(raw_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, world_size, rank = setup_runtime(config.meta.seed)
    logger = get_logger(__name__)
    logger.info("Runtime device=%s rank=%d world_size=%d", device, rank, world_size)
    logger.info("Writing evaluation outputs to: %s", output_dir)

    encoder, target_encoder = init_encoder(config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info("encoder_embed_dim: %d", encoder_embed_dim)

    main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(config, encoder)
    main_timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=config.data.num_target_frames)
    logger.info(
        "Main encoder timeline: raw_frames=%d stride=%d predictor_steps=%d observed_steps=%d "
        "future_steps=%d tokens_per_step=%d predictor_img_size=%s",
        main_timeline.raw_num_frames,
        main_timeline.frame_stride,
        main_timeline.num_time_steps,
        main_timeline.num_observed_steps,
        main_timeline.num_future_steps,
        main_timeline.tokens_per_frame,
        predictor_img_size_override if predictor_img_size_override is not None else config.data.crop_size,
    )

    predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
        config,
        device=device,
        encoder_embed_dim=encoder_embed_dim,
        raw_tokens_per_frame_override=main_tokens_override,
        predictor_img_size_override=predictor_img_size_override,
    )
    maybe_register_parallel_predictor_tokens(
        predictor=predictor,
        config=config,
        embed_dim=encoder_embed_dim,
        future_steps=main_timeline.num_future_steps,
        tokens_per_frame=tokens_per_frame,
        device=device,
    )

    seg_neck, seg_head = init_segmentation_modules(device, config.segmentation.use_segmentation)
    num_poses = config.data.num_target_frames - config.train.num_observed_frames
    planner = init_planner(
        config,
        encoder_embed_dim,
        device,
        num_poses=num_poses,
        tokens_per_frame_override=tokens_per_frame,
    )

    load_inference_weights(
        config=config,
        checkpoint_path=checkpoint_path,
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        seg_neck=seg_neck,
        seg_head=seg_head,
        planner=planner,
        rank=rank,
        world_size=world_size,
    )

    configure_drive_jepa_encoder_trainability(encoder, config)
    configure_drive_jepa_encoder_trainability(target_encoder, config, trainable=False)
    freeze_for_inference(encoder, target_encoder, predictor, seg_neck, seg_head, planner, token_ae)

    transform = create_transforms(config)
    val_loader, val_sampler = create_val_dataloader(config, rank, world_size, transform)
    if val_loader is None or val_sampler is None:
        raise RuntimeError("Validation dataloader was not created. Check data.navsim.val_* paths in the YAML config.")

    # Keep references to token_ae/runtime_normalize_reps for future compatibility and diagnostics.
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return config, encoder, predictor, planner, val_loader, val_sampler, token_ae, runtime_normalize_reps, rank, world_size


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from app.vjepa_cowa_world_model.val_command import run_validation
    from src.utils.logging import get_logger

    raw_config = apply_cli_overrides(load_yaml_config(args.config), args)
    config_folder = raw_config.get("folder", "")
    checkpoint_path = resolve_checkpoint_path(
        config_folder=config_folder,
        explicit_checkpoint=args.checkpoint,
        resume_checkpoint=raw_config.get("meta", {}).get("resume_checkpoint"),
    )
    output_dir = resolve_output_dir(config_folder, checkpoint_path, args.output_dir)

    (
        config,
        encoder,
        predictor,
        planner,
        val_loader,
        val_sampler,
        token_ae,
        runtime_normalize_reps,
        rank,
        world_size,
    ) = build_runtime(raw_config, checkpoint_path, output_dir)

    logger = get_logger(__name__)
    logger.info("token_ae_enabled=%s runtime_normalize_reps=%s", token_ae is not None, runtime_normalize_reps)

    vis_output_dir = None if args.disable_vis else str(output_dir / "visualizations")
    metrics = run_validation(
        encoder=encoder,
        predictor=predictor,
        planner=planner,
        val_loader=val_loader,
        val_sampler=val_sampler,
        config=config,
        epoch=args.epoch,
        rank=rank,
        world_size=world_size,
        use_tubelet_repeat=config.data.use_tubelet_repeat,
        vis_output_dir=vis_output_dir,
        token_ae=token_ae,
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "config": str(_expand_path(args.config)),
        "output_dir": str(output_dir),
        "metrics": metrics,
    }
    metrics_path = output_dir / "eval_metrics.json"
    if rank == 0:
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(result), handle, ensure_ascii=False, indent=2, allow_nan=True)
        logger.info("Wrote metrics: %s", metrics_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline nuScenes/NavSim-PKL validation for a train_navsim_v2 world-model checkpoint.",
    )
    parser.add_argument("--config", "--fname", dest="config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    parser.add_argument("--output-dir", default=None, help="Directory for eval_metrics.json and visualizations")
    parser.add_argument("--disable-vis", action="store_true", help="Disable trajectory visualization output")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch label used for deterministic eval seeds and plots")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers")
    parser.add_argument("--max-val-scenes", type=int, default=None, help="Override data.navsim.max_val_scenes")
    parser.add_argument("--val-data-path", default=None, help="Override data.navsim.val_data_path")
    parser.add_argument("--val-sensor-blobs-path", default=None, help="Override data.navsim.val_sensor_blobs_path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
