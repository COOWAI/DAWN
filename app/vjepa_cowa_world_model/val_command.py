# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""
轨迹预测评估模块 (for train_command.py)
与 train_command.py 的 encoder/predictor/planner 调用方式保持一致

相比 val_giant.py 的修复:
1. prepare_status_feature 支持 use_states_for_planner / action_dim
2. predictor forward 支持 predictor_inference_consistent 模式
3. predictor forward 支持 use_states_for_predictor

评估指标:
- ADE (Average Displacement Error): 所有预测点与GT点的平均L2距离
- FDE (Final Displacement Error): 最后一个预测点与GT点的L2距离
- minADE@K / minFDE@K: 多模态轨迹的最小ADE/FDE
"""

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel  # noqa: F401

from app.vjepa_cowa_world_model.training.config import (
    is_drive_jepa_main_encoder_config,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.main_encoder_runtime import (
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    forward_main_context,
)
from app.vjepa_cowa_world_model.training.predictor_parallel import forward_parallel_predictor, use_parallel_predictor

# 使用重构后的公共模块
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    mask_future_actions,
    prepare_inference_consistent_states,
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_planner_use_drive_command,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import (
    extract_batch_metadata,
    make_validation_rng_seed,
    seed_eval_rng,
)
from app.vjepa_cowa_world_model.utils.metrics import (
    WORLD4DRIVE_REPORTED_SECONDS,
    compute_collision_rate,
    compute_world4drive_l2_metrics,
    populate_point_l2_horizons,
    populate_world4drive_collision_horizons,
    populate_world4drive_l2_horizons,
)
from app.vjepa_cowa_world_model.utils.planner_training import resolve_validation_timestep_sec
from app.vjepa_cowa_world_model.utils.visualization import visualize_multimodal_trajectory, visualize_trajectory
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_config_value(config, key, default=None):
    """
    兼容 dict 和 dataclass 两种配置类型的访问方式
    """
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


def _get_nested_config(config, *keys, default=None):
    """
    获取嵌套配置值，兼容 dict 和 dataclass
    """
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


# =====================================================================
#  Metric functions (unchanged from val_giant.py)
# =====================================================================


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    pred_xy = pred_traj[..., :2]
    gt_xy = gt_traj[..., :2]
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)
    return displacement.mean(dim=-1)


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    pred_xy_final = pred_traj[:, -1, :2]
    gt_xy_final = gt_traj[:, -1, :2]
    return torch.norm(pred_xy_final - gt_xy_final, dim=-1)


def compute_metrics(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    ade_per_sample = compute_ade(pred_traj, gt_traj)
    fde_per_sample = compute_fde(pred_traj, gt_traj)
    return {
        "ade": ade_per_sample.mean().item(),
        "fde": fde_per_sample.mean().item(),
        "ade_per_sample": ade_per_sample,
        "fde_per_sample": fde_per_sample,
    }


def compute_minade_minfde_k(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    pred_xy = pred_trajs[..., :2]
    gt_xy = gt_traj[:, None, :, :2]
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, K, num_poses]

    ade_k = displacement.mean(dim=-1)  # [B, K]
    fde_k = displacement[:, :, -1]  # [B, K]

    minade_per_sample = ade_k.min(dim=1).values  # [B]
    minfde_per_sample = fde_k.min(dim=1).values  # [B]

    return {
        "minade_k": minade_per_sample.mean().item(),
        "minfde_k": minfde_per_sample.mean().item(),
        "minade_per_sample": minade_per_sample,
        "minfde_per_sample": minfde_per_sample,
    }


# =====================================================================
#  Helper functions
# =====================================================================


def _prepare_encoder_input(context_clips: torch.Tensor) -> torch.Tensor:
    """
    与 train_command_v2.forward_context() 中的 encoder 输入构造保持完全一致。

    输入:  [B, C, T, H, W]
    输出:  [B*T, C, 2, H, W]
    """
    return context_clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)


def _compress_tokens_with_token_ae(token_ae, tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Apply frozen Token AE compression on concatenated frame tokens."""
    ae_tokens_per_frame = int(getattr(token_ae, "tokens_per_frame"))
    expected_tokens = int(num_frames) * ae_tokens_per_frame
    if tokens.size(1) != expected_tokens:
        if tokens.size(1) % ae_tokens_per_frame != 0:
            raise ValueError(
                "Cannot infer TokenAE frame count: "
                f"tokens={tokens.size(1)}, num_frames={num_frames}, ae_tokens_per_frame={ae_tokens_per_frame}"
            )
        num_frames = tokens.size(1) // ae_tokens_per_frame
    return token_ae.encode(tokens, num_frames=num_frames)


# =====================================================================
#  Core validation
# =====================================================================


@torch.no_grad()
def validate_one_epoch(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    device,
    dtype,
    mixed_precision,
    tubelet_size,
    tokens_per_frame,
    num_poses,
    num_time_steps,
    world_size,
    rank,
    epoch,
    normalize_reps: bool = True,
    status_mode: str = "first",
    z_ar_mode: str = "full",
    use_z_context: bool = False,
    use_tubelet_repeat: bool = False,
    # --- fix #1: planner status 参数 ---
    use_states_for_planner: bool = False,
    action_dim: int = 7,
    # --- fix #2: inference_consistent 参数 ---
    predictor_inference_consistent: bool = False,
    num_observed_frames: int = 2,
    # --- fix #3: predictor states 输入模式 ---
    use_states_for_predictor: bool = True,
    # --- fix #4: predictor 完全无辅助输入 (train_giant_first.py) ---
    predictor_no_aux_input: bool = False,
    # --- fix #5: 观测帧 tokens ---
    use_observed_tokens: bool = False,
    # --- fix #6: IC 状态维度 ---
    state_dim: int = 7,
    # --- fix #6b: planner status 维度（与 predictor state_dim 解耦；默认回落到 state_dim） ---
    planner_status_dim: int = 0,
    # --- fix #6c: predictor / planner use_drive_command 分离 ---
    predictor_use_drive_command: bool = True,
    planner_use_drive_command: bool = True,
    # --- fix #7: diffusion planner anchor_state ---
    planner_type: str = "transformer",
    # --- NuScenes L2 和碰撞指标参数 ---
    timestep_sec: float = 0.5,
    compute_collision: bool = True,
    # --- 可视化参数 ---
    vis_output_dir: str = None,
    vis_every_n_batches: int = 50,
    vis_samples_per_batch: int = 20,
    token_ae=None,
    config=None,
) -> dict:
    """
    执行一个完整 epoch 的验证，与 train_command.py / train_giant_first.py 的前向逻辑完全对齐。
    """
    encoder_unwrapped = encoder.module if hasattr(encoder, "module") else encoder
    predictor_unwrapped = predictor.module if hasattr(predictor, "module") else predictor
    planner_unwrapped = planner.module if hasattr(planner, "module") else planner

    encoder_was_training = encoder_unwrapped.training
    predictor_was_training = predictor_unwrapped.training
    planner_was_training = planner_unwrapped.training

    def _restore_validation_training_states():
        encoder_unwrapped.train(encoder_was_training)
        predictor_unwrapped.train(predictor_was_training)
        planner_unwrapped.train(planner_was_training)

    encoder_unwrapped.eval()
    predictor_unwrapped.eval()
    planner_unwrapped.eval()

    runtime_token_ae = token_ae
    runtime_normalize_reps = normalize_reps
    drive_jepa_main = config is not None and is_drive_jepa_main_encoder_config(config)

    total_ade = 0.0
    total_fde = 0.0
    total_minade_k = 0.0
    total_minfde_k = 0.0
    total_samples = 0
    failed_batches = 0

    # L2 per timestep 和碰撞率累加变量
    total_l2_per_step = None  # 延迟初始化为 list[float]
    total_collision_counts = None  # 延迟初始化为 np.ndarray (box collision)
    total_point_collision_counts = None  # 延迟初始化为 np.ndarray (point collision)
    total_gt_collision_counts = None  # 延迟初始化为 np.ndarray (GT 碰撞排除数)
    missing_bev_segmentation_warned = False

    val_sampler.set_epoch(epoch)

    for batch_idx, sample in enumerate(val_loader):
        try:
            metadata = extract_batch_metadata(sample)
            eval_rng_seed = make_validation_rng_seed(
                config,
                stage="navsim",
                epoch=epoch,
                batch_idx=batch_idx,
                metadata=metadata,
                rank=rank,
            )
            seed_eval_rng(eval_rng_seed, device)
            context_frames = sample[0].to(device, non_blocking=True)
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)

            # driving_command / ego_dynamics (NavSim 7-tuple collate)
            driving_command = sample[5].to(device, dtype=torch.float, non_blocking=True) if len(sample) > 5 else None
            ego_dynamics = sample[6].to(device, dtype=torch.float, non_blocking=True) if len(sample) > 6 else None

            # agent annotations (index 7, 8) — not directly used for collision;
            # kept in collate for backward compatibility but not extracted here.

            # Pre-computed BEV segmentation for new collision rate API (kept as numpy, CPU-side)
            bev_segmentation = sample[9].numpy() if len(sample) > 9 and sample[9] is not None else None
            if compute_collision and bev_segmentation is None and not missing_bev_segmentation_warned:
                logger.warning(
                    "Validation batches do not include BEV segmentation; " "collision metrics will be reported as inf."
                )
                missing_bev_segmentation_warned = True

            B = context_frames.shape[0]
            T = context_frames.shape[2]

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # ============ Encoder Forward ============
                parallel_predictor = config is not None and use_parallel_predictor(config)
                if drive_jepa_main:
                    z = forward_main_context(
                        encoder=encoder_unwrapped,
                        context_clips=context_frames,
                        config=config,
                        runtime_normalize_reps=runtime_normalize_reps,
                        token_ae=runtime_token_ae,
                    )
                    if parallel_predictor:
                        batch_timeline = build_parallel_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                        )
                    else:
                        batch_timeline = build_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                        )
                    predictor_actions = batch_timeline.actions
                    predictor_states = batch_timeline.states
                    predictor_extrinsics = batch_timeline.extrinsics
                    predictor_driving_command = batch_timeline.driving_command
                    predictor_ego_dynamics = batch_timeline.ego_dynamics
                    batch_tokens_per_frame = batch_timeline.tokens_per_frame
                    predictor_total_steps = batch_timeline.num_time_steps
                    predictor_observed_steps = batch_timeline.num_observed_steps
                    predictor_frame_stride = getattr(batch_timeline, "frame_stride", 1)
                else:
                    encoder_input = _prepare_encoder_input(context_frames)
                    z_context_out = encoder_unwrapped([encoder_input])
                    z = z_context_out[0]
                    z = z.view(B, T, -1, z.size(-1)).flatten(1, 2)

                    if runtime_token_ae is None and _get_nested_config(config, "token_ae", "enabled", default=False):
                        from app.vjepa_cowa_world_model.training.models import load_frozen_token_ae

                        runtime_token_ae, runtime_normalize_reps = load_frozen_token_ae(
                            config,
                            device=device,
                            encoder_embed_dim=z.size(-1),
                            tokens_per_frame=z.size(1) // T,
                            normalize_reps=runtime_normalize_reps,
                            dtype=dtype,
                        )

                    if runtime_token_ae is not None:
                        z = _compress_tokens_with_token_ae(runtime_token_ae, z, num_frames=T)

                    if runtime_normalize_reps:
                        z = F.layer_norm(z, (z.size(-1),))

                    predictor_actions = actions
                    if parallel_predictor and predictor_actions.shape[1] == T - 1:
                        predictor_actions = torch.cat(
                            [predictor_actions, predictor_actions.new_zeros(B, 1, predictor_actions.shape[-1])],
                            dim=1,
                        )
                    predictor_states = states
                    predictor_extrinsics = extrinsics
                    predictor_driving_command = driving_command
                    predictor_ego_dynamics = ego_dynamics
                    batch_tokens_per_frame = tokens_per_frame
                    predictor_total_steps = T
                    predictor_observed_steps = num_observed_frames
                    predictor_frame_stride = 1

                assert z.shape[1] == predictor_total_steps * batch_tokens_per_frame, (
                    f"z shape mismatch: {z.shape}, expected second dim to be "
                    f"{predictor_total_steps * batch_tokens_per_frame}"
                )
                expected_action_steps = predictor_total_steps if parallel_predictor else predictor_total_steps - 1
                assert predictor_actions.shape[1] == expected_action_steps, (
                    f"actions shape mismatch: {predictor_actions.shape}, expected second dim to be "
                    f"{expected_action_steps}"
                )
                assert predictor_states.shape[1] == predictor_total_steps, (
                    f"states shape mismatch: {predictor_states.shape}, expected second dim to be "
                    f"{predictor_total_steps}"
                )
                assert predictor_extrinsics.shape[1] == predictor_total_steps, (
                    f"extrinsics shape mismatch: {predictor_extrinsics.shape}, expected second dim to be "
                    f"{predictor_total_steps}"
                )

                # ============ Predictor Forward (与 train_command.py 对齐) ============
                if parallel_predictor:
                    parallel_output = forward_parallel_predictor(
                        predictor=predictor_unwrapped,
                        observed_tokens=z,
                        actions=predictor_actions,
                        states=predictor_states,
                        extrinsics=predictor_extrinsics,
                        config=config,
                        tokens_per_frame=batch_tokens_per_frame,
                        runtime_normalize_reps=runtime_normalize_reps,
                        num_observed_steps=predictor_observed_steps,
                        driving_command=predictor_driving_command,
                        ego_dynamics=predictor_ego_dynamics,
                        predictor_no_aux_input=predictor_no_aux_input,
                    )
                    z_ar = parallel_output.z_future
                else:
                    # 预计算 inference_consistent_states（如果启用）
                    if predictor_inference_consistent:
                        num_obs = predictor_observed_steps
                        num_known = num_obs - 1
                        _ic_states_full = prepare_inference_consistent_states(
                            predictor_states,
                            num_observed=num_obs,
                            driving_command=predictor_driving_command,
                            ego_dynamics=predictor_ego_dynamics,
                            state_dim=state_dim,
                            use_drive_command=predictor_use_drive_command,
                        )
                    else:
                        num_obs = 1
                        num_known = actions.shape[1]
                        _ic_states_full = None

                    def _step_predictor(_z, _a, _s, _e, _a_mask=None):
                        if predictor_no_aux_input:
                            _a_input = torch.zeros_like(_a)
                            _s_input = torch.zeros_like(_s)
                            _e_input = torch.zeros_like(_e)
                        elif predictor_inference_consistent:
                            T_needed = _s.shape[1]
                            _a_input = _a
                            _s_input = _ic_states_full[:, :T_needed]
                            _e_input = _e
                        elif use_states_for_predictor:
                            _a_input = _a
                            _s_input = _s
                            _e_input = _e
                        else:
                            _a_input = _a
                            _s_input = torch.zeros_like(_s)
                            _e_input = _e
                        _z_out = predictor_unwrapped(_z, _a_input, _s_input, _e_input, action_mask=_a_mask)
                        if runtime_normalize_reps:
                            _z_out = F.layer_norm(_z_out, (_z_out.size(-1),))
                        return _z_out

                    # Teacher forcing
                    _z_enc = z[:, :-batch_tokens_per_frame]
                    _s, _e = predictor_states[:, :-1], predictor_extrinsics[:, :-1]
                    if predictor_inference_consistent:
                        _a, _a_mask = mask_future_actions(predictor_actions, num_known)
                    else:
                        _a = predictor_actions
                        _a_mask = None
                    z_tf = _step_predictor(_z_enc, _a, _s, _e, _a_mask=_a_mask)

                    # Autoregressive rollout
                    num_total = z.size(1) // batch_tokens_per_frame

                    if predictor_inference_consistent:
                        _z = z[:, : num_obs * batch_tokens_per_frame]
                        start_step = num_obs
                    else:
                        _z = torch.cat([z[:, :batch_tokens_per_frame], z_tf[:, :batch_tokens_per_frame]], dim=1)
                        start_step = 2

                    for k in range(start_step, num_total):
                        if k == num_total - 1:
                            _a_full = predictor_actions
                            _s_k = predictor_states[:, :-1]
                            _e_k = predictor_extrinsics[:, :-1]
                        else:
                            _a_full = predictor_actions[:, :k]
                            _s_k = predictor_states[:, :k]
                            _e_k = predictor_extrinsics[:, :k]

                        if predictor_inference_consistent:
                            _a_k, _a_mask = mask_future_actions(_a_full, num_known)
                        else:
                            _a_k = _a_full
                            _a_mask = None

                        _z_nxt = _step_predictor(_z, _a_k, _s_k, _e_k, _a_mask=_a_mask)[:, -batch_tokens_per_frame:]
                        _z = torch.cat([_z, _z_nxt], dim=1)

                    if predictor_inference_consistent:
                        z_ar = _z[:, num_obs * batch_tokens_per_frame :]
                    else:
                        z_ar = _z[:, batch_tokens_per_frame:]

                z_ar_planner = z_ar if z_ar_mode == "full" else z_ar[:, :batch_tokens_per_frame]

                # ============ Planner Forward ============
                if predictor_inference_consistent:
                    _planner_sd = planner_status_dim if planner_status_dim > 0 else state_dim
                    status_feature = prepare_inference_consistent_status_vector(
                        states,
                        num_observed=num_observed_frames,
                        driving_command=driving_command,
                        ego_dynamics=ego_dynamics,
                        state_dim=_planner_sd,
                        use_drive_command=planner_use_drive_command,
                    )
                else:
                    status_feature = prepare_status_feature(
                        states,
                        actions,
                        status_mode=status_mode,
                        use_states_for_planner=use_states_for_planner,
                        action_dim=action_dim,
                        driving_command=driving_command,
                        ego_dynamics=ego_dynamics,
                    )
                z_first_frame = z[:, :batch_tokens_per_frame] if use_z_context else None
                # 观测帧 tokens
                if use_observed_tokens:
                    z_observed = z[:, : predictor_observed_steps * batch_tokens_per_frame]
                else:
                    z_observed = None
                planner_action_history = None
                if _get_nested_config(config, "planner", "use_action_history_for_planner", default=False):
                    planner_action_history = build_observed_action_trajectory_history(
                        predictor_actions,
                        num_observed_frames=predictor_observed_steps,
                        action_history_dim=int(_get_nested_config(config, "planner", "action_history_dim", default=3)),
                        dt=timestep_sec * max(int(predictor_frame_stride), 1),
                    )
                if planner_type == "diffusion":
                    # 构造 anchor_state，与 train_navsim_v2.py 对齐
                    _anchor_state = None
                    if hasattr(planner_unwrapped, "use_anchor_frame") and planner_unwrapped.use_anchor_frame:
                        _future_start = num_observed_frames if predictor_inference_consistent else 1
                        _origin_idx = _future_start - 1
                        # origin 帧即坐标系原点: x=0, y=0, yaw=0 → cos=1, sin=0
                        if planner_unwrapped.traj_dim == 4:
                            _anchor_state = torch.stack(
                                [
                                    torch.zeros(B, device=device),  # x = 0
                                    torch.zeros(B, device=device),  # y = 0
                                    torch.ones(B, device=device),  # cos(0) = 1
                                    torch.zeros(B, device=device),  # sin(0) = 0
                                ],
                                dim=-1,
                            ).float()  # [B, 4]
                        else:
                            # vx, vy 从 ego_dynamics 取该帧的真实速度
                            if ego_dynamics is not None:
                                _a_vx = ego_dynamics[:, _origin_idx, 0].float()
                                _a_vy = ego_dynamics[:, _origin_idx, 1].float()
                            else:
                                _a_vx = torch.zeros(B, device=device)
                                _a_vy = torch.zeros(B, device=device)
                            _anchor_state = torch.stack(
                                [
                                    torch.zeros(B, device=device),  # x = 0
                                    torch.zeros(B, device=device),  # y = 0
                                    _a_vx,  # vx
                                    _a_vy,  # vy
                                    torch.ones(B, device=device),  # cos(0) = 1
                                    torch.zeros(B, device=device),  # sin(0) = 0
                                ],
                                dim=-1,
                            ).float()  # [B, 6]
                    planner_output = planner_unwrapped(
                        z_ar_planner,
                        status_feature,
                        z_context=z_first_frame,
                        z_observed=z_observed,
                        action_history=planner_action_history,
                        anchor_state=_anchor_state,
                    )
                else:
                    planner_output = planner_unwrapped(
                        z_ar_planner,
                        status_feature,
                        z_context=z_first_frame,
                        z_observed=z_observed,
                        action_history=planner_action_history,
                    )

                pred_trajs = None
                if "trajectories" in planner_output:
                    pred_trajs = planner_output["trajectories"]  # [B, K, num_poses, 3]
                    pred_conf = planner_output["confidences"]  # [B, K]
                    best_idx = pred_conf.argmax(dim=1)
                    best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(-1, 1, pred_trajs.shape[2], pred_trajs.shape[3])
                    traj_output = pred_trajs.gather(1, best_idx_exp).squeeze(1)
                else:
                    traj_output = planner_output["trajectory"]

                # Cast to float32 to avoid BFloat16 errors in metrics / numpy
                traj_output = traj_output.float()
                if pred_trajs is not None:
                    pred_trajs = pred_trajs.float()

                # ============ GT 轨迹转换 ============
                # Use float64 for position diff to guard against large UTM values.
                StateSE2_indices = [0, 1, 5]
                states_se2 = states[:, :, StateSE2_indices].double()

                future_start_idx = num_observed_frames if predictor_inference_consistent else 1
                origin_idx = future_start_idx - 1
                origin_x = states_se2[:, origin_idx, 0]
                origin_y = states_se2[:, origin_idx, 1]
                origin_yaw = states_se2[:, origin_idx, 2]

                dx = states_se2[:, future_start_idx:, 0] - origin_x[:, None]
                dy = states_se2[:, future_start_idx:, 1] - origin_y[:, None]
                dyaw = states_se2[:, future_start_idx:, 2] - origin_yaw[:, None]

                cos_h = torch.cos(-origin_yaw)
                sin_h = torch.sin(-origin_yaw)
                ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
                gt_trajectory = gt_trajectory[:, :num_poses]

                # ============ 计算指标 ============
                metrics = compute_metrics(traj_output, gt_trajectory)
                batch_ade = metrics["ade"]
                batch_fde = metrics["fde"]
                if pred_trajs is not None:
                    min_metrics = compute_minade_minfde_k(pred_trajs, gt_trajectory)
                    batch_minade_k = min_metrics["minade_k"]
                    batch_minfde_k = min_metrics["minfde_k"]
                else:
                    batch_minade_k = batch_ade
                    batch_minfde_k = batch_fde

                # ============ L2 per timestep ============
                l2_metrics = compute_world4drive_l2_metrics(
                    traj_output,
                    gt_trajectory,
                    timestep_sec=timestep_sec,
                )

                # ============ Collision rate ============
                if compute_collision and bev_segmentation is not None:
                    # 切出 future-only 的 BEV seg maps: [B, T_future, H, W]
                    bev_seg_future = bev_segmentation[:, future_start_idx:, :, :]
                    # 确保时间步对齐
                    T_future_traj = traj_output.shape[1]
                    bev_seg_future = bev_seg_future[:, :T_future_traj]
                    collision_metrics = compute_collision_rate(
                        pred_traj=traj_output.cpu().numpy(),
                        gt_traj=gt_trajectory.cpu().numpy(),
                        segmentation=bev_seg_future,
                        ego_poses=states.cpu().numpy(),
                        future_start_idx=future_start_idx,
                        timestep_sec=timestep_sec,
                        reference_frame_idx=origin_idx,
                    )
                else:
                    collision_metrics = None

                total_ade += batch_ade * B
                total_fde += batch_fde * B
                total_minade_k += batch_minade_k * B
                total_minfde_k += batch_minfde_k * B
                total_samples += B

                # L2: 累加每步 L2 × batch_size
                if total_l2_per_step is None:
                    total_l2_per_step = [v * B for v in l2_metrics["l2_per_step"]]
                else:
                    for i, v in enumerate(l2_metrics["l2_per_step"]):
                        if i < len(total_l2_per_step):
                            total_l2_per_step[i] += v * B

                # 碰撞: 累加 raw counts (box, point, gt_exclusion)
                if collision_metrics is not None and "collision_counts" in collision_metrics:
                    cc = np.array(collision_metrics["collision_counts"], dtype=np.int64)
                    pc = np.array(collision_metrics["point_collision_counts"], dtype=np.int64)
                    gc = np.array(collision_metrics["gt_collision_counts"], dtype=np.int64)
                    if total_collision_counts is None:
                        total_collision_counts = cc
                        total_point_collision_counts = pc
                        total_gt_collision_counts = gc
                    else:
                        total_collision_counts += cc
                        total_point_collision_counts += pc
                        total_gt_collision_counts += gc

            if batch_idx % 50 == 0:
                logger.info(
                    f"Validation Epoch {epoch}, Batch {batch_idx}/{len(val_loader)}, "
                    f"ADE: {batch_ade:.4f}, FDE: {batch_fde:.4f}, "
                    f"minADE@K: {batch_minade_k:.4f}, minFDE@K: {batch_minfde_k:.4f}"
                )

            # Visualization: sample every N batches, rank 0 only
            if vis_output_dir and rank == 0 and batch_idx % vis_every_n_batches == 0:
                if pred_trajs is not None:
                    visualize_multimodal_trajectory(
                        pred_trajs=pred_trajs,
                        pred_conf=pred_conf,
                        gt_traj=gt_trajectory,
                        output_dir=vis_output_dir,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        limit=vis_samples_per_batch,
                    )
                else:
                    visualize_trajectory(
                        pred_traj=traj_output,
                        gt_traj=gt_trajectory,
                        output_dir=vis_output_dir,
                        epoch=epoch,
                        itr=batch_idx,
                        limit=vis_samples_per_batch,
                    )

        except Exception as e:
            failed_batches += 1
            logger.warning(f"Validation batch {batch_idx} failed: {e}")
            continue

    if total_samples == 0:
        _restore_validation_training_states()
        raise RuntimeError(
            f"Validation produced zero successful samples (failed_batches={failed_batches}). "
            "Please check model interfaces and validation data pipeline."
        )

    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    avg_minade_k = total_minade_k / total_samples
    avg_minfde_k = total_minfde_k / total_samples

    if world_size > 1:
        metrics_tensor = torch.tensor(
            [total_ade, total_fde, total_minade_k, total_minfde_k, total_samples], dtype=torch.float64, device=device
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_samples_all = metrics_tensor[4].item()
        total_samples = int(total_samples_all)  # 更新为全局总数，供后续碰撞率分母使用
        if total_samples_all > 0:
            avg_ade = metrics_tensor[0].item() / total_samples_all
            avg_fde = metrics_tensor[1].item() / total_samples_all
            avg_minade_k = metrics_tensor[2].item() / total_samples_all
            avg_minfde_k = metrics_tensor[3].item() / total_samples_all

        # L2 per step: all_reduce (只 reduce L2 累加和，分母用已 reduce 的 total_samples_all)
        if total_l2_per_step is not None:
            l2_tensor = torch.tensor(total_l2_per_step, dtype=torch.float64, device=device)
            dist.all_reduce(l2_tensor, op=dist.ReduceOp.SUM)
            if total_samples_all > 0:
                total_l2_per_step = [v / total_samples_all for v in l2_tensor.tolist()]
        else:
            pass  # single-process: total_l2_per_step already averaged locally

        # Collision: all_reduce counts (box + point + gt_exclusion)
        if total_collision_counts is not None:
            col_tensor = torch.tensor(
                list(total_collision_counts) + list(total_point_collision_counts) + list(total_gt_collision_counts),
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(col_tensor, op=dist.ReduceOp.SUM)
            n_steps = len(total_collision_counts)
            total_collision_counts = col_tensor[:n_steps].cpu().numpy().astype(np.int64)
            total_point_collision_counts = col_tensor[n_steps : 2 * n_steps].cpu().numpy().astype(np.int64)
            total_gt_collision_counts = col_tensor[2 * n_steps :].cpu().numpy().astype(np.int64)
    else:
        if total_l2_per_step is not None and total_samples > 0:
            total_l2_per_step = [v / total_samples for v in total_l2_per_step]

    # 计算最终 L2 per-timestep 指标
    result = {
        "ade": avg_ade,
        "fde": avg_fde,
        "minade_k": avg_minade_k,
        "minfde_k": avg_minfde_k,
    }

    if total_l2_per_step is not None:
        l2_per_step = np.asarray(total_l2_per_step, dtype=np.float64)
        result["l2_per_step"] = l2_per_step.tolist()
        populate_world4drive_l2_horizons(result, l2_per_step, timestep_sec)
        populate_point_l2_horizons(result, l2_per_step, timestep_sec)

    if total_collision_counts is not None:
        box_collision_per_step = total_collision_counts / max(float(total_samples), 1.0)
        point_collision_per_step = total_point_collision_counts / max(float(total_samples), 1.0)
        result["collision_per_step"] = box_collision_per_step.tolist()
        result["point_collision_per_step"] = point_collision_per_step.tolist()
        result["collision_counts"] = total_collision_counts.tolist()
        result["point_collision_counts"] = total_point_collision_counts.tolist()
        result["gt_collision_counts"] = total_gt_collision_counts.tolist()
        populate_world4drive_collision_horizons(
            result,
            total_collision_counts,
            total_samples=total_samples,
            timestep_sec=timestep_sec,
            metric_prefix="collision",
            avg_key="collision_rate",
        )
        populate_world4drive_collision_horizons(
            result,
            total_point_collision_counts,
            total_samples=total_samples,
            timestep_sec=timestep_sec,
            metric_prefix="point_collision",
            avg_key="point_collision_rate",
        )
    elif compute_collision:
        result["collision_rate"] = float("inf")
        result["point_collision_rate"] = float("inf")

    _restore_validation_training_states()

    return result


# =====================================================================
#  Entry point
# =====================================================================


def run_validation(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    config: dict,
    epoch: int,
    rank: int,
    world_size: int,
    use_tubelet_repeat: bool = False,
    vis_output_dir: str = None,
    token_ae=None,
) -> dict:
    """
    运行验证的入口函数 (for train_command.py)

    从 config 中自动读取所有与 train_command.py 对齐所需的参数，
    包括 predictor_inference_consistent 等。
    """
    # 设备
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # 数据类型 - 兼容 dict 和 dataclass
    which_dtype = _get_nested_config(config, "meta", "dtype", default="float32")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # 数据配置 - 兼容 dict 和 dataclass
    tubelet_size = _get_nested_config(config, "data", "tubelet_size", default=2)
    target_frame = _get_nested_config(config, "data", "num_target_frames", default=16)
    tokens_per_frame = resolve_main_encoder_tokens_per_frame(config, encoder)

    # Loss - 兼容 dict 和 dataclass
    normalize_reps = _get_nested_config(config, "loss", "normalize_reps", default=True)

    # Planner - 兼容 dict 和 dataclass
    status_mode = _get_nested_config(config, "planner", "states_mode", default="first")
    z_ar_mode = _get_nested_config(config, "planner", "z_ar_mode", default="full")
    use_z_context = _get_nested_config(config, "planner", "use_z_context", default=False)
    use_states_for_planner = _get_nested_config(config, "planner", "use_states_for_planner", default=True)
    use_observed_tokens = _get_nested_config(config, "planner", "use_observed_tokens", default=False)
    planner_type = _get_nested_config(config, "planner", "planner_type", default="transformer")
    fps = _get_nested_config(config, "data", "fps", default=None)
    diff_dt = _get_nested_config(config, "planner", "diff_dt", default=None)
    timestep_sec = resolve_validation_timestep_sec(fps=fps, diff_dt=diff_dt, default=0.5)
    assert z_ar_mode in ("full", "first_step"), f"Invalid planner.z_ar_mode={z_ar_mode}"
    if use_observed_tokens and z_ar_mode != "full":
        raise ValueError("planner.use_observed_tokens=True requires planner.z_ar_mode='full' in validation")
    if fps is not None and fps > 0 and diff_dt is not None and diff_dt > 0:
        data_timestep_sec = 1.0 / float(fps)
        if abs(data_timestep_sec - float(diff_dt)) > 1e-6:
            logger.warning(
                "Validation timestep mismatch: data.fps=%.4f -> %.4fs, planner.diff_dt=%.4fs. "
                "Using %.4fs for metrics.",
                float(fps),
                data_timestep_sec,
                float(diff_dt),
                timestep_sec,
            )

    # Train 配置 — predictor 输入模式 - 兼容 dict 和 dataclass
    use_states_for_predictor = _get_nested_config(config, "train", "use_states_for_predictor", default=True)
    predictor_no_aux_input = _get_nested_config(config, "train", "predictor_no_aux_input", default=False)
    if predictor_no_aux_input:
        use_states_for_predictor = False
    action_dim = _get_nested_config(config, "train", "action_dim", default=7)
    predictor_inference_consistent = _get_nested_config(
        config, "train", "predictor_inference_consistent", default=False
    )
    num_observed_frames = _get_nested_config(config, "train", "num_observed_frames", default=2)
    state_dim = _get_nested_config(config, "train", "state_dim", default=7)
    predictor_use_drive_command = _get_nested_config(config, "train", "use_drive_command", default=True)
    planner_use_drive_command = resolve_planner_use_drive_command(config)
    planner_status_dim_cfg = _get_nested_config(config, "planner", "status_dim", default=0)
    if predictor_inference_consistent:
        num_poses = target_frame - num_observed_frames
    else:
        num_poses = target_frame - 1
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_poses, encoder=encoder)
    if predictor_no_aux_input and predictor_inference_consistent:
        logger.warning(
            "predictor_no_aux_input=True and predictor_inference_consistent=True "
            "are mutually exclusive; disabling predictor_inference_consistent"
        )
        predictor_inference_consistent = False

    logger.info(f"Starting validation for epoch {epoch}...")
    logger.info(
        f"Validation config: tokens_per_frame={tokens_per_frame}, num_poses={num_poses}, "
        f"num_time_steps={num_time_steps}, normalize_reps={normalize_reps}, "
        f"status_mode={status_mode}, z_ar_mode={z_ar_mode}, use_z_context={use_z_context}, "
        f"use_tubelet_repeat={use_tubelet_repeat}, "
        f"predictor_no_aux_input={predictor_no_aux_input}, "
        f"use_states_for_predictor={use_states_for_predictor}, "
        f"use_states_for_planner={use_states_for_planner}, "
        f"predictor_inference_consistent={predictor_inference_consistent}, "
        f"num_observed_frames={num_observed_frames}, action_dim={action_dim}, "
        f"predictor_use_drive_command={predictor_use_drive_command}, "
        f"planner_use_drive_command={planner_use_drive_command}, "
        f"use_observed_tokens={use_observed_tokens}, planner_type={planner_type}, "
        f"timestep_sec={timestep_sec:.4f}"
    )

    metrics = validate_one_epoch(
        encoder=encoder,
        predictor=predictor,
        planner=planner,
        val_loader=val_loader,
        val_sampler=val_sampler,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        tubelet_size=tubelet_size,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=num_time_steps,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        normalize_reps=normalize_reps,
        status_mode=status_mode,
        z_ar_mode=z_ar_mode,
        use_z_context=use_z_context,
        use_tubelet_repeat=use_tubelet_repeat,
        # fix #1
        use_states_for_planner=use_states_for_planner,
        action_dim=action_dim,
        # fix #2
        predictor_inference_consistent=predictor_inference_consistent,
        num_observed_frames=num_observed_frames,
        # fix #3
        use_states_for_predictor=use_states_for_predictor,
        # fix #4
        predictor_no_aux_input=predictor_no_aux_input,
        # fix #5: 观测帧 tokens
        use_observed_tokens=use_observed_tokens,
        state_dim=state_dim,
        planner_status_dim=planner_status_dim_cfg,
        predictor_use_drive_command=predictor_use_drive_command,
        planner_use_drive_command=planner_use_drive_command,
        # fix #7: diffusion planner type
        planner_type=planner_type,
        timestep_sec=timestep_sec,
        # visualization
        vis_output_dir=vis_output_dir,
        token_ae=token_ae,
        config=config,
    )

    if rank == 0:
        logger.info("=" * 50)
        logger.info(f"Validation Results - Epoch {epoch}:")
        logger.info(f"  ADE (Average Displacement Error): {metrics['ade']:.4f} m")
        logger.info(f"  FDE (Final Displacement Error):    {metrics['fde']:.4f} m")
        logger.info(f"  minADE@K:                         {metrics['minade_k']:.4f} m")
        logger.info(f"  minFDE@K:                         {metrics['minfde_k']:.4f} m")
        # L2 per timestep
        if "l2_avg" in metrics:
            logger.info(f"  avg L2: {metrics['l2_avg']:.4f} m")
            for sec in WORLD4DRIVE_REPORTED_SECONDS:
                key = f"l2_at_{sec}s"
                if key in metrics:
                    logger.info(f"  L2@{sec}s: {metrics[key]:.4f} m")
            if "l2_point_avg" in metrics:
                logger.info(f"  point L2 avg: {metrics['l2_point_avg']:.4f} m")
                for sec in WORLD4DRIVE_REPORTED_SECONDS:
                    key = f"l2_point_at_{sec}s"
                    if key in metrics:
                        logger.info(f"  PointL2@{sec}s: {metrics[key]:.4f} m")
        # Collision rate
        if "collision_rate" in metrics:
            logger.info(f"  Collision Rate: {metrics['collision_rate']:.4f}")
            for sec in WORLD4DRIVE_REPORTED_SECONDS:
                key = f"collision_at_{sec}s"
                if key in metrics:
                    logger.info(f"  Collision@{sec}s: {metrics[key]:.4f}")
        logger.info("=" * 50)

    return metrics
