#!/usr/bin/env python3
"""
将 nuScenes 数据集（labels JSON + GT box NPZ + CAN bus）转换为 NavSim PKL 格式，
以便复用 NavSimWorldModelDataset 训练流水线。

用法:
    python scripts/convert_nuscenes_to_navsim_pkl.py \
        --nuscenes-root /path/nuScenes \
        --output-root /path/nuScenes/navsim_format_fix_2 \
        --split trainval \
        --workers 8

输出目录结构:
    {output_root}/
        train/
            scene-0001.pkl
            scene-0002.pkl
            ...
        val/
            scene-0003.pkl
            ...
        sensor_blobs/
            scene-0001/
                CAM_F0/
                    n015-2018-...jpg -> (symlink to original)
            ...

说明:
- 每个 PKL 文件是 List[Dict]，每个 Dict 代表一个 keyframe (2Hz)
- 相机图像通过 symlink 链接到原始 nuScenes 文件，不做复制
- Ego dynamics 从 CAN bus pose 数据中插值获取（速度/加速度）
- Driving command 从轨迹航向变化推断
- Agent annotations 从 GT box NPZ 转换
"""

import argparse
import json
import os
import pickle
import sys
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# =====================================================================
#  nuScenes 官方 train/val 场景划分（v1.0-trainval, 850 scenes）
#  来源: https://github.com/nutonomy/nuscenes-devkit
#  val 共 150 scenes，其余 700 为 train
# =====================================================================

NUSCENES_VAL_SCENES = {
    "scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015",
    "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036",
    "scene-0038", "scene-0039", "scene-0092", "scene-0093", "scene-0094",
    "scene-0095", "scene-0096", "scene-0097", "scene-0098", "scene-0099",
    "scene-0100", "scene-0101", "scene-0102", "scene-0103", "scene-0104",
    "scene-0105", "scene-0106", "scene-0107", "scene-0108", "scene-0109",
    "scene-0110", "scene-0221", "scene-0268", "scene-0269", "scene-0270",
    "scene-0271", "scene-0272", "scene-0273", "scene-0274", "scene-0275",
    "scene-0276", "scene-0277", "scene-0278", "scene-0329", "scene-0330",
    "scene-0331", "scene-0332", "scene-0344", "scene-0345", "scene-0346",
    "scene-0347", "scene-0348", "scene-0349", "scene-0350", "scene-0351",
    "scene-0352", "scene-0353", "scene-0354", "scene-0355", "scene-0356",
    "scene-0357", "scene-0358", "scene-0359", "scene-0360", "scene-0361",
    "scene-0362", "scene-0363", "scene-0364", "scene-0365", "scene-0366",
    "scene-0367", "scene-0368", "scene-0369", "scene-0370", "scene-0371",
    "scene-0372", "scene-0373", "scene-0374", "scene-0375", "scene-0376",
    "scene-0377", "scene-0378", "scene-0381", "scene-0382", "scene-0383",
    "scene-0384", "scene-0385", "scene-0386", "scene-0388", "scene-0389",
    "scene-0390", "scene-0391", "scene-0392", "scene-0393", "scene-0394",
    "scene-0395", "scene-0396", "scene-0397", "scene-0398", "scene-0399",
    "scene-0400", "scene-0401", "scene-0402", "scene-0403", "scene-0405",
    "scene-0406", "scene-0407", "scene-0408", "scene-0410", "scene-0411",
    "scene-0412", "scene-0413", "scene-0414", "scene-0415", "scene-0416",
    "scene-0417", "scene-0418", "scene-0419", "scene-0420", "scene-0421",
    "scene-0422", "scene-0423", "scene-0424", "scene-0425", "scene-0426",
    "scene-0427", "scene-0428", "scene-0429", "scene-0430", "scene-0431",
    "scene-0432", "scene-0433", "scene-0434", "scene-0435", "scene-0436",
    "scene-0437", "scene-0438", "scene-0439", "scene-0440", "scene-0441",
    "scene-0442", "scene-0443", "scene-0444", "scene-0445", "scene-0446",
    "scene-0447", "scene-0448", "scene-0449",
}

# =====================================================================
#  nuScenes GT box category ID → NavSim gt_names 映射
#  col 7 of gt_box NPZ: 0-4=vehicle variants, 5=pedestrian, 6-7=bicycle/motorcycle, 8=skip
# =====================================================================

CATEGORY_MAP = {
    0: "vehicle",     # car
    1: "vehicle",     # truck
    2: "vehicle",     # construction_vehicle
    3: "vehicle",     # bus
    4: "vehicle",     # trailer
    5: "pedestrian",  # pedestrian
    6: "bicycle",     # motorcycle
    7: "bicycle",     # bicycle
    8: None,          # barrier / traffic_cone / other — skip
}

# nuScenes 相机顺序 → NavSim 相机名
# images[1] = CAM_FRONT → NavSim "CAM_F0"
NUSCENES_CAM_FRONT_INDEX = 1
NAVSIM_CAMERA_NAME = "CAM_F0"

# Driving command 阈值
DRIVING_CMD_YAW_THRESHOLD_DEG = 15.0       # 累积 yaw 变化超过此值视为 TURN
DRIVING_CMD_UTURN_THRESHOLD_DEG = 150.0     # 累积 yaw 变化超过此值视为 U_TURN


def _rotation_matrix_to_quaternion_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → quaternion [w, x, y, z]"""
    rot = Rotation.from_matrix(rot_matrix)
    q_xyzw = rot.as_quat()  # scipy 返回 [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _yaw_from_rotation_matrix(rot_matrix: np.ndarray) -> float:
    """从 3x3 旋转矩阵提取 yaw（绕 Z 轴旋转角度）"""
    return float(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))


def _interpolate_canbus_at_timestamp(
    canbus_entries: List[Dict[str, Any]],
    canbus_utimes: np.ndarray,
    target_utime: int,
) -> Optional[Dict[str, Any]]:
    """在 CAN bus 时间序列中找到最近时间戳的条目。

    Parameters
    ----------
    canbus_entries : CAN bus JSON 条目列表（已按 utime 排序）
    canbus_utimes  : 预提取的 utime 数组
    target_utime   : 目标时间戳（微秒）

    Returns
    -------
    最近的 CAN bus 条目，如果距离 > 0.1s 则返回 None
    """
    idx = bisect_left(canbus_utimes, target_utime)

    # 检查边界
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(canbus_utimes):
        candidates.append(idx)

    if not candidates:
        return None

    best = min(candidates, key=lambda i: abs(canbus_utimes[i] - target_utime))
    dt_sec = abs(canbus_utimes[best] - target_utime) / 1e6

    if dt_sec > 0.1:  # 超过 100ms 视为无匹配
        return None

    return canbus_entries[best]


def _build_ego_dynamic_state(
    canbus_pose: Optional[Dict[str, Any]],
    pose_matrix: np.ndarray,
) -> np.ndarray:
    """构建 ego_dynamic_state [vx_ego, vy_ego, ax_ego, ay_ego]

    如果有 CAN bus pose 数据，直接使用其 ego-frame vel/accel 前两维。
    否则返回全零。

    Parameters
    ----------
    canbus_pose  : CAN bus pose 条目 (vel, accel already in ego coords)
    pose_matrix  : 4x4 ego2global 变换矩阵（保留参数以兼容调用方）

    Returns
    -------
    [vx_ego, vy_ego, ax_ego, ay_ego] float32
    """
    if canbus_pose is None:
        return np.zeros(4, dtype=np.float32)

    del pose_matrix

    vel_ego = np.array(canbus_pose["vel"], dtype=np.float64)[:2]
    accel_ego = np.array(canbus_pose["accel"], dtype=np.float64)[:2]

    return np.array([vel_ego[0], vel_ego[1], accel_ego[0], accel_ego[1]], dtype=np.float32)


def _compute_driving_command(
    yaw_sequence: np.ndarray,
    yaw_threshold_deg: float = DRIVING_CMD_YAW_THRESHOLD_DEG,
    uturn_threshold_deg: float = DRIVING_CMD_UTURN_THRESHOLD_DEG,
) -> np.ndarray:
    """从 yaw 序列推断 driving command。

    使用累积 yaw 变化判断：
    - |cumulative yaw change| < threshold → GO_STRAIGHT
    - threshold <= ... < uturn_threshold → TURN_LEFT / TURN_RIGHT (取决于符号)
    - >= uturn_threshold → U_TURN

    Parameters
    ----------
    yaw_sequence : [T] yaw 值序列
    yaw_threshold_deg : 转弯阈值（度）
    uturn_threshold_deg : 掉头阈值（度）

    Returns
    -------
    [T, 4] one-hot driving command: [GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, U_TURN]
    """
    T = len(yaw_sequence)
    commands = np.zeros((T, 4), dtype=np.float32)

    # 计算累积 yaw 变化（从第一帧开始）
    yaw_diff = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        dy = yaw_sequence[i] - yaw_sequence[0]
        yaw_diff[i] = np.arctan2(np.sin(dy), np.cos(dy))

    # 使用 future 的总 yaw 变化来决定命令
    total_yaw_change = yaw_diff[-1] if T > 1 else 0.0
    total_yaw_deg = np.degrees(abs(total_yaw_change))

    if total_yaw_deg >= uturn_threshold_deg:
        cmd_idx = 3  # U_TURN
    elif total_yaw_deg >= yaw_threshold_deg:
        cmd_idx = 1 if total_yaw_change > 0 else 2  # LEFT or RIGHT
    else:
        cmd_idx = 0  # GO_STRAIGHT

    # 所有帧使用相同的 command（整段轨迹的意图）
    commands[:, cmd_idx] = 1.0
    return commands


def _convert_annotations(
    npz_path: str,
) -> Optional[Dict[str, Any]]:
    """将 nuScenes GT box NPZ 转换为 NavSim anns dict。

    Parameters
    ----------
    npz_path : GT box NPZ 文件路径

    Returns
    -------
    NavSim 格式的 anns dict，或 None（如果文件不存在或无有效数据）
    """
    if not os.path.exists(npz_path):
        return None

    gt_data = np.load(npz_path)["gt_box"]  # [N, 9]

    if gt_data.shape[0] == 0:
        return {
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": [],
        }

    # 过滤掉 category == 8 (barrier / traffic_cone)
    cat_ids = gt_data[:, 7].astype(int)
    keep_mask = np.array([CATEGORY_MAP.get(c) is not None for c in cat_ids])

    if not keep_mask.any():
        return {
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": [],
        }

    kept = gt_data[keep_mask]
    # NPZ gt_box columns: [x, y, length, width, z, height, yaw, category, ...]
    # NavSim expects:     [x, y, z, length, width, height, heading]
    raw = kept[:, :7].astype(np.float32)
    gt_boxes = np.zeros_like(raw)
    gt_boxes[:, 0] = raw[:, 0]  # x
    gt_boxes[:, 1] = raw[:, 1]  # y
    gt_boxes[:, 2] = raw[:, 4]  # z
    gt_boxes[:, 3] = raw[:, 2]  # length
    gt_boxes[:, 4] = raw[:, 3]  # width
    gt_boxes[:, 5] = raw[:, 5]  # height
    gt_boxes[:, 6] = raw[:, 6]  # yaw / heading
    gt_names = [CATEGORY_MAP[int(c)] for c in kept[:, 7].astype(int)]
    n_kept = len(gt_names)

    return {
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "gt_velocity_3d": np.zeros((n_kept, 3), dtype=np.float32),
        "instance_tokens": [f"inst_{i}" for i in range(n_kept)],
        "track_tokens": [f"track_{i}" for i in range(n_kept)],
    }


def _convert_single_scene(
    scene_name: str,
    nuscenes_root: str,
    output_pkl_path: str,
    sensor_blobs_scene_dir: str,
) -> Tuple[str, bool, str]:
    """转换单个 nuScenes 场景为 NavSim PKL 格式。

    Parameters
    ----------
    scene_name            : 场景名 (e.g. "scene-0001")
    nuscenes_root         : nuScenes 根目录
    output_pkl_path       : 输出 PKL 文件路径
    sensor_blobs_scene_dir : 输出 sensor_blobs/{scene_name}/CAM_F0/ 目录

    Returns
    -------
    (scene_name, success, message)
    """
    try:
        labels_dir = os.path.join(nuscenes_root, "labels")
        canbus_dir = os.path.join(nuscenes_root, "can_bus")

        # 1. 加载场景 JSON
        json_path = os.path.join(labels_dir, f"{scene_name}.json")
        if not os.path.exists(json_path):
            return (scene_name, False, f"JSON not found: {json_path}")

        with open(json_path, "r") as f:
            raw_frames = json.load(f)

        if not raw_frames:
            return (scene_name, False, "Empty frames list")

        # 2. 加载 CAN bus pose 数据（用于 ego dynamics）
        canbus_pose_path = os.path.join(canbus_dir, f"{scene_name}_pose.json")
        canbus_poses: List[Dict[str, Any]] = []
        canbus_utimes: np.ndarray = np.array([], dtype=np.int64)
        if os.path.exists(canbus_pose_path):
            with open(canbus_pose_path, "r") as f:
                canbus_poses = json.load(f)
            canbus_utimes = np.array([e["utime"] for e in canbus_poses], dtype=np.int64)

        # 3. 提取所有帧的 yaw 值（用于推断 driving command）
        yaw_sequence = np.array(
            [_yaw_from_rotation_matrix(np.array(fr["pose"])[:3, :3]) for fr in raw_frames],
            dtype=np.float64,
        )
        driving_commands = _compute_driving_command(yaw_sequence)

        # 4. 转换每帧
        navsim_frames: List[Dict[str, Any]] = []
        scene_gt_dir = os.path.join(labels_dir, scene_name)

        for frame_idx, raw_frame in enumerate(raw_frames):
            pose_matrix = np.array(raw_frame["pose"], dtype=np.float64)  # [4, 4]
            lidar_ts = raw_frame["lidar_record"]["timestamp"]

            # === ego2global_translation ===
            ego2global_translation = pose_matrix[:3, 3].tolist()

            # === ego2global_rotation (quaternion wxyz) ===
            ego2global_rotation = _rotation_matrix_to_quaternion_wxyz(pose_matrix[:3, :3]).tolist()

            # === ego_dynamic_state ===
            canbus_entry = _interpolate_canbus_at_timestamp(canbus_poses, canbus_utimes, lidar_ts)
            ego_dynamic_state = _build_ego_dynamic_state(canbus_entry, pose_matrix)

            # === driving_command ===
            driving_command = driving_commands[frame_idx]

            # === cams: 只保留 CAM_FRONT ===
            cam_front_rel_path = raw_frame["images"][NUSCENES_CAM_FRONT_INDEX]
            cam_image_basename = os.path.basename(cam_front_rel_path)

            cams = {
                NAVSIM_CAMERA_NAME: {
                    "data_path": cam_image_basename,
                }
            }

            # === anns: agent annotations ===
            gt_box_filename = raw_frame["gt_box"]
            npz_path = os.path.join(scene_gt_dir, gt_box_filename)
            anns = _convert_annotations(npz_path)

            # === 组装 NavSim frame dict ===
            navsim_frame = {
                "ego2global_translation": ego2global_translation,
                "ego2global_rotation": ego2global_rotation,
                "ego_dynamic_state": ego_dynamic_state.tolist(),
                "driving_command": driving_command.tolist(),
                "cams": cams,
                "anns": anns,
            }
            navsim_frames.append(navsim_frame)

        # 5. 创建 sensor_blobs 目录并 symlink 图像
        cam_dir = os.path.join(sensor_blobs_scene_dir, NAVSIM_CAMERA_NAME)
        os.makedirs(cam_dir, exist_ok=True)

        for raw_frame in raw_frames:
            cam_front_rel_path = raw_frame["images"][NUSCENES_CAM_FRONT_INDEX]
            src_abs = os.path.join(nuscenes_root, cam_front_rel_path)
            dst_basename = os.path.basename(cam_front_rel_path)
            dst_abs = os.path.join(cam_dir, dst_basename)

            if not os.path.exists(dst_abs):
                if os.path.exists(src_abs):
                    os.symlink(src_abs, dst_abs)
                else:
                    # 原始图像不存在，跳过
                    pass

        # 6. 保存 PKL
        os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
        with open(output_pkl_path, "wb") as f:
            pickle.dump(navsim_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

        return (scene_name, True, f"{len(navsim_frames)} frames")

    except Exception as exc:
        return (scene_name, False, str(exc))


def main():
    parser = argparse.ArgumentParser(
        description="将 nuScenes 数据集转换为 NavSim PKL 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default="/path/nuScenes",
        help="nuScenes 数据集根目录（包含 labels/, samples/, can_bus/ 等）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/path/nuScenes/navsim_format_fix_2",
        help="输出目录根路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "trainval"],
        default="trainval",
        help="要转换的 split (train, val, trainval)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行转换的 worker 数量",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印场景列表，不执行转换",
    )
    args = parser.parse_args()

    # 发现所有可用场景
    labels_dir = os.path.join(args.nuscenes_root, "labels")
    if not os.path.isdir(labels_dir):
        print(f"错误: labels 目录不存在: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    all_json_files = sorted(Path(labels_dir).glob("scene-*.json"))
    all_scene_names = [p.stem for p in all_json_files]  # e.g. "scene-0001"
    print(f"发现 {len(all_scene_names)} 个场景")

    # 按 split 过滤
    if args.split == "val":
        scenes_to_convert = [(s, "val") for s in all_scene_names if s in NUSCENES_VAL_SCENES]
    elif args.split == "train":
        scenes_to_convert = [(s, "train") for s in all_scene_names if s not in NUSCENES_VAL_SCENES]
    else:  # trainval
        scenes_to_convert = [
            (s, "val" if s in NUSCENES_VAL_SCENES else "train") for s in all_scene_names
        ]

    train_count = sum(1 for _, sp in scenes_to_convert if sp == "train")
    val_count = sum(1 for _, sp in scenes_to_convert if sp == "val")
    print(f"将转换 {len(scenes_to_convert)} 个场景 (train={train_count}, val={val_count})")

    if args.dry_run:
        for scene_name, split in scenes_to_convert:
            print(f"  {split}: {scene_name}")
        return

    # 准备输出路径
    sensor_blobs_root = os.path.join(args.output_root, "sensor_blobs")

    # 并行转换
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scene_name, split in scenes_to_convert:
            output_pkl = os.path.join(args.output_root, split, f"{scene_name}.pkl")
            sensor_blobs_scene = os.path.join(sensor_blobs_root, scene_name)

            futures.append(
                pool.submit(
                    _convert_single_scene,
                    scene_name,
                    args.nuscenes_root,
                    output_pkl,
                    sensor_blobs_scene,
                )
            )

        # 收集结果
        success_count = 0
        fail_count = 0
        for future in as_completed(futures):
            scene_name, success, message = future.result()
            if success:
                success_count += 1
                if success_count % 100 == 0:
                    print(f"  进度: {success_count}/{len(futures)} 完成")
            else:
                fail_count += 1
                print(f"  失败: {scene_name}: {message}", file=sys.stderr)

    print(f"\n转换完成: {success_count} 成功, {fail_count} 失败")
    print(f"输出目录: {args.output_root}")
    print(f"  train PKL: {args.output_root}/train/")
    print(f"  val PKL:   {args.output_root}/val/")
    print(f"  图像 symlinks: {sensor_blobs_root}/")


if __name__ == "__main__":
    main()
