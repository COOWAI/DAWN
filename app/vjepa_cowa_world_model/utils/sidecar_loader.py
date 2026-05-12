"""
sidecar_loader.py

加载 sidecar JSONL 语义标注，并提供按 (trajectory_name, frame_idx)
查询交通灯/障碍物 bbox 的接口。

bbox 输入格式: [x, y, w, h] (像素坐标, 基于原图 1920x1080)
输出格式: [x1, y1, x2, y2] 并归一化到 [0, 1]
"""

import json
import os
from bisect import bisect_left
from typing import Dict, List, Tuple

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SidecarAnnotations:
    """Sidecar 标注索引器。"""

    def __init__(
        self,
        sidecar_root: str,
        split: str,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> None:
        self.sidecar_root = sidecar_root
        self.split = split
        self.image_width = float(image_width)
        self.image_height = float(image_height)

        sidecar_path = os.path.join(sidecar_root, split, f"{split}_sidecar.jsonl")
        if not os.path.exists(sidecar_path):
            raise FileNotFoundError(f"Sidecar file not found: {sidecar_path}")

        self._index: Dict[str, Dict[int, Dict[str, List[List[float]]]]] = {}
        self._sorted_frame_indices: Dict[str, List[int]] = {}

        num_lines = 0
        num_frames = 0
        with open(sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                num_lines += 1
                entry = json.loads(line)
                trajectory_name = entry.get("trajectory_name")
                if trajectory_name is None:
                    continue

                traj_dict: Dict[int, Dict[str, List[List[float]]]] = {}
                for frame_item in entry.get("frames", []):
                    frame_idx = int(frame_item.get("frame_idx", -1))
                    if frame_idx < 0:
                        continue
                    tl_boxes = [obj.get("bbox", []) for obj in frame_item.get("traffic_light_bboxes", [])]
                    obs_boxes = [obj.get("bbox", []) for obj in frame_item.get("obstacles", [])]
                    traj_dict[frame_idx] = {
                        "traffic_light_bboxes": [b for b in tl_boxes if len(b) == 4],
                        "obstacles": [b for b in obs_boxes if len(b) == 4],
                    }

                self._index[trajectory_name] = traj_dict
                self._sorted_frame_indices[trajectory_name] = sorted(traj_dict.keys())
                num_frames += len(traj_dict)

        logger.info(
            "[Sidecar] Loaded split=%s from %s | trajectories=%d lines=%d frames=%d",
            split,
            sidecar_path,
            len(self._index),
            num_lines,
            num_frames,
        )

    def _xywh_to_xyxy_norm(self, bbox_xywh: List[float]) -> List[float]:
        x, y, w, h = bbox_xywh
        x1 = max(0.0, float(x))
        y1 = max(0.0, float(y))
        x2 = min(self.image_width, x1 + max(0.0, float(w)))
        y2 = min(self.image_height, y1 + max(0.0, float(h)))

        if x2 <= x1 or y2 <= y1:
            return []

        return [
            x1 / self.image_width,
            y1 / self.image_height,
            x2 / self.image_width,
            y2 / self.image_height,
        ]

    def _nearest_frame_idx(self, trajectory_name: str, frame_idx: int) -> int:
        frame_list = self._sorted_frame_indices.get(trajectory_name, [])
        if not frame_list:
            return -1

        pos = bisect_left(frame_list, frame_idx)
        if pos == 0:
            return frame_list[0]
        if pos == len(frame_list):
            return frame_list[-1]

        left = frame_list[pos - 1]
        right = frame_list[pos]
        if abs(left - frame_idx) <= abs(right - frame_idx):
            return left
        return right

    def get_frame_bboxes(self, trajectory_name: str, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定轨迹帧的 bbox。

        若 frame_idx 不在 sidecar 中，使用最近邻帧。

        Parameters
        ----------
        trajectory_name : str
        frame_idx : int

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            tl_boxes: [N_tl, 4]  (xyxy normalized)
            obs_boxes: [N_obs, 4] (xyxy normalized)
        """
        traj_dict = self._index.get(trajectory_name)
        if traj_dict is None:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

        nearest_idx = self._nearest_frame_idx(trajectory_name, int(frame_idx))
        if nearest_idx < 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

        frame_ann = traj_dict.get(nearest_idx, {})
        tl_boxes_raw = frame_ann.get("traffic_light_bboxes", [])
        obs_boxes_raw = frame_ann.get("obstacles", [])

        tl_boxes = [self._xywh_to_xyxy_norm(b) for b in tl_boxes_raw]
        obs_boxes = [self._xywh_to_xyxy_norm(b) for b in obs_boxes_raw]

        tl_boxes = [b for b in tl_boxes if len(b) == 4]
        obs_boxes = [b for b in obs_boxes if len(b) == 4]

        tl_tensor = (
            torch.tensor(tl_boxes, dtype=torch.float32) if tl_boxes else torch.zeros((0, 4), dtype=torch.float32)
        )
        obs_tensor = (
            torch.tensor(obs_boxes, dtype=torch.float32) if obs_boxes else torch.zeros((0, 4), dtype=torch.float32)
        )
        return tl_tensor, obs_tensor
