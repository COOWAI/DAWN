"""NavSim dataloader utilities for world model training."""

import hashlib
import json
import os
import pickle
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_cowa_world_model.utils.metrics import BEV_SIZE, _rasterize_agents_to_bev
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SceneRecord:
    scene_name: str
    pkl_path: str
    camera_dir: str
    valid_frame_indices: List[int]


@dataclass
class WindowRecord:
    """A single sliding-window sample inside a scene (log).

    Attributes
    ----------
    scene_idx : int
        Index into the parent ``scenes`` list (for PKL loading / caching).
    start_pos : int
        Start position within ``SceneRecord.valid_frame_indices``.
    """

    scene_idx: int
    start_pos: int


class NavSimWorldModelDataset(Dataset):
    """Dataset that adapts NavSim logs to the world-model batch contract."""

    # 碰撞检测需要过滤的动态 agent 类型
    _DYNAMIC_AGENT_TYPES = {"vehicle", "pedestrian", "bicycle"}

    def __init__(
        self,
        data_path: str,
        sensor_blobs_path: str,
        camera_name: str = "CAM_F0",
        frames_per_clip: int = 20,
        fps: int = 2,
        tubelet_size: int = 2,
        transform: Any = None,
        proposal_transform: Any = None,
        max_scenes: Optional[int] = None,
        action_dim: int = 3,
        cache_size: int = 8,
        index_cache: bool = True,
        window_stride: int = 1,
        max_frame_gap: int = 3,
        max_agents: int = 50,
        load_agent_annotations: bool = True,
    ):
        self.data_path = data_path
        self.sensor_blobs_path = sensor_blobs_path
        self.camera_name = camera_name
        self.frames_per_clip = int(frames_per_clip)
        self.fps = int(max(1, fps))
        self.tubelet_size = int(max(1, tubelet_size))
        self.transform = transform
        self.proposal_transform = proposal_transform
        self.max_scenes = max_scenes
        self.action_dim = action_dim
        self.cache_size = max(1, int(cache_size))
        self.index_cache = index_cache
        self.window_stride = max(1, int(window_stride))
        self.max_frame_gap = max(1, int(max_frame_gap))
        self.max_agents = max_agents
        self.load_agent_annotations = load_agent_annotations

        # NavSim logs are typically around 2Hz.
        self.base_fps = 2.0
        self.sample_step = max(1, int(round(self.base_fps / float(self.fps))))
        self.min_valid_frames = 1 + (self.frames_per_clip - 1) * self.sample_step

        self._scene_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self.scenes = self._build_scene_index()

        if not self.scenes:
            raise ValueError(
                "No valid NavSim scenes found. "
                f"data_path={data_path}, sensor_blobs_path={sensor_blobs_path}, "
                f"camera={camera_name}, required_frames={self.min_valid_frames}"
            )

        # Build sliding-window entries from scenes.
        self.windows = self._build_window_index()

        if not self.windows:
            raise ValueError(
                "No valid sliding windows could be built from scenes. "
                f"scenes={len(self.scenes)}, min_valid_frames={self.min_valid_frames}, "
                f"window_stride={self.window_stride}"
            )

        logger.info(
            "NavSim dataset ready: scenes=%d, windows=%d, camera=%s, "
            "frames_per_clip=%d, sample_step=%d, window_stride=%d",
            len(self.scenes),
            len(self.windows),
            self.camera_name,
            self.frames_per_clip,
            self.sample_step,
            self.window_stride,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        max_retries = 5
        window_idx = int(index)

        for retry in range(max_retries):
            window = self.windows[window_idx]
            scene = self.scenes[window.scene_idx]
            try:
                frames = self._load_scene_frames(scene.pkl_path)
                sampled_frame_indices = self._window_frame_indices(scene.valid_frame_indices, window.start_pos)

                buffer = self._load_clip_images(scene, frames, sampled_frame_indices)
                raw_buffer = buffer
                if self.transform is not None:
                    buffer = self.transform(buffer)
                    if not torch.is_tensor(buffer):
                        buffer = torch.as_tensor(buffer)
                else:
                    buffer = torch.from_numpy(buffer).permute(3, 0, 1, 2).float() / 255.0

                proposal_buffer = None
                if self.proposal_transform is not None:
                    proposal_buffer = self.proposal_transform(raw_buffer)
                    if not torch.is_tensor(proposal_buffer):
                        proposal_buffer = torch.as_tensor(proposal_buffer)

                reduced_frame_indices = sampled_frame_indices[:: self.tubelet_size]
                states = self._build_states(frames, reduced_frame_indices)
                actions = self._build_actions(states)
                driving_command = self._build_driving_commands(frames, reduced_frame_indices)
                ego_dynamics = self._build_ego_dynamics(frames, reduced_frame_indices)

                # Keep a stable 7D shape for compatibility with existing world-model path.
                extrinsics = np.zeros_like(states, dtype=np.float32)

                # --- agent annotations (for collision detection) ---
                if self.load_agent_annotations:
                    agent_boxes, agent_mask = self._build_agent_annotations(frames, reduced_frame_indices)
                    bev_segmentation = self._build_bev_segmentation(agent_boxes, agent_mask)
                else:
                    T_reduced = len(reduced_frame_indices)
                    agent_boxes = np.zeros((T_reduced, self.max_agents, 7), dtype=np.float32)
                    agent_mask = np.zeros((T_reduced, self.max_agents), dtype=np.bool_)
                    bev_segmentation = np.zeros((T_reduced, BEV_SIZE, BEV_SIZE), dtype=np.uint8)

                return {
                    "buffer": buffer,
                    "proposal_buffer": proposal_buffer,
                    "actions": actions,
                    "states": states,
                    "extrinsics": extrinsics,
                    "indices": np.asarray(sampled_frame_indices, dtype=np.int64),
                    "scene_name": scene.scene_name,
                    "pkl_path": scene.pkl_path,
                    "window_start_pos": int(window.start_pos),
                    "sampled_frame_indices": np.asarray(sampled_frame_indices, dtype=np.int64),
                    "seg_masks": None,
                    "seg_frame_indices": None,
                    "driving_command": driving_command,
                    "ego_dynamics": ego_dynamics,
                    "agent_boxes": agent_boxes,
                    "agent_mask": agent_mask,
                    "bev_segmentation": bev_segmentation,
                }
            except Exception as exc:
                if retry == max_retries - 1:
                    raise
                logger.warning(
                    "NavSim sample load failed (scene=%s, window_start=%d, retry=%d/%d): %s",
                    scene.scene_name,
                    window.start_pos,
                    retry + 1,
                    max_retries,
                    str(exc),
                )
                window_idx = random.randint(0, len(self.windows) - 1)

        raise RuntimeError("Unreachable NavSim dataset retry state")

    def _build_scene_index(self) -> List[SceneRecord]:
        if not os.path.isdir(self.data_path):
            raise ValueError(f"NavSim data_path does not exist: {self.data_path}")
        if not os.path.isdir(self.sensor_blobs_path):
            raise ValueError(f"NavSim sensor_blobs_path does not exist: {self.sensor_blobs_path}")

        pkl_paths = sorted(glob(os.path.join(self.data_path, "*.pkl")))
        if self.max_scenes is not None:
            pkl_paths = pkl_paths[: int(self.max_scenes)]

        # --- try loading from disk cache ---
        cache_path = self._get_index_cache_path(len(pkl_paths))
        if cache_path is not None:
            cached = self._load_index_cache(cache_path, len(pkl_paths))
            if cached is not None:
                return cached

        # --- build index from scratch (with progress logging) ---
        t0 = time.monotonic()
        total = len(pkl_paths)
        logger.info("Building NavSim scene index from %d pkl files ...", total)

        scenes: List[SceneRecord] = []
        skipped_missing_camera_dir = 0
        skipped_not_enough_frames = 0

        for i, pkl_path in enumerate(pkl_paths):
            scene_name = os.path.splitext(os.path.basename(pkl_path))[0]
            camera_dir = os.path.join(self.sensor_blobs_path, scene_name, self.camera_name)
            if not os.path.isdir(camera_dir):
                skipped_missing_camera_dir += 1
                continue

            image_names = {name for name in os.listdir(camera_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))}
            if not image_names:
                skipped_missing_camera_dir += 1
                continue

            frames = self._read_scene_frames_no_cache(pkl_path)
            valid_indices = self._compute_valid_frame_indices(frames, image_names)

            if len(valid_indices) < self.min_valid_frames:
                skipped_not_enough_frames += 1
                continue

            scenes.append(
                SceneRecord(
                    scene_name=scene_name,
                    pkl_path=pkl_path,
                    camera_dir=camera_dir,
                    valid_frame_indices=valid_indices,
                )
            )

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  scene index progress: %d/%d (%.1f%%) elapsed=%.1fs",
                    i + 1,
                    total,
                    100.0 * (i + 1) / total,
                    elapsed,
                )

        elapsed = time.monotonic() - t0
        logger.info(
            "NavSim scene index built: total_pkls=%d, kept=%d, "
            "skipped_missing_camera=%d, skipped_short=%d, time=%.1fs",
            total,
            len(scenes),
            skipped_missing_camera_dir,
            skipped_not_enough_frames,
            elapsed,
        )

        # --- persist to disk cache ---
        if cache_path is not None:
            self._save_index_cache(cache_path, scenes, total)

        return scenes

    def _build_window_index(self) -> List[WindowRecord]:
        """Enumerate all sliding-window positions across all scenes.

        For each scene, the number of valid starting positions is determined by
        how many full ``min_valid_frames``-length windows fit, stepping by
        ``window_stride`` (in units of *valid-index positions*, not raw frame
        indices).

        Windows whose raw frame indices contain gaps larger than
        ``max_frame_gap`` are rejected to avoid GT trajectory jumps caused by
        missing intermediate frames.

        Returns
        -------
        List[WindowRecord]
            Flat list of (scene_idx, start_pos) pairs, one per window.
        """
        windows: List[WindowRecord] = []
        total_candidates = 0
        rejected_gap = 0
        for scene_idx, scene in enumerate(self.scenes):
            n_valid = len(scene.valid_frame_indices)
            max_start = n_valid - self.min_valid_frames
            for start in range(0, max_start + 1, self.window_stride):
                total_candidates += 1
                if not self._is_window_continuous(scene.valid_frame_indices, start):
                    rejected_gap += 1
                    continue
                windows.append(WindowRecord(scene_idx=scene_idx, start_pos=start))

        logger.info(
            "Built window index: %d windows from %d scenes "
            "(stride=%d, min_valid=%d, max_frame_gap=%d, "
            "rejected_gap=%d/%d candidates)",
            len(windows),
            len(self.scenes),
            self.window_stride,
            self.min_valid_frames,
            self.max_frame_gap,
            rejected_gap,
            total_candidates,
        )
        return windows

    def _window_frame_indices(self, valid_indices: Sequence[int], start_pos: int) -> List[int]:
        """Return deterministic frame indices for a window starting at *start_pos*."""
        positions = start_pos + np.arange(self.frames_per_clip) * self.sample_step
        return [valid_indices[int(p)] for p in positions]

    def _is_window_continuous(self, valid_indices: Sequence[int], start_pos: int) -> bool:
        """Check that raw frame indices in a window are within ``max_frame_gap``.

        For every pair of consecutive sampled positions the difference of the
        underlying raw frame indices must be <= ``self.max_frame_gap``.  This
        prevents windows that span scene-boundary gaps where hundreds of frames
        (and hundreds of metres) are missing.
        """
        positions = start_pos + np.arange(self.frames_per_clip) * self.sample_step
        for i in range(len(positions) - 1):
            raw_gap = valid_indices[int(positions[i + 1])] - valid_indices[int(positions[i])]
            if raw_gap > self.max_frame_gap:
                return False
        return True

    # ------------------------------------------------------------------
    # Scene index disk cache helpers
    # ------------------------------------------------------------------

    def _get_index_cache_path(self, total_pkls: int) -> Optional[str]:
        """Return cache file path, or ``None`` if caching is disabled."""
        if not self.index_cache:
            return None
        fingerprint_data = json.dumps(
            {
                "data_path": os.path.abspath(self.data_path),
                "sensor_blobs_path": os.path.abspath(self.sensor_blobs_path),
                "camera_name": self.camera_name,
                "frames_per_clip": self.frames_per_clip,
                "fps": self.fps,
                "tubelet_size": self.tubelet_size,
                "max_scenes": self.max_scenes,
                "max_frame_gap": self.max_frame_gap,
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        return os.path.join(self.data_path, f".navsim_scene_index_cache_{fingerprint}.pkl")

    def _load_index_cache(self, cache_path: str, current_total_pkls: int) -> Optional[List[SceneRecord]]:
        """Try to load & validate a cached scene index.  Return ``None`` on miss / stale."""
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or payload.get("version") != 1:
                logger.info("Scene index cache version mismatch, will rebuild.")
                return None
            if payload.get("total_pkls") != current_total_pkls:
                logger.info(
                    "Scene index cache stale (cached %s pkls vs current %d), will rebuild.",
                    payload.get("total_pkls"),
                    current_total_pkls,
                )
                return None
            scenes: List[SceneRecord] = payload["scenes"]
            logger.info("Loaded scene index from cache: %s (%d scenes)", cache_path, len(scenes))
            return scenes
        except Exception as exc:
            logger.warning("Failed to load scene index cache %s: %s", cache_path, exc)
            return None

    def _save_index_cache(self, cache_path: str, scenes: List[SceneRecord], total_pkls: int) -> None:
        """Persist the scene index to disk."""
        payload = {
            "version": 1,
            "total_pkls": total_pkls,
            "scenes": scenes,
        }
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            logger.info("Saved scene index cache: %s (%d scenes)", cache_path, len(scenes))
        except Exception as exc:
            logger.warning("Failed to save scene index cache %s: %s", cache_path, exc)

    def _compute_valid_frame_indices(self, frames: Sequence[Dict[str, Any]], image_names: set) -> List[int]:
        valid_indices: List[int] = []
        for idx, frame in enumerate(frames):
            cam_dict = frame.get("cams", {}).get(self.camera_name)
            if cam_dict is None:
                continue
            rel_path = cam_dict.get("data_path")
            if not rel_path:
                continue
            image_name = os.path.basename(rel_path)
            if image_name in image_names:
                valid_indices.append(idx)
        return valid_indices

    def _read_scene_frames_no_cache(self, pkl_path: str) -> List[Dict[str, Any]]:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list):
            raise ValueError(f"Unexpected NavSim pickle structure: {pkl_path}")
        return frames

    def _load_scene_frames(self, pkl_path: str) -> List[Dict[str, Any]]:
        cached = self._scene_cache.get(pkl_path)
        if cached is not None:
            self._scene_cache.move_to_end(pkl_path)
            return cached

        frames = self._read_scene_frames_no_cache(pkl_path)
        self._scene_cache[pkl_path] = frames
        self._scene_cache.move_to_end(pkl_path)

        if len(self._scene_cache) > self.cache_size:
            self._scene_cache.popitem(last=False)

        return frames

    def _load_clip_images(
        self,
        scene: SceneRecord,
        frames: Sequence[Dict[str, Any]],
        sampled_frame_indices: Sequence[int],
    ) -> np.ndarray:
        images: List[np.ndarray] = []

        for frame_idx in sampled_frame_indices:
            frame = frames[frame_idx]
            rel_path = frame["cams"][self.camera_name]["data_path"]
            image_name = os.path.basename(rel_path)
            image_path = os.path.join(scene.camera_dir, image_name)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as img:
                images.append(np.asarray(img.convert("RGB"), dtype=np.uint8))

        return np.stack(images, axis=0)

    def _build_agent_annotations(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从 NavSim frame 的 anns 字段提取 agent bounding boxes。

        Parameters
        ----------
        frames               : NavSim raw frame dicts (PKL)
        reduced_frame_indices : tubelet 降采样后的帧索引

        Returns
        -------
        agent_boxes : [T, max_agents, 7]  float32
            每个 agent 的 [x, y, z, length, width, height, heading]，
            坐标系为该帧自车坐标系。
        agent_mask  : [T, max_agents]     bool
            True 表示该位置有有效 agent。
        """
        T = len(reduced_frame_indices)
        agent_boxes = np.zeros((T, self.max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((T, self.max_agents), dtype=np.bool_)

        for t_idx, frame_idx in enumerate(reduced_frame_indices):
            frame = frames[frame_idx]
            anns = frame.get("anns")
            if anns is None:
                continue

            gt_boxes = np.asarray(anns["gt_boxes"], dtype=np.float32)  # [N, 7]
            gt_names = anns["gt_names"]  # List[str]

            # 过滤：只保留动态 agent 类型
            keep_indices = [i for i, name in enumerate(gt_names) if name in self._DYNAMIC_AGENT_TYPES]

            if not keep_indices:
                continue

            kept_boxes = gt_boxes[keep_indices]

            # 截断到 max_agents
            n_keep = min(len(kept_boxes), self.max_agents)
            agent_boxes[t_idx, :n_keep] = kept_boxes[:n_keep]
            agent_mask[t_idx, :n_keep] = True

        return agent_boxes, agent_mask

    def _build_bev_segmentation(
        self,
        agent_boxes: np.ndarray,
        agent_mask: np.ndarray,
    ) -> np.ndarray:
        """将 agent boxes 栅格化为 per-frame BEV 分割图。

        每帧的 seg map 在该帧的 ego 坐标系下（与 ST-P3/VAD 对齐）。

        Parameters
        ----------
        agent_boxes : [T, max_agents, 7]  各帧 ego 坐标系
        agent_mask  : [T, max_agents]     bool

        Returns
        -------
        bev_seg : [T, BEV_SIZE, BEV_SIZE]  uint8, 1=occupied, 0=free
        """
        T = agent_boxes.shape[0]
        bev_seg = np.zeros((T, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        for t in range(T):
            bev_seg[t] = _rasterize_agents_to_bev(agent_boxes[t], agent_mask[t])
        return bev_seg

    def _build_states(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Build states array from NavSim frames.

        NavSim stores ego poses as UTM global coordinates (``ego2global_translation``,
        magnitude ~10^5–10^6).  Storing these directly as ``float32`` causes significant
        precision loss (up to 0.5 m for the y-component).  To avoid this we:

        1. Read translations in **float64**.
        2. Subtract the **first frame's translation** (scene-centre) so that all
           subsequent positions are O(0–100 m).
        3. Cast to **float32** – now safe because the magnitudes are small.

        The centering is algebraically transparent to all downstream consumers
        (actions, GT trajectory, status features) because they only use
        *differences* between frames.
        """
        # --- collect raw translations in float64 for precision ---
        translations_f64: List[np.ndarray] = []
        orientations: List[np.ndarray] = []
        speeds: List[float] = []

        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]

            translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
            translations_f64.append(translation)

            quat_wxyz = np.asarray(frame["ego2global_rotation"], dtype=np.float64)
            quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
            roll_pitch_yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz")
            orientations.append(roll_pitch_yaw)

            ego_dynamic_state = frame.get("ego_dynamic_state", [0.0, 0.0, 0.0, 0.0])
            speeds.append(float(ego_dynamic_state[0]))

        # --- centre translations around first frame (float64 arithmetic) ---
        translations_arr = np.stack(translations_f64, axis=0)  # [T, 3] float64
        origin_translation = translations_arr[0].copy()  # first-frame UTM origin
        translations_arr -= origin_translation  # now O(0–100 m)

        # --- assemble final states [T, 7] as float32 ---
        orientations_arr = np.stack(orientations, axis=0)  # [T, 3] float64
        speeds_arr = np.asarray(speeds, dtype=np.float64).reshape(-1, 1)  # [T, 1]
        states = np.concatenate([translations_arr, orientations_arr, speeds_arr], axis=1)  # [T, 7]

        return states.astype(np.float32)

    def _build_driving_commands(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Extract per-frame driving_command from NavSim raw data.

        Each frame in the NavSim PKL contains a ``driving_command`` field — a
        4-element integer one-hot array from the nuPlan route planner indicating
        the high-level navigation intent (GO_STRAIGHT / TURN_LEFT / TURN_RIGHT /
        U_TURN).

        Parameters
        ----------
        frames               : raw NavSim frame dicts loaded from PKL.
        reduced_frame_indices : indices into *frames* (after tubelet down-sampling).

        Returns
        -------
        np.ndarray
            ``[T, 4]`` float32 one-hot driving commands.

        Raises
        ------
        KeyError
            If any frame is missing the ``driving_command`` field.
        """
        cmds: List[np.ndarray] = []
        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]
            # 字段缺失时直接报错，不做静默降级
            cmd = np.asarray(frame["driving_command"], dtype=np.float32)
            cmds.append(cmd[:4])  # 确保只取前 4 维
        return np.stack(cmds, axis=0)  # [T, 4]

    def _build_ego_dynamics(
        self,
        frames: Sequence[Dict[str, Any]],
        reduced_frame_indices: Sequence[int],
    ) -> np.ndarray:
        """Extract per-frame ego_dynamic_state [vx, vy, ax, ay] from NavSim raw data.

        Parameters
        ----------
        frames               : raw NavSim frame dicts loaded from PKL.
        reduced_frame_indices : indices into *frames* (after tubelet down-sampling).

        Returns
        -------
        np.ndarray
            ``[T, 4]`` float32 array of ``[vx, vy, ax, ay]``.

        Raises
        ------
        KeyError
            If any frame is missing the ``ego_dynamic_state`` field.
        """
        dynamics: List[np.ndarray] = []
        for frame_idx in reduced_frame_indices:
            frame = frames[frame_idx]
            # 字段缺失时直接报错，不做静默降级
            dyn = np.asarray(frame["ego_dynamic_state"], dtype=np.float32)
            dynamics.append(dyn[:4])  # [vx, vy, ax, ay]
        return np.stack(dynamics, axis=0)  # [T, 4]

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        if states.shape[0] < 2:
            raise ValueError("Need at least 2 reduced states to build actions")
        return self._build_actions_3d(states)

    def _build_actions_3d(self, states: np.ndarray) -> np.ndarray:
        """Build 3D actions: [dx_ego, dy_ego, d_yaw].

        NavSim states[:3] are global UTM coordinates (ego2global_translation).
        We rotate the global position difference into the ego frame of the
        *current* timestep so that dx ≈ forward displacement and dy ≈ lateral
        displacement, consistent with Mongo Raw's local-coordinate actions.
        """
        t_steps = states.shape[0]
        actions = np.zeros((t_steps - 1, 3), dtype=np.float32)

        for t in range(t_steps - 1):
            # --- global position diff → ego-frame position diff ---
            dx_global = states[t + 1, 0] - states[t, 0]
            dy_global = states[t + 1, 1] - states[t, 1]

            yaw = states[t, 5]  # current ego yaw in world frame
            cos_h = np.cos(-yaw)
            sin_h = np.sin(-yaw)
            dx_ego = cos_h * dx_global - sin_h * dy_global
            dy_ego = sin_h * dx_global + cos_h * dy_global

            # --- yaw diff ---
            d_yaw = states[t + 1, 5] - states[t, 5]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))

            actions[t] = np.asarray([dx_ego, dy_ego, d_yaw], dtype=np.float32)

        return actions


def navsim_world_model_collate_fn(batch: Sequence[Dict[str, Any]]):
    context_frames = torch.stack([item["buffer"] for item in batch])
    actions = torch.stack([torch.from_numpy(item["actions"]) for item in batch])
    states = torch.stack([torch.from_numpy(item["states"]) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item["extrinsics"]) for item in batch])

    seg_targets = [None for _ in batch]

    # NavSim-specific fields (driving_command, ego_dynamics) appended at tuple tail
    driving_command = torch.stack([torch.from_numpy(item["driving_command"]) for item in batch])
    ego_dynamics = torch.stack([torch.from_numpy(item["ego_dynamics"]) for item in batch])

    # Agent annotations for collision detection (index 7, 8)
    agent_boxes = torch.stack([torch.from_numpy(item["agent_boxes"]) for item in batch])
    agent_mask = torch.stack([torch.from_numpy(item["agent_mask"].astype(np.uint8)).bool() for item in batch])

    # Pre-computed BEV segmentation maps for collision rate (index 9)
    bev_segmentation = torch.stack([torch.from_numpy(item["bev_segmentation"]) for item in batch])

    # Optional proposal encoder frames (index 10), e.g. Drive-JEPA 256x512 while
    # the main V-JEPA branch keeps its own 256x256 transform.
    proposal_context_frames = None
    if batch and batch[0].get("proposal_buffer") is not None:
        proposal_context_frames = torch.stack([item["proposal_buffer"] for item in batch])

    metadata = {
        "scene_name": [str(item.get("scene_name", "")) for item in batch],
        "pkl_path": [str(item.get("pkl_path", "")) for item in batch],
        "window_start_pos": torch.as_tensor(
            [int(item.get("window_start_pos", -1)) for item in batch], dtype=torch.long
        ),
        "sampled_frame_indices": torch.stack(
            [
                torch.as_tensor(item.get("sampled_frame_indices", item.get("indices", [])), dtype=torch.long)
                for item in batch
            ]
        ),
    }

    return (
        context_frames,
        actions,
        states,
        extrinsics,
        seg_targets,
        driving_command,
        ego_dynamics,
        agent_boxes,
        agent_mask,
        bev_segmentation,
        proposal_context_frames,
        metadata,
    )


def init_navsim_data(
    data_path: str,
    sensor_blobs_path: str,
    batch_size: int,
    frames_per_clip: int = 20,
    fps: int = 2,
    tubelet_size: int = 2,
    transform: Any = None,
    proposal_transform: Any = None,
    num_workers: int = 4,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    rank: int = 0,
    world_size: int = 1,
    camera_name: str = "CAM_F0",
    max_scenes: Optional[int] = None,
    action_dim: int = 7,
    shuffle: bool = True,
    index_cache: bool = True,
    window_stride: int = 1,
    max_frame_gap: int = 3,
    max_agents: int = 50,
    load_agent_annotations: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    dataset = NavSimWorldModelDataset(
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        camera_name=camera_name,
        frames_per_clip=frames_per_clip,
        fps=fps,
        tubelet_size=tubelet_size,
        transform=transform,
        proposal_transform=proposal_transform,
        max_scenes=max_scenes,
        action_dim=action_dim,
        index_cache=index_cache,
        window_stride=window_stride,
        max_frame_gap=max_frame_gap,
        max_agents=max_agents,
        load_agent_annotations=load_agent_annotations,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=navsim_world_model_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=drop_last,
    )

    logger.info(
        "NavSim dataloader created: batches=%d, batch_size=%d, workers=%d",
        len(loader),
        batch_size,
        num_workers,
    )
    return loader, sampler
