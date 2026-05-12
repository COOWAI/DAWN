"""
token_ae_sem_loss.py

Token AE 语义 bbox 监督损失（Hungarian 匹配版）。

旧版本使用最近邻匹配，存在问题：
- 无唯一性约束: 多个 GT 可匹配同一个 pred slot
- 无 objectness 监督: 无法训练模型区分有/无目标

新版本改进：
1. Hungarian 匹配 (scipy.optimize.linear_sum_assignment): 保证一一对应
2. 匹配代价: L1(bbox)
3. 损失组成:
   - 匹配 pair 的 L1(xyxy) bbox 回归损失
   - 所有 slot 的 objectness BCE (匹配=正, 未匹配=负)
4. Fallback: 若 scipy 不可用，退化为贪心 NN 匹配
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """二分图最优匹配。

    Parameters
    ----------
    cost : [K, N]  代价矩阵 (K pred slots, N gt boxes)

    Returns
    -------
    pred_idx : [M]  匹配到的 pred slot 下标
    gt_idx   : [M]  匹配到的 gt 下标, M = min(K, N)
    """
    try:
        from scipy.optimize import linear_sum_assignment

        cost_np = cost.detach().cpu().float().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        return (
            torch.as_tensor(row_ind, dtype=torch.long, device=cost.device),
            torch.as_tensor(col_ind, dtype=torch.long, device=cost.device),
        )
    except ImportError:
        # Fallback: 贪心 NN 匹配 (每个 GT 找最近 pred，不保证唯一)
        N = cost.shape[1]
        gt_idx = torch.arange(N, device=cost.device)
        pred_idx = cost[:, :N].argmin(dim=0)  # [N]
        return pred_idx, gt_idx


def sem_bbox_loss(
    pred_boxes: torch.Tensor,
    gt_boxes_per_frame: List[torch.Tensor],
    pred_obj: Optional[torch.Tensor] = None,
    obj_loss_weight: float = 1.0,
) -> torch.Tensor:
    """计算语义 bbox 损失（Hungarian 匹配 + objectness BCE）。

    Parameters
    ----------
    pred_boxes         : [B*T, K, 4]
        模型预测 bbox（xyxy, normalized）
    gt_boxes_per_frame : list of Tensor[N_i, 4]
        每帧 GT bbox 列表，可为空框 Tensor[0,4]
    pred_obj           : [B*T, K] or None
        objectness logits (pre-sigmoid)。若为 None 则不计算 objectness loss，
        保持向后兼容
    obj_loss_weight    : float
        objectness loss 权重（默认 1.0）

    Returns
    -------
    loss : scalar Tensor
        bbox_loss + obj_loss_weight * obj_loss
        若无有效 GT 返回 0
    """
    assert pred_boxes.ndim == 3 and pred_boxes.shape[-1] == 4, "pred_boxes must be [B*T, K, 4]"
    assert len(gt_boxes_per_frame) == pred_boxes.shape[0], "gt list length must match B*T"

    device = pred_boxes.device
    dtype = pred_boxes.dtype
    K = pred_boxes.shape[1]

    total_bbox_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_obj_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_matched_coords = 0
    total_frames_with_obj = 0

    for frame_idx, gt_boxes in enumerate(gt_boxes_per_frame):
        pred_frame = pred_boxes[frame_idx]  # [K, 4]

        # Objectness target: 默认全为 0 (无目标)
        if pred_obj is not None:
            obj_target = torch.zeros(K, device=device, dtype=dtype)

        if gt_boxes is not None and gt_boxes.numel() > 0:
            gt_boxes = gt_boxes.to(device=device, dtype=dtype)

            # L1 代价矩阵: [K, N]
            cost = torch.cdist(pred_frame, gt_boxes, p=1)

            # Hungarian 一一匹配
            matched_pred_idx, matched_gt_idx = _hungarian_match(cost)

            # Bbox L1 loss on matched pairs
            matched_pred = pred_frame[matched_pred_idx]  # [M, 4]
            matched_gt = gt_boxes[matched_gt_idx]  # [M, 4]
            total_bbox_loss = total_bbox_loss + F.l1_loss(matched_pred, matched_gt, reduction="sum")
            total_matched_coords += matched_pred.shape[0] * 4

            # 标记匹配到的 pred slot 为正样本
            if pred_obj is not None:
                obj_target[matched_pred_idx] = 1.0

        # Objectness BCE loss (所有 slot)
        if pred_obj is not None:
            frame_obj = pred_obj[frame_idx]  # [K]
            total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                frame_obj, obj_target, reduction="mean"
            )
            total_frames_with_obj += 1

    # Average bbox loss over matched coordinates
    bbox_loss = total_bbox_loss / max(total_matched_coords, 1)

    # Average objectness loss over frames
    if pred_obj is not None and total_frames_with_obj > 0:
        obj_loss = total_obj_loss / total_frames_with_obj
    else:
        obj_loss = pred_boxes.sum() * 0.0

    total_loss = bbox_loss + obj_loss_weight * obj_loss

    # 若无任何匹配也无 obj 输出，返回 0 loss (不影响梯度)
    if total_matched_coords == 0 and (pred_obj is None or total_frames_with_obj == 0):
        return pred_boxes.sum() * 0.0

    return total_loss
