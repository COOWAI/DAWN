"""
token_ae_det_head.py

Token AE 语义监督辅助检测头 (DETR-style cross-attention)：

旧版本使用 mean-pool + MLP，存在以下问题：
1. mean pooling 破坏空间信息 → 所有 slot 看到相同特征，无法区分不同位置目标
2. 无 objectness 输出 → 无法处理可变数量目标
3. 单层 MLP 无 slot 间交互 → 无法避免重复预测

新版本参考 DETR 设计：
- learnable object queries 通过 cross-attention 关注输入 latent tokens 的不同区域
- self-attention 让 query 之间互相感知，避免重复检测
- 每个 query 独立预测 bbox + objectness
- 配合 Hungarian 匹配损失实现一一对应

输入 per-frame latent tokens [B*T, L, D]
输出 bbox [B*T, max_obj, 4] (xyxy normalized) + objectness logits [B*T, max_obj]
"""

from typing import Tuple

import torch
import torch.nn as nn


class TokenAEDetHead(nn.Module):
    """DETR-style 辅助检测头。

    使用 learnable object queries 通过 TransformerDecoder 对
    输入 per-frame latent tokens 执行 cross-attention，实现
    多目标空间定位和 objectness 预测。

    Parameters
    ----------
    embed_dim  : int   输入 token 维度 (如 1408)
    max_obj    : int   每帧最大目标数 (query 数量)
    num_heads  : int   cross-attention heads (默认 8)
    num_layers : int   decoder 层数 (默认 2)
    ffn_dim    : int   FFN 中间维度 (默认 512)
    dropout    : float dropout rate (默认 0.0, 因为是辅助任务)
    """

    def __init__(
        self,
        embed_dim: int,
        max_obj: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_obj = max_obj
        self.embed_dim = embed_dim

        # Learnable object queries
        self.object_queries = nn.Parameter(torch.zeros(max_obj, embed_dim))
        nn.init.normal_(self.object_queries, std=0.02)

        # TransformerDecoder: self-attn (query ↔ query) + cross-attn (query → memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        # Per-query bbox head: embed_dim → 4 (xyxy normalized)
        hidden_dim = max(embed_dim // 4, 128)
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

        # Per-query objectness head: embed_dim → 1
        self.obj_head = nn.Linear(embed_dim, 1)

    def forward(self, z_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_frames : [B*T, L, D]  per-frame latent tokens (L = num_latent_tokens)

        Returns
        -------
        pred_boxes : [B*T, max_obj, 4]
            归一化 xyxy，范围 [0, 1]
        pred_obj : [B*T, max_obj]
            objectness logits (pre-sigmoid)，正值表示存在目标
        """
        BT = z_frames.shape[0]

        # Expand queries for batch: [1, K, D] -> [BT, K, D]
        queries = self.object_queries.unsqueeze(0).expand(BT, -1, -1)

        # Decoder: queries attend to latent tokens via cross-attention
        # 内部包含 self-attn (queries 之间) + cross-attn (queries → z_frames)
        decoded = self.decoder(queries, z_frames)  # [BT, K, D]
        decoded = self.norm(decoded)

        # Predict bbox (sigmoid → [0,1]) and objectness (raw logits)
        pred_boxes_raw = self.bbox_head(decoded).sigmoid()  # [BT, K, 4]
        pred_obj = self.obj_head(decoded).squeeze(-1)  # [BT, K]

        # 保证 xyxy 格式：x1 <= x2, y1 <= y2
        x1 = torch.minimum(pred_boxes_raw[..., 0], pred_boxes_raw[..., 2])
        y1 = torch.minimum(pred_boxes_raw[..., 1], pred_boxes_raw[..., 3])
        x2 = torch.maximum(pred_boxes_raw[..., 0], pred_boxes_raw[..., 2])
        y2 = torch.maximum(pred_boxes_raw[..., 1], pred_boxes_raw[..., 3])
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return pred_boxes, pred_obj
