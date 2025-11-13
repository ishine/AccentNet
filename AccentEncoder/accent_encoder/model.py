"""
Model definitions for the accent encoder.

The backbone is a small Conformer-style stack operating on log-mel frames,
followed by attentive statistics pooling and projection heads for accent /
speaker classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    """Lightweight conformer-style block with depthwise convolution."""

    def __init__(
        self,
        dim: int,
        ff_multiplier: int = 4,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_multiplier * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dim, dim),
            nn.Dropout(dropout),
        )
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_multiplier * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dim, dim),
            nn.Dropout(dropout),
        )
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_pointwise1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.conv_activation = nn.GLU(dim=1)
        self.conv_depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=dim,
        )
        self.conv_batchnorm = nn.BatchNorm1d(dim)
        self.conv_pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = residual + 0.5 * self.ff1(x)

        attn_input = x
        if padding_mask is not None:
            attn_mask = ~padding_mask
        else:
            attn_mask = None
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, key_padding_mask=attn_mask)
        x = x + attn_out

        residual = x
        x = residual + 0.5 * self.ff2(x)

        conv_input = self.conv_norm(x)
        conv_input = conv_input.transpose(1, 2)
        conv_out = self.conv_pointwise1(conv_input)
        conv_out = self.conv_activation(conv_out)
        conv_out = self.conv_depthwise(conv_out)
        conv_out = self.conv_batchnorm(conv_out)
        conv_out = F.silu(conv_out)
        conv_out = self.conv_pointwise2(conv_out)
        conv_out = self.conv_dropout(conv_out).transpose(1, 2)
        x = x + conv_out

        return self.final_norm(x)


class SelfAttentivePooling(nn.Module):
    """Multi-head attention pooling over time dimension."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.context = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, time, dim = x.size()
        context = self.context.expand(batch_size, 1, dim)
        attn_out, _ = self.attention(context, x, x, key_padding_mask=self._make_padding_mask(lengths, time))
        pooled = attn_out.squeeze(1)
        stats = torch.cat([pooled, x.mean(dim=1)], dim=-1)
        return stats

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]


class AccentEncoder(nn.Module):
    """End-to-end accent encoder with classifiers and GRL-friendly head."""

    def __init__(
        self,
        input_dim: int = 80,
        model_dim: int = 256,
        num_blocks: int = 4,
        accent_classes: int = 2,
        speaker_classes: int = 100,
        emb_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [ConformerBlock(model_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.pool = SelfAttentivePooling(model_dim, num_heads=4)
        pooled_dim = model_dim * 2
        self.embedding = nn.Sequential(
            nn.Linear(pooled_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.accent_classifier = nn.Linear(emb_dim, accent_classes)
        self.speaker_classifier = nn.Linear(emb_dim, speaker_classes)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.arange(mel.size(1), device=mel.device)[None, :] < lengths[:, None]
        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x, padding_mask=mask)
        stats = self.pool(x, lengths)
        embedding = F.normalize(self.embedding(stats), dim=-1)
        accent_logits = self.accent_classifier(embedding)
        speaker_logits = self.speaker_classifier(embedding)
        return embedding, accent_logits, speaker_logits


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)
