"""Byte path: ByteEmbed + Conv stack + coding-rate patching.

Architecture follows pico-type (github.com/eulogik/pico-type, Apache-2.0):
  ByteEmbed(256→96) → 3×Conv(k=3,5,7) → patch → project to encoder hidden.

Coding-rate patching groups bytes into fixed patches for the encoder state
segment; full ByteFlow Top-K boundary detection is v1.1 (plan §5 Days 61–90
owns the 32k sparse path — v1 wires the path, not the long-ctx router).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BytePathConfig:
    embed_dim: int = 96
    trunk_dim: int = 192
    conv_kernels: tuple[int, ...] = (3, 5, 7)
    patch_size: int = 8
    encoder_hidden: int = 1024
    max_bytes: int = 4096
    dropout: float = 0.1


class ConvBlock(nn.Module):
    """Depthwise-style Conv1d + LN + GELU + residual (pico-type ConvBlock)."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = (
            nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self.conv(x).transpose(1, 2)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h.transpose(1, 2) + residual


class BytePath(nn.Module):
    """bytes [B, L] → patches [B, n_patches, encoder_hidden]."""

    def __init__(self, config: BytePathConfig | None = None):
        super().__init__()
        self.config = config or BytePathConfig()
        cfg = self.config
        self.embed = nn.Embedding(256, cfg.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        in_dim = cfg.embed_dim
        self.convs = nn.ModuleList()
        for k in cfg.conv_kernels:
            self.convs.append(ConvBlock(in_dim, cfg.trunk_dim, k, cfg.dropout))
            in_dim = cfg.trunk_dim
        self.patch_size = cfg.patch_size
        self.to_encoder = nn.Linear(cfg.trunk_dim, cfg.encoder_hidden)
        # Zero-init the encoder projection: ByteFusion is exactly identity at
        # construction (random-init byte patches were corrupting laya's
        # pretrained embeddings — diagnosis 2026-09-25). Gradients still reach
        # to_encoder through the nonzero fusion scale from step 0.
        nn.init.zeros_(self.to_encoder.weight)
        nn.init.zeros_(self.to_encoder.bias)
        # Normalize trunk features BEFORE the zero-init projection. A LayerNorm
        # after the projection normalizes the tiny first optimizer step to unit
        # variance, injecting full-strength input corruption at any learning
        # rate (one-step bisection, 2026-09-25: byte alone crashed top20
        # 0.5844 -> 0.2208). With norm-before-proj, injection magnitude scales
        # with to_encoder weights (i.e. lr x steps) and starts at 0.
        self.norm = nn.LayerNorm(cfg.trunk_dim)

    def forward(self, byte_ids: torch.Tensor, byte_mask: torch.Tensor | None = None) -> torch.Tensor:
        """byte_ids: [B, L] int64 in [0,255]. Returns [B, n_patches, H]."""
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.long()
        if byte_ids.size(1) > self.config.max_bytes:
            byte_ids = byte_ids[:, : self.config.max_bytes]
            if byte_mask is not None:
                byte_mask = byte_mask[:, : self.config.max_bytes]

        x = self.embed(byte_ids).transpose(1, 2)
        for block in self.convs:
            x = block(x)
        x = x.transpose(1, 2)  # [B, L, trunk]

        # Fixed-size coding-rate patches: group consecutive trunk vectors.
        B, L, D = x.shape
        ps = self.patch_size
        pad = (ps - L % ps) % ps
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            if byte_mask is not None:
                byte_mask = torch.nn.functional.pad(byte_mask, (0, pad), value=0)
        Lp = x.size(1)
        x = x.view(B, Lp // ps, ps, D).mean(dim=2)  # [B, n_patches, D]
        if byte_mask is not None:
            m = byte_mask.view(B, Lp // ps, ps).sum(dim=-1, keepdim=True).float()
            m = (m > 0).float()
            x = x * m

        return self.to_encoder(self.norm(x))


class ByteFusion(nn.Module):
    def __init__(self, config: BytePathConfig | None = None):
        super().__init__()
        self.config = config or BytePathConfig()
        self.path = BytePath(self.config)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        byte_ids: torch.Tensor | None,
        byte_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if byte_ids is None:
            return token_embeddings
        # No LayerNorm after path output: norm-after-projection normalizes any
        # nonzero to_encoder output to unit variance (see BytePath.norm note).
        patches = self.path(byte_ids, byte_mask)
        # Unconditional interpolate (no `if patches.size(1) != ...` guard):
        # linear interpolate to the same size is bit-exact identity, and a
        # Python branch on symbolic shapes blocks ONNX export. Patches are
        # always 64 wide (512 bytes / patch 8) while states vary, so the old
        # guard took this path on every real input anyway.
        patches = nn.functional.interpolate(
            patches.transpose(1, 2),
            size=token_embeddings.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        if byte_mask is not None:
            valid = byte_mask.to(torch.bool).any(dim=1, keepdim=True)
            patches = patches * valid.unsqueeze(-1).to(patches.dtype)
        return token_embeddings + self.scale * patches


def encode_bytes(data: bytes, max_len: int = 4096, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """bytes → (ids [1, max_len], mask [1, max_len]) like pico-type.encode_bytes."""
    raw = list(data[:max_len])
    ids = raw + [pad] * (max_len - len(raw))
    mask = [1] * len(raw) + [0] * (max_len - len(raw))
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
    )
