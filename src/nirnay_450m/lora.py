"""Minimal LoRA for Phase A (plan §2: LoRA on the encoder).

No external peft dependency — a thin Linear wrapper: base weights frozen,
low-rank A/B trainable. Applied to encoder projection linears by name
pattern; safe to call twice (idempotent per module).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

LORA_TARGET_SUFFIXES = ("Wqkv", "Wo", "Wi", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    """y = base(x) + scale · x Aᵀ Bᵀ with base frozen."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        lora = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return h + self.scaling * lora


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_suffixes: tuple[str, ...] = LORA_TARGET_SUFFIXES,
) -> int:
    """Wrap matching nn.Linear children in place. Returns count wrapped."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and (
            name in target_suffixes or name.endswith(target_suffixes)
        ):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += apply_lora(child, rank, alpha, target_suffixes)
    return replaced


def lora_parameters(module: nn.Module):
    for p in module.parameters():
        if p.requires_grad and any(
            n.endswith(("lora_A", "lora_B"))
            for n, q in module.named_parameters()
            if q is p
        ):
            yield p


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_frozen(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)
