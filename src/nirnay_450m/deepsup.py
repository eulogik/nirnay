"""Deep supervision at encoder layers 4/8/12 (plan §1/§2, V-JEPA trick).

L = ... + 0.2*L_deep where L_deep is the mean auxiliary CE over the
designated intermediate layers. Non-designated layers are ignored even if
their states are supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEEP_LAYERS: tuple[int, ...] = (4, 8, 12)
DEEP_WEIGHT: float = 0.2


class DeepSupervision(nn.Module):
    """Per-layer linear probes on selected encoder layers → aux CE loss."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        layers: Sequence[int] = DEEP_LAYERS,
        weight: float = DEEP_WEIGHT,
    ):
        super().__init__()
        if num_labels < 2:
            raise ValueError("num_labels must be >= 2")
        self.layers = tuple(int(i) for i in layers)
        self.weight = float(weight)
        self.num_labels = int(num_labels)
        self.probes = nn.ModuleDict(
            {
                str(i): nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, self.num_labels),
                )
                for i in self.layers
            }
        )

    def forward(
        self,
        hidden_by_layer: Mapping[int, torch.Tensor],
        targets: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """hidden_by_layer: layer_idx → [B, L, H] (pooled by mean over L).

        Returns (weighted_total, {layer: unweighted_ce}). Total equals
        weight * mean(per-layer CE) exactly.
        """
        if targets.ndim != 1:
            raise ValueError("targets must be [B]")
        per_layer: dict[int, torch.Tensor] = {}
        for idx in self.layers:
            if idx not in hidden_by_layer:
                raise KeyError(f"missing hidden state for layer {idx}")
            h = hidden_by_layer[idx]
            if h.ndim == 3:
                if attention_mask is None:
                    h = h.mean(dim=1)
                else:
                    mask = attention_mask.to(h.dtype).unsqueeze(-1)
                    h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            logits = self.probes[str(idx)](h)
            per_layer[idx] = F.cross_entropy(logits, targets)
        mean_ce = torch.stack([per_layer[i] for i in self.layers]).mean()
        total = self.weight * mean_ce
        return total, per_layer
