"""Model facade: byte path, NoPE, hypercube, SGDR, concepts, deepsup, RLCD, coarse2fine."""

from __future__ import annotations

from .bytes import BytePath, BytePathConfig, encode_bytes
from .concepts import PARAM_BUDGET, ConceptBottleneck, ConceptConfig, MoMESlots
from .coarse2fine import DEFAULT_TOP_K, CoarseToFine, stage1_retrieve, stage2_pointer
from .deepsup import DEEP_LAYERS, DEEP_WEIGHT, DeepSupervision
from .hypercube import hypercube_adjacency, hypercube_mask, should_apply
from .nope import apply_nope_mask, param_count, split_head_range
from .rlcd import (
    brier_reward,
    decision_token_ce,
    group_mean_advantages,
    rlcd_total_loss,
)
from .sgdr import SGDROUTERConfig, route_blocks

__all__ = [
    "BytePath",
    "BytePathConfig",
    "encode_bytes",
    "apply_nope_mask",
    "param_count",
    "split_head_range",
    "hypercube_adjacency",
    "hypercube_mask",
    "should_apply",
    "route_blocks",
    "SGDROUTERConfig",
    "ConceptBottleneck",
    "ConceptConfig",
    "MoMESlots",
    "PARAM_BUDGET",
    "DeepSupervision",
    "DEEP_LAYERS",
    "DEEP_WEIGHT",
    "brier_reward",
    "decision_token_ce",
    "group_mean_advantages",
    "rlcd_total_loss",
    "CoarseToFine",
    "stage1_retrieve",
    "stage2_pointer",
    "DEFAULT_TOP_K",
]
