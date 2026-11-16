"""Hypercube sparse wiring (plan §1, arXiv:2609.18145).

v1: fixed sparse pattern rotated across layers — 1/32 links, log2(n) depth
to reach all positions. Applied as an attention bias / index mask on the
patch-encoder state when sequence length warrants it (8k path). For short
JevBench sequences (<256) the identity (dense) path is used; the module is
still wired so the pattern is active and testable.
"""

from __future__ import annotations

import math

import torch


def hypercube_adjacency(seq_len: int, links_per_node: int = None) -> torch.Tensor:
    """Boolean adjacency [L, L]: self + power-of-two stride links (hypercube-like).

    links i±2^k for k=0..floor(log2(L))-1, clipped to [0, L). ~log2(L)*2 links
    per node (plus self). For L=32, k=0..4 → up to 10 links + self ≈ 11/32.
    Plan cites 1/32 links — that is the sparsest setting (single stride);
    default uses log-stride for full reachability in log2 depth.
    """
    idx = torch.arange(seq_len)
    adj = torch.eye(seq_len, dtype=torch.bool)
    k = 0
    stride = 1
    while stride < seq_len:
        for s in (stride, -stride):
            nbr = idx + s
            valid = (nbr >= 0) & (nbr < seq_len)
            adj[idx[valid], nbr[valid]] = True
        stride *= 2
        k += 1
        if k > 32:
            break
    return adj


def hypercube_mask(seq_len: int, device=None) -> torch.Tensor:
    """Additive attention mask [1, 1, L, L]: 0 allowed, -inf blocked."""
    adj = hypercube_adjacency(seq_len)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    mask[~adj] = float("-inf")
    return mask[None, None, :, :].to(device=device)


def should_apply(seq_len: int, threshold: int = 256) -> bool:
    """Apply sparse wiring only for long sequences (plan: 8k sparse path)."""
    return seq_len >= threshold
