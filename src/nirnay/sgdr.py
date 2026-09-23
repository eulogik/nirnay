"""SGDR block router (plan §1, arXiv:2609.22884).

Training-free long-range routing: pre-RoPE semantic pooling + offline
geometric prior → closed-form block scores. v1 wires a minimal router for
8k-patch paths; short sequences bypass (dense). The full paper's video
path and 128K gains are out of scope — plan targets 8k with measured probe
in Wk 1–2 (Days 1–14: structure + smoke, not tuned gains).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SGDROUTERConfig:
    block_size: int = 64
    top_k_blocks: int = 8
    use_geometric_prior: bool = True


def semantic_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool sequence into blocks: [B, L, D] → [B, n_blocks, D]."""
    B, L, D = x.shape
    pad = (block_size - L % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    n = x.size(1) // block_size
    return x.view(B, n, block_size, D).mean(dim=2)


def block_scores(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    geometric_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form routing scores [B, n_q_blocks, n_k_blocks]."""
    scores = torch.einsum("bqd,bkd->bqk", q_blocks, k_blocks) / (q_blocks.size(-1) ** 0.5)
    if geometric_prior is not None:
        scores = scores + geometric_prior
    return scores


def geometric_distance_prior(n_blocks: int, decay: float = 0.1) -> torch.Tensor:
    """Offline relative block-distance bias [n_blocks, n_blocks]."""
    i = torch.arange(n_blocks).float()
    dist = (i[:, None] - i[None, :]).abs()
    return -decay * dist


def route_blocks(
    x: torch.Tensor,
    config: SGDROUTERConfig | None = None,
) -> torch.Tensor:
    """Return keep-mask [B, L] marking tokens in top-k blocks for each query mix.

    v1 smoke: scores all blocks against mean query, keeps global top-k blocks
    (covers long-range); returns boolean mask over original length.
    """
    cfg = config or SGDROUTERConfig()
    B, L, D = x.shape
    blocks = semantic_pool(x, cfg.block_size)  # [B, n, D]
    n = blocks.size(1)
    q = blocks.mean(dim=1, keepdim=True)  # [B, 1, D]
    prior = geometric_distance_prior(n) if cfg.use_geometric_prior else None
    scores = block_scores(q, blocks, prior)  # [B, 1, n]
    k = min(cfg.top_k_blocks, n)
    top = scores.topk(k, dim=-1).indices  # [B, 1, k]
    keep_blocks = torch.zeros(B, n, dtype=torch.bool, device=x.device)
    keep_blocks.scatter_(1, top.squeeze(1), True)
    # Expand blocks → tokens
    token_keep = torch.arange(L, device=x.device)[None, :] // cfg.block_size
    token_keep = token_keep.clamp(max=n - 1)
    return keep_blocks.gather(1, token_keep)
