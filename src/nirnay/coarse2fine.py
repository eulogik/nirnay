"""2-stage coarse-to-fine choice scoring (plan §1 item 3; Banking77 path).

Stage 1: retrieve top-k candidates (default k=20) from the full label bank
(77+ options) — cosine bank retrieval by default; the training path passes
dense [MASK]-scorer top-k as `candidates` instead (stronger zero-shot prior,
trained through `logit_bias`). Stage 2: pointer softmax over the retrieved
set only — fixes the 192–256 tok head budget collapse Laya showed at 77-way.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TOP_K = 20  # plan §1: "Stage-1: pico-coarse retrieve top-20 from 77+"


def stage1_retrieve(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int = DEFAULT_TOP_K,
) -> torch.Tensor:
    """Cosine top-k label indices. query: [B, D], bank: [N, D] → [B, k]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, bank.size(0))
    q = F.normalize(query, dim=-1)
    b = F.normalize(bank, dim=-1)
    sims = q @ b.t()  # [B, N]
    return sims.topk(k, dim=-1).indices


def stage2_pointer(
    query: torch.Tensor,
    candidates: torch.Tensor,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pointer distribution over retrieved candidates.

    query: [B, D], candidates: [B, K, D] → probs [B, K] (sums to 1).
    logit_bias: optional [B, K] additive scores (e.g. [MASK]-scorer logits)
    fused with the cosine similarity.
    """
    q = F.normalize(query, dim=-1)
    c = F.normalize(candidates, dim=-1)
    scores = torch.einsum("bd,bkd->bk", q, c)
    if logit_bias is not None:
        scores = scores + logit_bias
    return F.softmax(scores, dim=-1)


class CoarseToFine(nn.Module):
    """Embeds a label bank once; retrieve → pointer in two stages."""

    def __init__(self, hidden_size: int, num_labels: int, top_k: int = DEFAULT_TOP_K):
        super().__init__()
        if num_labels < 2:
            raise ValueError("num_labels must be >= 2")
        self.top_k = int(top_k)
        self.num_labels = int(num_labels)
        self.label_embed = nn.Embedding(num_labels, hidden_size)
        nn.init.normal_(self.label_embed.weight, std=0.02)

    @property
    def bank(self) -> torch.Tensor:
        return self.label_embed.weight

    def forward(
        self,
        query: torch.Tensor,
        gold_idx: torch.Tensor | None = None,
        *,
        candidates: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """query: [B, H]. Returns cand_indices [B, k], probs [B, k],
        full_probs [B, N] (zeros outside candidates). If gold_idx given,
        includes gold in candidates (teacher-forcing retrieve for training).

        candidates: optional precomputed [B, k] indices (train path passes the
        dense [MASK]-scorer top-k here — differentiably trained via logit_bias;
        default None keeps the cosine stage-1 retrieval). logit_bias: optional
        [B, k] scorer logits fused into the pointer score.
        """
        if candidates is None:
            cand = stage1_retrieve(query, self.bank, k=self.top_k)
        else:
            cand = candidates
        if gold_idx is not None:
            gold = gold_idx.view(-1, 1)
            # Ensure gold present: replace last slot when missing.
            present = (cand == gold).any(dim=-1, keepdim=True)
            cand = torch.where(present, cand, torch.cat([cand[:, :-1], gold], dim=-1))
        candidates = self.label_embed(cand)  # [B, K, H]
        probs = stage2_pointer(query, candidates, logit_bias=logit_bias)
        full = torch.zeros(
            query.size(0), self.num_labels, device=query.device, dtype=probs.dtype
        )
        full = full.scatter(1, cand, probs)
        return {"cand_indices": cand, "probs": probs, "full_probs": full}
