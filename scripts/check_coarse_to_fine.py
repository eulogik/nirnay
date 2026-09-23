"""G11: 2-stage coarse-to-fine — gold in top-k, valid pointer, k≤20 default."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.coarse2fine import (
        DEFAULT_TOP_K,
        CoarseToFine,
        stage1_retrieve,
        stage2_pointer,
    )

    torch.manual_seed(0)
    N, D, B = 77, 32, 16  # Banking77-sized bank
    if not (1 <= DEFAULT_TOP_K <= 20):
        print(f"COARSE2FINE_FAIL default_k={DEFAULT_TOP_K}", file=sys.stderr)
        return 1

    # Separable fixture: bank rows are unit axes-ish; queries = gold rows.
    bank = torch.randn(N, D)
    gold_idx = torch.randint(0, N, (B,))
    query = bank[gold_idx] + 0.01 * torch.randn(B, D)

    cand = stage1_retrieve(query, bank, k=DEFAULT_TOP_K)
    if cand.shape != (B, DEFAULT_TOP_K):
        print(f"COARSE2FINE_FAIL cand_shape={tuple(cand.shape)}", file=sys.stderr)
        return 1
    # Gold must be retrieved for every query (nearest to itself).
    hit = (cand == gold_idx.unsqueeze(-1)).any(dim=-1)
    if not bool(hit.all()):
        miss = (~hit).nonzero().flatten().tolist()
        print(f"COARSE2FINE_FAIL gold_misses={miss}", file=sys.stderr)
        return 1

    # Stage-2 pointer over retrieved candidates only.
    cand_embs = bank[cand]  # [B, k, D]
    probs = stage2_pointer(query, cand_embs)
    if probs.shape != (B, DEFAULT_TOP_K):
        print(f"COARSE2FINE_FAIL probs_shape={tuple(probs.shape)}", file=sys.stderr)
        return 1
    sums = probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        print(f"COARSE2FINE_FAIL probs_sum={sums[:4].tolist()}", file=sys.stderr)
        return 1
    if float(probs.min()) < -1e-6 or float(probs.max()) > 1.0 + 1e-6:
        print("COARSE2FINE_FAIL probs_range", file=sys.stderr)
        return 1

    # Separable fixture: argmax over candidates must be gold.
    arg_local = probs.argmax(dim=-1)
    arg_global = cand.gather(1, arg_local.unsqueeze(-1)).squeeze(-1)
    if not torch.equal(arg_global, gold_idx):
        bad = (arg_global != gold_idx).nonzero().flatten().tolist()
        print(f"COARSE2FINE_FAIL argmax_misses={bad}", file=sys.stderr)
        return 1

    # Module path: full_probs valid, mass only on candidates, default k≤20.
    m = CoarseToFine(hidden_size=D, num_labels=N)
    out = m(query, gold_idx=gold_idx)
    full = out["full_probs"]
    if full.shape != (B, N):
        print(f"COARSE2FINE_FAIL full_shape={tuple(full.shape)}", file=sys.stderr)
        return 1
    full_sums = full.sum(dim=-1)
    if not torch.allclose(full_sums, torch.ones_like(full_sums), atol=1e-5):
        print(f"COARSE2FINE_FAIL full_sum={full_sums[:4].tolist()}", file=sys.stderr)
        return 1
    # Gold probability must be > 0 (teacher-forced ensure) and peak on fixture.
    gold_p = full.gather(1, gold_idx.unsqueeze(-1)).squeeze(-1)
    if float(gold_p.detach().min()) <= 0.0:
        print(f"COARSE2FINE_FAIL gold_p_min={float(gold_p.min())}", file=sys.stderr)
        return 1
    if m.top_k > 20:
        print(f"COARSE2FINE_FAIL module_k={m.top_k}", file=sys.stderr)
        return 1

    print(
        f"COARSE2FINE_OK n_labels={N} default_k={DEFAULT_TOP_K} "
        f"gold_in_topk={B}/{B} pointer_sums=1 argmax=gold full_sums=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
