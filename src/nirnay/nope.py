"""NoPE head-masking (v1: mask only; full GLA split is v1.1).

Plan §1: "v1: NoPE masking on 1/3 global-retrieval heads (local heads keep
Laya RoPE as-is)".

ModernBERT (transformers 5.x) applies RoPE via
`apply_rotary_pos_emb(q, k, cos, sin)` where cos/sin are [B, L, D] and are
unsqueezed to broadcast across all heads. To give the first `nope_fraction`
of heads identity rotation (NoPE) while local heads keep RoPE, we patch
`apply_rotary_pos_emb` in the modernbert modeling module to expand cos/sin
per-head and neutralize the NoPE slice. No parameters are added or removed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def split_head_range(num_heads: int, nope_fraction: float = 1.0 / 3.0) -> tuple[int, int]:
    """(nope_end, local_start): heads [0, nope_end) are NoPE; rest keep RoPE."""
    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if num_heads == 1:
        return 1, 1
    nope_end = max(1, int(round(num_heads * nope_fraction)))
    nope_end = min(nope_end, num_heads - 1)
    return nope_end, nope_end


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _make_nope_apply(nope_end: int, num_heads: int):
    def apply_rotary_pos_emb_nope(q, k, cos, sin, unsqueeze_dim=1):
        original_dtype = q.dtype
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        H = q.size(1)
        if cos.size(1) == 1 and H > 1:
            cos = cos.expand(cos.size(0), H, cos.size(2), cos.size(3)).clone()
            sin = sin.expand(sin.size(0), H, sin.size(2), sin.size(3)).clone()
            end = min(nope_end, H)
            cos[:, :end] = 1.0
            sin[:, :end] = 0.0
        q_embed = (q.float() * cos) + (_rotate_half(q.float()) * sin)
        k_embed = (k.float() * cos) + (_rotate_half(k.float()) * sin)
        return q_embed.to(original_dtype), k_embed.to(original_dtype)

    return apply_rotary_pos_emb_nope


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def apply_nope_mask(encoder: nn.Module, nope_fraction: float = 1.0 / 3.0) -> dict:
    """Apply NoPE head-masking to a ModernBERT encoder in-place.

    Returns a report dict. `params_unchanged` is True iff param count before
    == after (mask adds no parameters).
    """
    before = param_count(encoder)

    num_heads = None
    if hasattr(encoder, "config") and hasattr(encoder.config, "num_attention_heads"):
        num_heads = int(encoder.config.num_attention_heads)
    if num_heads is None:
        for m in encoder.modules():
            nh = getattr(m, "num_heads", None)
            if isinstance(nh, int) and nh > 0:
                num_heads = nh
                break

    report = {
        "nope_fraction": nope_fraction,
        "num_heads": num_heads,
        "masked": False,
        "method": "unavailable",
        "params_unchanged": False,
    }
    if num_heads is None:
        report["params_unchanged"] = param_count(encoder) == before
        return report

    nope_end, _ = split_head_range(num_heads, nope_fraction)
    report["nope_end"] = nope_end

    try:
        import transformers.models.modernbert.modeling_modernbert as mb

        mb.apply_rotary_pos_emb = _make_nope_apply(nope_end, num_heads)  # type: ignore[assignment]
        report["masked"] = True
        report["method"] = "patch_modernbert_apply_rotary_pos_emb"
    except Exception as e:  # noqa: BLE001
        report["method"] = f"failed:{type(e).__name__}"
        report["error"] = str(e)[:200]

    after = param_count(encoder)
    report["params_unchanged"] = after == before
    report["params"] = after
    return report
