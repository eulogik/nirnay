"""G4: NoPE mask active on designated heads; param count unchanged."""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModel

    from nirnay.nope import apply_nope_mask, param_count, split_head_range

    # Use ModernBERT config only (no full Laya download needed for mask logic).
    cfg = AutoConfig.from_pretrained("answerdotai/ModernBERT-large")
    # Build from config with random init — structure matches Laya encoder.
    enc = AutoModel.from_config(cfg)

    before = param_count(enc)
    report = apply_nope_mask(enc, nope_fraction=1.0 / 3.0)
    after = param_count(enc)

    nh = int(cfg.num_attention_heads)
    expect_end, _ = split_head_range(nh, 1.0 / 3.0)

    # Functional check: with NoPE patch, cos/sin for head 0 should be identity.
    import transformers.models.modernbert.modeling_modernbert as mb

    has_nope_fn = "nope" in type(mb.apply_rotary_pos_emb).__name__ or getattr(
        mb.apply_rotary_pos_emb, "__name__", ""
    ).endswith("_nope")

    # Positive control: run apply on a tiny tensor and confirm head-0 unchanged.
    B, H, L, D = 1, nh, 8, cfg.hidden_size // nh
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    cos = torch.rand(B, L, D)
    sin = torch.rand(B, L, D)
    q0, _ = mb.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    # Head 0 NoPE: q0[0,0] should equal q[0,0] (up to float cast path)
    head0_identity = torch.allclose(q0[0, 0], q[0, 0].float().to(q0.dtype), atol=1e-5)
    # A local head (last) should differ when sin/cos are non-trivial.
    head_local_changed = not torch.allclose(q0[0, -1], q[0, -1].float().to(q0.dtype), atol=1e-5)

    masked = bool(report.get("masked"))
    params_ok = before == after == report.get("params")
    nope_end_ok = report.get("nope_end") == expect_end

    if masked and params_ok and nope_end_ok and head0_identity and head_local_changed:
        print(
            f"NOPE_OK masked_heads>0 params_unchanged=true "
            f"nope_end={report.get('nope_end')}/{nh} method={report.get('method')}"
        )
        return 0
    print(
        f"NOPE_FAIL report={report} head0_identity={head0_identity} "
        f"local_changed={head_local_changed} has_nope_fn={has_nope_fn}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
