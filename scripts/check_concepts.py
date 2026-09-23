"""G8: concept bottleneck + MoME structural checks (codes, gate, NCP, budget)."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.concepts import PARAM_BUDGET, ConceptBottleneck, ConceptConfig

    torch.manual_seed(0)
    cfg = ConceptConfig()  # 1024→128 product VQ, chunk 4, 32 codes, M=4 slots
    module = ConceptBottleneck(cfg).eval()

    B, L, H = 2, 16, cfg.encoder_hidden
    hidden = torch.randn(B, L, H)
    out, info = module(hidden)

    # Shapes
    if out.shape != hidden.shape or not torch.isfinite(out).all():
        print(f"CONCEPTS_FAIL out_shape={tuple(out.shape)}", file=sys.stderr)
        return 1

    codes = info["codes"]
    if codes.shape != (B, L, cfg.num_chunks):
        print(f"CONCEPTS_FAIL codes_shape={tuple(codes.shape)}", file=sys.stderr)
        return 1
    if int(codes.min()) < 0 or int(codes.max()) >= cfg.codes_per_chunk:
        print(
            f"CONCEPTS_FAIL codes_range=[{int(codes.min())},{int(codes.max())}] "
            f"n_codes={cfg.codes_per_chunk}",
            file=sys.stderr,
        )
        return 1

    gate = info["gate_weights"]
    if gate.shape != (B, L, cfg.num_slots):
        print(f"CONCEPTS_FAIL gate_shape={tuple(gate.shape)}", file=sys.stderr)
        return 1
    row_sums = gate.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        print(f"CONCEPTS_FAIL gate_row_sum={row_sums.flatten()[:4].tolist()}",
              file=sys.stderr)
        return 1
    if float(gate.detach().min()) < 0.0 or float(gate.detach().max()) > 1.0:
        print("CONCEPTS_FAIL gate_range", file=sys.stderr)
        return 1

    ncp = info["ncp_loss"]
    if ncp.ndim != 0 or not torch.isfinite(ncp) or float(ncp) <= 0.0:
        print(f"CONCEPTS_FAIL ncp_loss={ncp}", file=sys.stderr)
        return 1

    # Gradient flows to encoder projection (training sanity).
    out.sum().backward()
    enc_grad = module.encode[0].weight.grad
    if enc_grad is None or not torch.isfinite(enc_grad).all():
        print("CONCEPTS_FAIL no encoder grad", file=sys.stderr)
        return 1

    params = module.param_count()
    if not (0 < params <= PARAM_BUDGET):
        print(f"CONCEPTS_FAIL params={params} budget={PARAM_BUDGET}", file=sys.stderr)
        return 1

    print(
        f"CONCEPTS_OK codes_range=[0,{cfg.codes_per_chunk}) chunks={cfg.num_chunks} "
        f"slots={cfg.num_slots} gate_rows_sum=1 ncp={float(ncp):.4f} "
        f"params={params}<={PARAM_BUDGET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
