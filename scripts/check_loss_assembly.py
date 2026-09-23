"""G14: plan §2 loss assembly — exact 0.3 / 0.2 / λ coefficients on fixed tensors."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.losses import (
        CAL_LAMBDA_DEFAULT,
        DEEP_COEF,
        NCP_COEF,
        assemble_plan_loss,
        cal_ce,
        choice_ce,
        noul_bce,
        rps_loss,
    )

    torch.manual_seed(0)

    # --- component definitions finite ---
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.bool)
    targets = torch.tensor([0, 2])
    c = choice_ce(logits, targets, mask)
    if not torch.isfinite(c) or float(c) < 0:
        print(f"LOSS_FAIL choice_ce={c}", file=sys.stderr)
        return 1

    probs = torch.softmax(torch.randn(8, 5), dim=-1)
    s_targets = torch.randint(0, 5, (8,))
    r = rps_loss(probs, s_targets)
    if not torch.isfinite(r) or float(r) < 0 or float(r) > 1.0 + 1e-6:
        print(f"LOSS_FAIL rps={r}", file=sys.stderr)
        return 1

    n_probs = torch.softmax(torch.randn(6, 2), dim=-1)
    n_targets = torch.randint(0, 2, (6,))
    nb = noul_bce(n_probs, n_targets)
    if not torch.isfinite(nb) or float(nb) < 0:
        print(f"LOSS_FAIL noul={nb}", file=sys.stderr)
        return 1

    log_p = torch.log(torch.softmax(torch.randn(8, 77), dim=-1).clamp_min(1e-12))
    lab = torch.randint(0, 77, (8,))
    cal = cal_ce(log_p, lab)
    if not torch.isfinite(cal) or float(cal) < 0:
        print(f"LOSS_FAIL cal={cal}", file=sys.stderr)
        return 1

    # --- exact plan coefficients ---
    # Fixed scalars (as detached tensors) so hand-check is trivial.
    choice_t = torch.tensor(1.5)
    rps_t = torch.tensor(0.25)
    noul_t = torch.tensor(0.4)
    rel_t = torch.tensor(0.1)
    ncp_t = torch.tensor(2.0)   # * 0.3 → 0.6
    deep_t = torch.tensor(3.0)  # * 0.2 → 0.6
    cal_t = torch.tensor(10.0)  # * 0.005 → 0.05

    parts = assemble_plan_loss(
        choice_t, rps_t, noul_t, rel_t, ncp_t, deep_t, cal_t
    )
    expected = 1.5 + 0.25 + 0.4 + 0.1 + 0.3 * 2.0 + 0.2 * 3.0 + 0.005 * 10.0
    # expected = 1.5+0.25+0.4+0.1+0.6+0.6+0.05 = 3.5
    if abs(float(parts.total) - expected) > 1e-6:
        print(
            f"LOSS_FAIL total={float(parts.total)} expected={expected}",
            file=sys.stderr,
        )
        return 1
    if parts.ncp_coef != 0.3 or parts.deep_coef != 0.2:
        print(
            f"LOSS_FAIL coef ncp={parts.ncp_coef} deep={parts.deep_coef}",
            file=sys.stderr,
        )
        return 1
    if abs(parts.cal_lambda - CAL_LAMBDA_DEFAULT) > 1e-12:
        print(f"LOSS_FAIL lambda={parts.cal_lambda}", file=sys.stderr)
        return 1
    if abs(NCP_COEF - 0.3) > 1e-12 or abs(DEEP_COEF - 0.2) > 1e-12:
        print("LOSS_FAIL plan_constants", file=sys.stderr)
        return 1

    # Wrong coefficients must be rejected (plan is frozen).
    try:
        assemble_plan_loss(choice_t, rps_t, noul_t, rel_t, ncp_t, deep_t, cal_t, ncp_coef=0.5)
        print("LOSS_FAIL bad_ncp_accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass
    try:
        assemble_plan_loss(choice_t, rps_t, noul_t, rel_t, ncp_t, deep_t, cal_t, deep_coef=0.9)
        print("LOSS_FAIL bad_deep_accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass

    # Gradients flow through assembled total to learnable leaves
    ncp_g = torch.tensor(2.0, requires_grad=True)
    deep_g = torch.tensor(3.0, requires_grad=True)
    parts2 = assemble_plan_loss(c, r, nb, rel_t, ncp_g, deep_g, cal)
    parts2.total.backward()
    if ncp_g.grad is None or deep_g.grad is None:
        print("LOSS_FAIL no_grad", file=sys.stderr)
        return 1
    if abs(float(ncp_g.grad) - 0.3) > 1e-6 or abs(float(deep_g.grad) - 0.2) > 1e-6:
        print(
            f"LOSS_FAIL grad_coef ncp={float(ncp_g.grad)} deep={float(deep_g.grad)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"LOSS_OK total_repro={float(parts.total):.4f} ncp_coef=0.3 deep_coef=0.2 "
        f"lambda={parts.cal_lambda} grad_ncp=0.3 grad_deep=0.2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
