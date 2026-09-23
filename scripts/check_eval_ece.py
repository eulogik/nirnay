"""G19: held-out eval — accuracy, Brier, raw ECE, and fitted ECE together."""

from __future__ import annotations

import math
import sys


def main() -> int:
    import numpy as np

    from nirnay.eval import evaluate_logits, evaluate_probs

    rng = np.random.default_rng(0)

    # --- synthetic calibrated-ish logits ---
    N, K = 400, 5
    true = rng.normal(size=(N, K))
    labels = true.argmax(axis=1)
    logits = true * 3.0 + rng.normal(size=(N, K)) * 0.5

    m = evaluate_logits(logits, labels)
    if m.n != N:
        print(f"EVAL_ECE_FAIL n={m.n}", file=sys.stderr)
        return 1
    for name, v in (
        ("accuracy", m.accuracy),
        ("brier", m.brier),
        ("ece_raw", m.ece_raw),
        ("ece_fitted", m.ece_fitted),
        ("temperature", m.temperature),
        ("mean_conf", m.mean_conf),
    ):
        if v != v or math.isinf(v):
            print(f"EVAL_ECE_FAIL {name}={v}", file=sys.stderr)
            return 1
    if not (0.0 <= m.accuracy <= 1.0):
        print(f"EVAL_ECE_FAIL acc={m.accuracy}", file=sys.stderr)
        return 1
    if not (0.0 <= m.brier <= 2.0 + 1e-6):
        print(f"EVAL_ECE_FAIL brier={m.brier}", file=sys.stderr)
        return 1
    if not (0.0 <= m.ece_raw <= 1.0 + 1e-6):
        print(f"EVAL_ECE_FAIL ece_raw={m.ece_raw}", file=sys.stderr)
        return 1
    if not (0.0 <= m.ece_fitted <= 1.0 + 1e-6):
        print(f"EVAL_ECE_FAIL ece_fitted={m.ece_fitted}", file=sys.stderr)
        return 1
    if not (0.5 <= m.temperature <= 5.0):
        print(f"EVAL_ECE_FAIL temp={m.temperature}", file=sys.stderr)
        return 1

    # Perfect predictions → acc=1, ece≈0, brier≈0
    perfect = np.full((32, K), -10.0)
    perfect[np.arange(32), labels[:32] if labels.size >= 32 else labels] = 10.0
    lab_p = labels[:32] if labels.size >= 32 else labels
    # rebuild cleanly
    lab_p = rng.integers(0, K, size=32)
    perfect = np.full((32, K), -10.0)
    perfect[np.arange(32), lab_p] = 10.0
    mp = evaluate_logits(perfect, lab_p)
    if mp.accuracy < 0.99 or mp.ece_raw > 0.05 or mp.brier > 0.05:
        print(
            f"EVAL_ECE_FAIL perfect acc={mp.accuracy} ece={mp.ece_raw} brier={mp.brier}",
            file=sys.stderr,
        )
        return 1

    # Fitted ECE must be present alongside raw (never drop a column)
    d = m.as_dict()
    for key in ("ece_raw", "ece_fitted", "accuracy", "brier"):
        if key not in d:
            print(f"EVAL_ECE_FAIL missing_key={key}", file=sys.stderr)
            return 1

    # Overconfident wrong: raw ECE high; after fit still reports both
    over = np.full((100, 3), -5.0)
    over[:, 0] = 5.0  # always predicts class 0
    lab_o = np.zeros(100, dtype=np.int64)
    lab_o[50:] = 1  # half wrong
    mo = evaluate_logits(over, lab_o)
    if mo.ece_raw < 0.1:
        print(f"EVAL_ECE_FAIL overconf_ece_raw={mo.ece_raw}", file=sys.stderr)
        return 1

    print(
        f"EVAL_ECE_OK n={m.n} acc={m.accuracy:.4f} brier={m.brier:.4f} "
        f"ece_raw={m.ece_raw:.4f} ece_fitted={m.ece_fitted:.4f} T={m.temperature:.3f} "
        f"perfect_acc={mp.accuracy:.3f} overconf_raw={mo.ece_raw:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
