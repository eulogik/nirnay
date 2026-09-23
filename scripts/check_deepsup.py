"""G9: deep supervision at layers 4/8/12 only; total == 0.2 * mean CE."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.deepsup import DEEP_LAYERS, DEEP_WEIGHT, DeepSupervision

    torch.manual_seed(0)
    H, K, B, L = 64, 10, 8, 12
    ds = DeepSupervision(hidden_size=H, num_labels=K).eval()

    if ds.layers != DEEP_LAYERS or abs(ds.weight - DEEP_WEIGHT) > 1e-9:
        print(f"DEEPSUP_FAIL layers={ds.layers} weight={ds.weight}", file=sys.stderr)
        return 1
    if set(ds.probes.keys()) != {str(i) for i in DEEP_LAYERS}:
        print(f"DEEPSUP_FAIL probes={list(ds.probes.keys())}", file=sys.stderr)
        return 1

    targets = torch.randint(0, K, (B,))
    hidden_by_layer = {i: torch.randn(B, L, H) for i in (4, 5, 8, 12, 13)}

    total, per_layer = ds(hidden_by_layer, targets)
    if set(per_layer.keys()) != set(DEEP_LAYERS):
        print(f"DEEPSUP_FAIL per_layer_keys={sorted(per_layer)}", file=sys.stderr)
        return 1
    if total.ndim != 0 or not torch.isfinite(total) or float(total.detach()) <= 0.0:
        print(f"DEEPSUP_FAIL total={total}", file=sys.stderr)
        return 1

    mean_ce = torch.stack([per_layer[i] for i in DEEP_LAYERS]).mean()
    expected = DEEP_WEIGHT * mean_ce
    if abs(float(total) - float(expected)) > 1e-6:
        print(
            f"DEEPSUP_FAIL weight_mismatch total={float(total)} "
            f"expected={float(expected)}",
            file=sys.stderr,
        )
        return 1

    # Non-selected layers (5, 13) must be ignored: perturb them → same total.
    hidden_perturbed = dict(hidden_by_layer)
    hidden_perturbed[5] = hidden_by_layer[5] * 10.0 + 5.0
    hidden_perturbed[13] = -hidden_by_layer[13]
    total2, _ = ds(hidden_perturbed, targets)
    if abs(float(total2) - float(total)) > 1e-6:
        print(
            f"DEEPSUP_FAIL nonselected_layer_changed delta={float(total2 - total)}",
            file=sys.stderr,
        )
        return 1

    # Missing a selected layer must raise (cannot silently skip 4/8/12).
    try:
        ds({4: hidden_by_layer[4], 8: hidden_by_layer[8]}, targets)
        print("DEEPSUP_FAIL missing_layer_not_raised", file=sys.stderr)
        return 1
    except KeyError:
        pass

    print(
        f"DEEPSUP_OK layers={ds.layers} weight={ds.weight} "
        f"total==0.2*mean_ce nonselected_ignored=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
