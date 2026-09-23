"""G6: temperature refit writes finite temps in [0.5, 5.0] for >=4 buckets."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    from nirnay.temps import TEMP_MAX, TEMP_MIN, fit_all_buckets, write_temps

    rng = np.random.default_rng(0)
    records = []
    # Four+ buckets with known optimal-ish scale (logits too sharp → T>1).
    specs = [
        ("choice", 2, 8.0),
        ("choice", 4, 5.0),
        ("score", 3, 3.0),
        ("noul", 2, 6.0),
        ("choice", 12, 4.0),
    ]
    for qtype, k, scale in specs:
        for _ in range(64):
            true = rng.normal(size=k)
            logits = true * scale + rng.normal(size=k) * 0.1
            target = np.zeros(k)
            target[int(np.argmax(true))] = 1.0
            records.append(
                {"qtype": qtype, "k": k, "logits": logits.tolist(), "target": target.tolist()}
            )

    temps = fit_all_buckets(records)
    if len(temps) < 4:
        print(f"TEMPS_FAIL buckets={len(temps)}", file=sys.stderr)
        return 1
    for k, v in temps.items():
        if not (TEMP_MIN <= v <= TEMP_MAX) or not np.isfinite(v):
            print(f"TEMPS_FAIL {k}={v}", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "temperature_by_options.json"
        write_temps(path, temps)
        if not path.exists():
            print("TEMPS_FAIL write", file=sys.stderr)
            return 1

    print(f"TEMPS_OK buckets_written={len(temps)} temps={ {k: round(v, 3) for k, v in sorted(temps.items())} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
