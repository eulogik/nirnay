"""Per-(type, option-count) temperature fitting (plan §2 Calibration).

Fit one temperature per bucket on held-out logits; clamp to [0.5, 5.0]
(same bounds as Laya's clamp_temperature). Writes temperature_by_options.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TEMP_MIN = 0.5
TEMP_MAX = 5.0


def temp_bucket(qtype: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


def _nll_at_t(logits: np.ndarray, targets: np.ndarray, t: float) -> float:
    """Mean NLL of softmax(logits/t) against one-hot or soft targets."""
    z = logits / t
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=-1, keepdims=True)
    return float(-(targets * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=-1).mean())


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    lo: float = TEMP_MIN,
    hi: float = TEMP_MAX,
    grid: int = 60,
) -> float:
    """Grid-search scalar temperature minimizing NLL; clamp to [lo, hi]."""
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must be [N, K] and matching")
    best_t, best_loss = 1.0, _nll_at_t(logits, targets, 1.0)
    for t in np.linspace(lo, hi, grid):
        loss = _nll_at_t(logits, targets, float(t))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return float(min(hi, max(lo, best_t)))


def fit_all_buckets(
    records: list[dict],
) -> dict[str, float]:
    """records: each {qtype, k, logits: list[float], target: list[float]}."""
    groups: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for r in records:
        key = temp_bucket(r["qtype"], int(r["k"]))
        groups.setdefault(key, ([], []))
        groups[key][0].append(np.asarray(r["logits"], dtype=np.float64))
        groups[key][1].append(np.asarray(r["target"], dtype=np.float64))
    out: dict[str, float] = {}
    for key, (L, T) in groups.items():
        logits = np.stack(L, axis=0)
        target = np.stack(T, axis=0)
        out[key] = fit_temperature(logits, target)
    return out


def write_temps(path: str | Path, temps: dict[str, float]) -> Path:
    path = Path(path)
    for k, v in temps.items():
        if not math.isfinite(v) or not (TEMP_MIN <= v <= TEMP_MAX):
            raise ValueError(f"temperature {k}={v} out of range")
    path.write_text(json.dumps(temps, indent=2, sort_keys=True) + "\n")
    return path


def load_temps(path: str | Path) -> dict[str, float]:
    return json.loads(Path(path).read_text())
