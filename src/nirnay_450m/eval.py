"""Held-out evaluation (plan §2/§3): accuracy, Brier, raw AND fitted ECE.

Never report fitted-only (plan §0/§6). Fitted ECE uses one scalar temperature
on confidence — same numbers reported raw and after grid fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .temps import fit_temperature


@dataclass
class EvalMetrics:
    n: int
    accuracy: float
    brier: float
    ece_raw: float
    ece_fitted: float
    temperature: float
    mean_conf: float

    def as_dict(self) -> dict:
        return asdict(self)


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Standard equal-width ECE over confidence in [0,1]."""
    if conf.size == 0:
        raise ValueError("empty batch")
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    temperatures: np.ndarray | None = None,
    n_bins: int = 15,
) -> EvalMetrics:
    """probs [N,K] (already softmax); targets [N].

    Optional `temperatures` are pre-divided logits reconstructed as
    p^(1/T)/sum — prefer `evaluate_logits` which fits T on the batch.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or targets.shape != (probs.shape[0],):
        raise ValueError("probs [N,K] and targets [N] required")
    if probs.shape[0] == 0:
        raise ValueError("empty batch")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    accuracy = float(correct.mean())

    onehot = np.zeros_like(probs)
    onehot[np.arange(targets.size), targets] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    ece_raw = _ece(conf, correct, n_bins=n_bins)

    # Fit temperature on log-probs: T minimizes NLL of softmax(log p / T).
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    # Build logit-like matrix: log p is fine for temp scaling (equiv up to const).
    t = fit_temperature(log_p, onehot, lo=0.5, hi=5.0)
    z = log_p / t
    z = z - z.max(axis=1, keepdims=True)
    p_t = np.exp(z)
    p_t = p_t / p_t.sum(axis=1, keepdims=True)
    conf_t = p_t.max(axis=1)
    pred_t = p_t.argmax(axis=1)
    correct_t = (pred_t == targets).astype(np.float64)
    ece_fit = _ece(conf_t, correct_t, n_bins=n_bins)

    return EvalMetrics(
        n=int(probs.shape[0]),
        accuracy=accuracy,
        brier=brier,
        ece_raw=ece_raw,
        ece_fitted=ece_fit,
        temperature=float(t),
        mean_conf=float(conf.mean()),
    )


def evaluate_logits(logits: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> EvalMetrics:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    return evaluate_probs(p, targets, n_bins=n_bins)
