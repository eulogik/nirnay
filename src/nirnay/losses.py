"""Plan §2 loss assembly (exact coefficients).

L = L_choiceCE + L_RPS(score) + L_BCE(noul) + L_relational
    + 0.3·L_NCP + 0.2·L_deep + λ·L_calCE     (λ ∈ [0.001, 0.01])

Phase B RL terms (REINFORCE + Brier reward) live in `rlcd.py` and are added
by the RL trainer; this module is the SFT (Phase A) total only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

NCP_COEF = 0.3
DEEP_COEF = 0.2
CAL_LAMBDA_DEFAULT = 0.005  # midpoint of plan [0.001, 0.01]


@dataclass
class LossParts:
    choice_ce: torch.Tensor
    rps: torch.Tensor
    noul_bce: torch.Tensor
    relational: torch.Tensor
    ncp: torch.Tensor
    deep: torch.Tensor
    cal_ce: torch.Tensor
    total: torch.Tensor
    ncp_coef: float = NCP_COEF
    deep_coef: float = DEEP_COEF
    cal_lambda: float = CAL_LAMBDA_DEFAULT


def choice_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked CE over marker logits. logits [B,K], targets [B], mask [B,K] bool."""
    masked = logits.masked_fill(~mask, -1e4)
    return F.cross_entropy(masked, targets)


def rps_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ranked Probability Score for ordered score questions (mean over batch).

    RPS = mean_i sum_k (CDF_p - CDF_t)^2 / (K-1) ∈ [0, 1] for one-hot targets.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [B, K]")
    K = probs.size(-1)
    if K < 2:
        return probs.new_zeros(())
    cdf_p = probs.cumsum(dim=-1)
    onehot = F.one_hot(targets, K).to(probs.dtype)
    cdf_t = onehot.cumsum(dim=-1)
    return ((cdf_p - cdf_t) ** 2).sum(dim=-1).mean() / (K - 1)


def noul_bce(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary CE on the true class of a 2-way noul distribution.

    probs [B, 2], targets {0,1} [B]. Equal to -mean log p[target].
    """
    p_true = probs.gather(1, targets.view(-1, 1)).squeeze(1)
    return -torch.log(p_true.clamp_min(1e-12)).mean()


def cal_ce(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    correct: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decision-token CE: one-hot when correct, uniform when wrong (ACL 2026)."""
    K = log_probs.size(-1)
    if correct is None:
        correct = log_probs.argmax(dim=-1) == targets
    onehot = F.one_hot(targets, K).to(log_probs.dtype)
    uniform = torch.full_like(onehot, 1.0 / K)
    mixed = torch.where(correct.unsqueeze(-1), onehot, uniform)
    return -(mixed * log_probs).sum(dim=-1).mean()


def assemble_plan_loss(
    choice_ce_t: torch.Tensor,
    rps_t: torch.Tensor,
    noul_bce_t: torch.Tensor,
    relational_t: torch.Tensor,
    ncp_t: torch.Tensor,
    deep_t: torch.Tensor,
    cal_ce_t: torch.Tensor,
    ncp_coef: float = NCP_COEF,
    deep_coef: float = DEEP_COEF,
    cal_lambda: float = CAL_LAMBDA_DEFAULT,
) -> LossParts:
    """Exact plan §2 combination. Raises on λ outside (0,1)."""
    if not (0.0 < cal_lambda < 1.0):
        raise ValueError("cal_lambda must be in (0, 1)")
    if ncp_coef != NCP_COEF:
        raise ValueError(f"plan §2 fixes NCP coefficient at {NCP_COEF}")
    if deep_coef != DEEP_COEF:
        raise ValueError(f"plan §2 fixes deep coefficient at {DEEP_COEF}")
    total = (
        choice_ce_t
        + rps_t
        + noul_bce_t
        + relational_t
        + ncp_coef * ncp_t
        + deep_coef * deep_t
        + cal_lambda * cal_ce_t
    )
    return LossParts(
        choice_ce=choice_ce_t,
        rps=rps_t,
        noul_bce=noul_bce_t,
        relational=relational_t,
        ncp=ncp_t,
        deep=deep_t,
        cal_ce=cal_ce_t,
        total=total,
        ncp_coef=ncp_coef,
        deep_coef=deep_coef,
        cal_lambda=cal_lambda,
    )


def plan_loss_from_batch(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    ncp: torch.Tensor,
    deep: torch.Tensor,
    relational: torch.Tensor | None = None,
    cal_lambda: float = CAL_LAMBDA_DEFAULT,
) -> LossParts:
    """Segment a batch by qtype and assemble the plan total.

    qtype: 0=choice, 1=score, 2=noul (laya QTYPES).
    Missing segments contribute 0 (keeps smoke batches valid).
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = probs.clamp_min(1e-12).log()

    sel_c = qtype == 0
    sel_s = qtype == 1
    sel_n = qtype == 2

    zero = logits.new_zeros(())
    if sel_c.any():
        c_logits = logits[sel_c]
        c_mask = marker_mask[sel_c]
        c_lab = labels[sel_c]
        # pad labels for cross_entropy against K markers: targets are option indices
        c_ce = choice_ce(c_logits, c_lab.clamp(min=0), c_mask)
        c_logp = log_probs[sel_c]
        # only valid marker cols for cal — use full row (masked cols ~0 mass)
        c_cal = cal_ce(c_logp, c_lab.clamp(min=0))
    else:
        c_ce = zero
        c_cal = zero

    if sel_s.any():
        s_probs = probs[sel_s]
        s_lab = labels[sel_s].clamp(min=0)
        rps_t = rps_loss(s_probs, s_lab)
    else:
        rps_t = zero

    if sel_n.any():
        n_probs = probs[sel_n]
        n_lab = labels[sel_n].clamp(min=0)
        noul_t = noul_bce(n_probs, n_lab)
    else:
        noul_t = zero

    rel = relational if relational is not None else zero
    # deep/ncp are per-batch scalars from modules
    return assemble_plan_loss(
        c_ce, rps_t, noul_t, rel, ncp, deep, c_cal, cal_lambda=cal_lambda
    )
