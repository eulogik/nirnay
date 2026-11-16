"""RLCD++ loss suite (plan §1 item 8, §2 Loss; Days 15–35).

- Brier+correctness reward (RLCR, bounded — never pure unbounded log reward)
- decision-token CE: one-hot target when correct, uniform when wrong
  (ACL 2026 calCE, plan λ ∈ [0.001, 0.01])
- REINFORCE + group-mean baseline (GRPO-style): advantages zero-mean per group

Never train on verifier-rejected-tail self-labels (plan §6/§7, arXiv:2609.01345).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CALCE_LAMBDA_DEFAULT = 0.005  # midpoint of plan λ range [0.001, 0.01]


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Multiclass Brier: sum_k (p_k - t_k)^2 ∈ [0, 2] for one-hot targets."""
    if probs.shape != targets.shape:
        raise ValueError("probs and targets must match shape")
    return (probs - targets).pow(2).sum(dim=-1)


def brier_reward(probs: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """1 - Brier/2 → bounded [0, 1]. Perfect one-hot → 1; wrong one-hot → 0."""
    onehot = F.one_hot(target_idx, probs.size(-1)).to(probs.dtype)
    return 1.0 - 0.5 * brier_score(probs, onehot)


def correctness_reward(probs: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """1.0 where argmax matches target else 0.0 (bounded)."""
    return (probs.argmax(dim=-1) == target_idx).to(probs.dtype)


def rlcd_sample_reward(
    probs: torch.Tensor, target_idx: torch.Tensor, correctness_weight: float = 1.0
) -> torch.Tensor:
    """Brier + correctness reward, normalized to [0, 1] by max possible 2."""
    r = brier_reward(probs, target_idx) + correctness_weight * correctness_reward(
        probs, target_idx
    )
    return r / (1.0 + correctness_weight)


def decision_token_ce(
    log_probs: torch.Tensor,
    target_idx: torch.Tensor,
    correct: torch.Tensor | None = None,
) -> torch.Tensor:
    """calCE: one-hot CE when correct, uniform CE when wrong (ACL 2026).

    log_probs: [B, K] log-softmax; target_idx: [B]; correct: [B] bool
    (defaults to argmax == target). Returns scalar mean.
    """
    K = log_probs.size(-1)
    if correct is None:
        correct = log_probs.argmax(dim=-1) == target_idx
    onehot = F.one_hot(target_idx, K).to(log_probs.dtype)
    uniform = torch.full_like(onehot, 1.0 / K)
    mixed = torch.where(correct.unsqueeze(-1), onehot, uniform)
    return -(mixed * log_probs).sum(dim=-1).mean()


def group_mean_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """REINFORCE group-mean baseline: advantages = r - mean(group).

    rewards: [G*group_size] (or [G, group_size]). Each group's advantages
    sum to ~0. Returns same shape as input flattened 1-D.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    r = rewards.reshape(-1)
    if r.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    groups = r.view(-1, group_size)
    adv = groups - groups.mean(dim=1, keepdim=True)
    return adv.reshape_as(r)


def policy_gradient_loss(
    log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    if log_probs.ndim != 2 or actions.ndim != 1 or advantages.ndim != 1:
        raise ValueError("log_probs [B,K], actions [B], advantages [B] required")
    if log_probs.size(0) != actions.numel() or advantages.numel() != actions.numel():
        raise ValueError("policy tensors must have the same batch size")
    selected = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
    return -(advantages.detach() * selected).mean()


def rlcd_total_loss(
    choice_ce: torch.Tensor,
    cal_ce: torch.Tensor,
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    actions: torch.Tensor,
    cal_lambda: float = CALCE_LAMBDA_DEFAULT,
) -> torch.Tensor:
    """L_choiceCE + λ*L_calCE − E[A·log π(action)]."""
    if not (0.0 < cal_lambda < 1.0):
        raise ValueError("cal_lambda must be in (0, 1)")
    policy = policy_gradient_loss(log_probs, actions, advantages)
    return choice_ce + cal_lambda * cal_ce + policy
