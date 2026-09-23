"""G10: RLCD++ losses — Brier reward bounds, calCE targets, group baseline."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.rlcd import (
        brier_reward,
        decision_token_ce,
        group_mean_advantages,
        rlcd_sample_reward,
        rlcd_total_loss,
    )

    torch.manual_seed(0)
    B, K, G = 32, 5, 8  # G groups × (B/G) would also work; use explicit below
    group_size = 4

    probs = torch.softmax(torch.randn(B, K), dim=-1)
    targets = torch.randint(0, K, (B,))

    # --- Brier reward ∈ [0, 1]; perfect → 1; all-mass-wrong → 0 ---
    rew = brier_reward(probs, targets)
    if rew.shape != (B,) or not torch.isfinite(rew).all():
        print(f"RLCD_FAIL brier_reward_shape={tuple(rew.shape)}", file=sys.stderr)
        return 1
    if float(rew.min()) < -1e-6 or float(rew.max()) > 1.0 + 1e-6:
        print(f"RLCD_FAIL brier_reward_range=[{float(rew.min())},{float(rew.max())}]",
              file=sys.stderr)
        return 1

    perfect = torch.zeros(B, K)
    perfect[torch.arange(B), targets] = 1.0
    rew_perfect = brier_reward(perfect, targets)
    if not torch.allclose(rew_perfect, torch.ones_like(rew_perfect), atol=1e-5):
        print(f"RLCD_FAIL perfect_reward={rew_perfect[:4].tolist()}", file=sys.stderr)
        return 1

    wrong = torch.zeros(B, K)
    wrong[torch.arange(B), (targets + 1) % K] = 1.0
    rew_wrong = brier_reward(wrong, targets)
    if not torch.allclose(rew_wrong, torch.zeros_like(rew_wrong), atol=1e-5):
        print(f"RLCD_FAIL wrong_reward={rew_wrong[:4].tolist()}", file=sys.stderr)
        return 1

    combo = rlcd_sample_reward(probs, targets)
    if float(combo.min()) < -1e-6 or float(combo.max()) > 1.0 + 1e-6:
        print(f"RLCD_FAIL combo_range=[{float(combo.min())},{float(combo.max())}]",
              file=sys.stderr)
        return 1

    # --- decision_token_ce: one-hot when correct, uniform when wrong ---
    log_p = torch.log(probs.clamp_min(1e-12))
    correct_mask = torch.ones(B, dtype=torch.bool)
    ce_correct = decision_token_ce(log_p, targets, correct_mask)
    # Standard NLL for all-correct case:
    nll = -log_p[torch.arange(B), targets].mean()
    if abs(float(ce_correct) - float(nll)) > 1e-5:
        print(
            f"RLCD_FAIL calCE_correct={float(ce_correct)} nll={float(nll)}",
            file=sys.stderr,
        )
        return 1

    wrong_mask = torch.zeros(B, dtype=torch.bool)
    ce_wrong = decision_token_ce(log_p, targets, wrong_mask)
    uniform_ce = -log_p.mean(dim=-1).mean()  # -mean_k log p_k
    if abs(float(ce_wrong) - float(uniform_ce)) > 1e-5:
        print(
            f"RLCD_FAIL calCE_wrong={float(ce_wrong)} uniform={float(uniform_ce)}",
            file=sys.stderr,
        )
        return 1
    if not (torch.isfinite(ce_correct) and torch.isfinite(ce_wrong)):
        print("RLCD_FAIL calCE_nonfinite", file=sys.stderr)
        return 1

    # --- group-mean advantages: each group mean ≈ 0 ---
    rewards = torch.rand(G * group_size)
    adv = group_mean_advantages(rewards, group_size)
    if adv.shape != rewards.shape:
        print(f"RLCD_FAIL adv_shape={tuple(adv.shape)}", file=sys.stderr)
        return 1
    adv_groups = adv.view(G, group_size).mean(dim=1)
    if not torch.allclose(adv_groups, torch.zeros_like(adv_groups), atol=1e-5):
        print(f"RLCD_FAIL group_means={adv_groups.tolist()}", file=sys.stderr)
        return 1

    # --- combined loss finite ---
    choice_ce = torch.nn.functional.cross_entropy(probs.log(), targets)
    total = rlcd_total_loss(choice_ce, ce_correct, adv)
    if total.ndim != 0 or not torch.isfinite(total):
        print(f"RLCD_FAIL total={total}", file=sys.stderr)
        return 1

    print(
        f"RLCD_OK brier_in=[0,1] perfect=1 wrong=0 "
        f"calCE_onehot_when_correct=true calCE_uniform_when_wrong=true "
        f"group_means~0 total={float(total):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
