"""G18: Phase B RLCD smoke — noise sampling, group baseline, rewards, ≥3 steps."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import os

    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch

    from nirnay.data import build_phase_a_mix, split_train_heldout
    from nirnay.rlcd import (
        brier_reward,
        group_mean_advantages,
        policy_gradient_loss,
        rlcd_sample_reward,
    )
    from nirnay.train import (
        NirnayTrainModel,
        PhaseBRLCD,
        collate_examples,
        encode_examples,
    )

    torch.manual_seed(0)

    # Pure function checks first (must pass even if model load fails later)
    G, gs = 8, 4
    rew = torch.rand(G * gs)
    adv = group_mean_advantages(rew, gs)
    means = adv.view(G, gs).mean(dim=1)
    if not torch.allclose(means, torch.zeros_like(means), atol=1e-5):
        print(f"PHASE_B_FAIL group_means={means.tolist()}", file=sys.stderr)
        return 1

    probs = torch.softmax(torch.randn(16, 5), dim=-1)
    tgt = torch.randint(0, 5, (16,))
    r = brier_reward(probs, tgt)
    if float(r.min()) < -1e-6 or float(r.max()) > 1.0 + 1e-6:
        print(f"PHASE_B_FAIL brier_range=[{float(r.min())},{float(r.max())}]", file=sys.stderr)
        return 1
    combo = rlcd_sample_reward(probs, tgt)
    if float(combo.min()) < -1e-6 or float(combo.max()) > 1.0 + 1e-6:
        print(f"PHASE_B_FAIL combo_range=[{float(combo.min())},{float(combo.max())}]", file=sys.stderr)
        return 1

    # Full trainer ≥3 steps on Laya backbone
    from nirnay import load as load_agent

    agent = load_agent(
        os.environ.get("NIRNAY_MODEL", "convaiinnovations/laya"),
        device="cpu",
        enable_byte_path=False,
        nope_fraction=1.0 / 3.0,
    )
    model = NirnayTrainModel(
        agent.model,
        use_lora=True,
        lora_rank=4,
        use_concepts=True,
        use_deepsup=True,
        use_c2f=True,
        use_byte_path=True,
    )
    mix = build_phase_a_mix(n_synth=24, banking_cache="data/banking77", banking_limit=24)
    train_ex, _ = split_train_heldout(mix, heldout_frac=0.2, seed=13)
    items = encode_examples(train_ex[:10], agent.tok)
    batch = collate_examples(items, agent.tok.pad_token_id)

    trainer = PhaseBRLCD(model, lr=1e-4, group_size=4, noise_std=0.1)
    history = trainer.fit([batch], steps=3)
    if len(history) != 3:
        print(f"PHASE_B_FAIL history={len(history)}", file=sys.stderr)
        return 1
    losses = [h["loss"] for h in history]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        print(f"PHASE_B_FAIL nonfinite={losses}", file=sys.stderr)
        return 1
    rewards = [h["reward_mean"] for h in history]
    if any(not (0.0 <= x <= 1.0 + 1e-6) for x in rewards):
        print(f"PHASE_B_FAIL rewards={rewards}", file=sys.stderr)
        return 1
    adv_means = [h["adv_mean"] for h in history]
    # group means ≈ 0 → adv_mean over expanded groups ~0
    if any(abs(x) > 1e-4 for x in adv_means):
        print(f"PHASE_B_FAIL adv_mean={adv_means}", file=sys.stderr)
        return 1

    policy_probe = torch.randn(8, 5, requires_grad=True)
    policy_actions = torch.randint(0, 5, (8,))
    policy_adv = torch.randn(8)
    policy = policy_gradient_loss(policy_probe, policy_actions, policy_adv)
    policy.backward()
    if policy_probe.grad is None or float(policy_probe.grad.abs().sum()) <= 0.0:
        print("PHASE_B_FAIL policy_grad", file=sys.stderr)
        return 1

    # Noise sampling: action within valid marker support
    out = model(batch)
    action, noisy_p = trainer.sample_action(out["logits"], batch["marker_mask"])
    if action.shape[0] != out["logits"].shape[0]:
        print(f"PHASE_B_FAIL action_shape={tuple(action.shape)}", file=sys.stderr)
        return 1
    if not torch.isfinite(noisy_p).all():
        print("PHASE_B_FAIL noisy_probs_nonfinite", file=sys.stderr)
        return 1

    model.remove_hooks()
    print(
        f"PHASE_B_OK steps=3 loss0={losses[0]:.4f} lossN={losses[-1]:.4f} "
        f"reward_range=[{min(rewards):.3f},{max(rewards):.3f}] "
        f"adv_mean~0 group_baseline=true noise_std=0.1 policy_grad=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
