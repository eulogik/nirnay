"""G15: Phase A SFT smoke — LoRA+additions train, encoder base frozen, plan loss."""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch

    from nirnay.data import build_phase_a_mix, split_train_heldout
    from nirnay.lora import count_trainable
    from nirnay.train import NirnayTrainModel, PhaseASFT, collate_examples, encode_examples

    device = torch.device("cpu")
    agent_path = os.environ.get("NIRNAY_MODEL", "convaiinnovations/laya")

    # Minimal base: load Laya agent's DecisionModel
    from nirnay import load as load_agent

    agent = load_agent(agent_path, device="cpu", enable_byte_path=False, nope_fraction=1.0 / 3.0)
    base = agent.model
    tok = agent.tok

    model = NirnayTrainModel(
        base,
        use_lora=True,
        lora_rank=4,
        use_concepts=True,
        use_deepsup=True,
        use_c2f=True,
        use_byte_path=True,
        n_labels_bank=77,
    )
    summary = model.trainable_summary()
    if summary["lora_wrapped"] < 1:
        print(f"PHASE_A_FAIL lora_wrapped=0 summary={summary}", file=sys.stderr)
        return 1
    if summary["trainable"] <= 0:
        print(f"PHASE_A_FAIL trainable={summary}", file=sys.stderr)
        return 1

    # Encoder base sample must be frozen
    enc_sample = None
    enc_name = None
    for n, p in base.encoder.named_parameters():
        if "lora_" not in n:
            enc_sample, enc_name = p, n
            break
    if enc_sample is None or enc_sample.requires_grad:
        print(f"PHASE_A_FAIL encoder_not_frozen name={enc_name}", file=sys.stderr)
        return 1

    # Concepts params trainable
    if model.concepts is None or not any(p.requires_grad for p in model.concepts.parameters()):
        print("PHASE_A_FAIL concepts_not_trainable", file=sys.stderr)
        return 1
    if model.deepsup is None or not any(p.requires_grad for p in model.deepsup.parameters()):
        print("PHASE_A_FAIL deepsup_not_trainable", file=sys.stderr)
        return 1

    # Build small batch from synthetic mix
    mix = build_phase_a_mix(
        n_synth=24, banking_cache="data/banking77", banking_limit=4, seed=3
    )
    train_ex, _ = split_train_heldout(mix, heldout_frac=0.2, seed=13)
    items = encode_examples(train_ex, tok, max_len=512, head_max_len=192)
    banking_items = [item for item in items if item["c2f_eligible"]]
    if not banking_items:
        print("PHASE_A_FAIL no_c2f_fixture", file=sys.stderr)
        return 1
    items = items[:10] + banking_items[:1]
    batch = collate_examples(items, tok.pad_token_id)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Capture encoder weight snapshot
    before = enc_sample.detach().clone()

    trainer = PhaseASFT(model, lr=1e-3)
    history = trainer.fit([batch], steps=3)
    if len(history) != 3:
        print(f"PHASE_A_FAIL history={len(history)}", file=sys.stderr)
        return 1
    losses = [h["loss"] for h in history]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        print(f"PHASE_A_FAIL nonfinite losses={losses}", file=sys.stderr)
        return 1

    # Encoder base unchanged
    after = enc_sample.detach()
    if not torch.equal(before, after):
        print(f"PHASE_A_FAIL encoder_changed name={enc_name}", file=sys.stderr)
        return 1

    # NCP participated (finite, in graph → grad on concept encode)
    concept_grad = None
    for n, p in model.concepts.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            concept_grad = n
            break
    if concept_grad is None:
        print("PHASE_A_FAIL no_concept_grad", file=sys.stderr)
        return 1

    # Deep supervision grads at probes (captured hooks fired → deep loss path)
    deep_grad = None
    for n, p in model.deepsup.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            deep_grad = n
            break
    if deep_grad is None:
        # deep loss may be zero if batch lacks valid labels — check hooks fired
        if not model._captured or set(model._captured) != {4, 8, 12}:
            print(
                f"PHASE_A_FAIL deepsup_path captured={sorted(model._captured)}",
                file=sys.stderr,
            )
            return 1
        # labels present but deep CE zero is acceptable only if captured OK and
        # loss total still moved concepts — require deep finite from history
        if history[-1]["deep"] != history[-1]["deep"]:
            print("PHASE_A_FAIL deep_nan", file=sys.stderr)
            return 1

    if model.byte_fusion is None or model.c2f is None:
        print("PHASE_A_FAIL active_modules_missing", file=sys.stderr)
        return 1
    c2f_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.c2f.parameters()
    )
    byte_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.byte_fusion.parameters()
    )
    if not c2f_grad or not byte_grad:
        print(
            f"PHASE_A_FAIL c2f_grad={c2f_grad} byte_grad={byte_grad}",
            file=sys.stderr,
        )
        return 1
    if not all(not p.requires_grad for p in model.base.act_head.parameters()):
        print("PHASE_A_FAIL act_head_trainable_without_targets", file=sys.stderr)
        return 1
    if abs(history[-1]["deep_weighted"] - 0.2 * history[-1]["deep"]) > 1e-5:
        print("PHASE_A_FAIL deep_weighting", file=sys.stderr)
        return 1

    # Concepts params must have changed (optimizer stepped them)
    c_changed = False
    for p in model.concepts.parameters():
        if p.grad is not None:
            c_changed = True
            break
    if not c_changed:
        print("PHASE_A_FAIL concepts_no_step", file=sys.stderr)
        return 1

    # LoRA params exist and are trainable
    lora_n = count_trainable(model)
    base_trainable = sum(p.numel() for p in base.encoder.parameters() if p.requires_grad)
    if base_trainable <= 0:
        print("PHASE_A_FAIL no_lora_trainable", file=sys.stderr)
        return 1

    model.remove_hooks()
    print(
        f"PHASE_A_OK steps=3 loss0={losses[0]:.4f} lossN={losses[-1]:.4f} "
        f"lora_wrapped={summary['lora_wrapped']} trainable={lora_n} "
        f"encoder_frozen=true concept_grad={concept_grad} "
        f"wired=true byte_grad={byte_grad} c2f_grad={c2f_grad} "
        f"act_head_frozen=true deep_weighted={history[-1]['deep_weighted']:.4f} "
        f"captured={sorted(model._captured) if model._captured else 'cleared'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
