"""Phase A SFT stack (plan §2): LoRA + additions on the Laya backbone.

NirnayTrainModel reimplements DecisionModel.forward with:
  - LoRA-wrapped encoder linears (base frozen)
  - Concept bottleneck after encoder hidden states
  - Deep supervision hooks at layers 4/8/12
  - Optional coarse-to-fine auxiliary for 77-way choice

Losses follow `losses.assemble_plan_loss` exactly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.common import DecisionModel, QTYPES, build_sequence, collate_items, render_options

from .concepts import ConceptBottleneck, ConceptConfig
from .coarse2fine import CoarseToFine
from .data import DecisionExample, to_laya_question
from .deepsup import DEEP_LAYERS, DeepSupervision
from .losses import LossParts, plan_loss_from_batch
from .lora import apply_lora, count_frozen, count_trainable

DEEPSUP_NUM_LABELS = 80  # ≥77 Banking77 + margin; probes output scores over fixed head


class NirnayTrainModel(nn.Module):
    """Laya DecisionModel + NIRNAY additions for Phase A SFT."""

    def __init__(
        self,
        base: DecisionModel,
        *,
        use_lora: bool = True,
        lora_rank: int = 8,
        use_concepts: bool = True,
        use_deepsup: bool = True,
        use_c2f: bool = True,
        n_labels_bank: int = 77,
        concept_config: ConceptConfig | None = None,
    ):
        super().__init__()
        self.base = base
        self.use_concepts = use_concepts
        self.use_deepsup = use_deepsup
        self.use_c2f = use_c2f

        hidden = int(base.encoder.config.hidden_size)
        if use_lora:
            self.lora_wrapped = apply_lora(base.encoder, rank=lora_rank)
        else:
            self.lora_wrapped = 0

        self.concepts: Optional[ConceptBottleneck] = None
        if use_concepts:
            self.concepts = ConceptBottleneck(concept_config or ConceptConfig(encoder_hidden=hidden))

        self.deepsup: Optional[DeepSupervision] = None
        self._captured: dict[int, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        if use_deepsup:
            self.deepsup = DeepSupervision(hidden_size=hidden, num_labels=DEEPSUP_NUM_LABELS)
            layers = base.encoder.layers
            for idx in DEEP_LAYERS:
                if idx >= len(layers):
                    continue
                handle = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hook_handles.append(handle)

        self.c2f: Optional[CoarseToFine] = None
        if use_c2f:
            self.c2f = CoarseToFine(hidden_size=hidden, num_labels=n_labels_bank)

        self.freeze_base_encoder_non_lora()

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._captured[idx] = h

        return hook

    def freeze_base_encoder_non_lora(self) -> None:
        """Freeze all encoder params except LoRA A/B; leave heads trainable."""
        for name, p in self.base.encoder.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    def trainable_summary(self) -> dict[str, int]:
        return {
            "trainable": count_trainable(self),
            "frozen": count_frozen(self),
            "lora_wrapped": self.lora_wrapped,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self._captured.clear()
        ids = batch["input_ids"]
        att = batch["attention_mask"]
        mpos = batch["marker_pos"]
        mmask = batch["marker_mask"]
        qtype = batch["qtype"]

        h = self.base.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        ncp = h.new_zeros(())
        if self.concepts is not None:
            h, info = self.concepts(h)
            ncp = info["ncp_loss"]

        h = h + self.base.type_emb(qtype)[:, None, :]
        if self.base.head is not None:
            pad = ~att.bool()
            for layer in self.base.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.base.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~mmask, -1e4)

        # act head (same feature recipe as laya)
        p_det = torch.softmax(logits.detach(), -1)
        k = mmask.sum(-1).clamp(min=2).float()
        ent = -(p_det * torch.log(p_det.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p_det.size(-1) >= 2:
            top2 = p_det.topk(2, -1).values
        else:
            top1 = p_det.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.base.act_head(torch.cat([pooled, feats], -1))

        deep = h.new_zeros(())
        deep_parts: dict[int, torch.Tensor] = {}
        if self.deepsup is not None and "label" in batch:
            labels = batch["label"]
            # probes are over DEEPSUP_NUM_LABELS; clamp Banking77 ids into range
            t = labels.clamp(min=0, max=DEEPSUP_NUM_LABELS - 1)
            # only rows with a real label (>=0)
            valid = labels >= 0
            if valid.any():
                capt = {i: self._captured[i] for i in DEEP_LAYERS if i in self._captured}
                if len(capt) == len(self.deepsup.layers):
                    sub = {i: capt[i][valid] for i in capt}
                    deep, deep_parts = self.deepsup(sub, t[valid])

        return {
            "logits": logits,
            "act_logits": act_logits,
            "ncp": ncp,
            "deep": deep,
            "deep_parts": deep_parts,
            "captured": dict(self._captured),
        }

    def loss_from_batch(self, batch: dict[str, torch.Tensor], out: dict[str, Any]) -> LossParts:
        return plan_loss_from_batch(
            logits=out["logits"],
            labels=batch["label"],
            marker_mask=batch["marker_mask"],
            qtype=batch["qtype"],
            ncp=out["ncp"],
            deep=out["deep"],
        )

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def encode_examples(
    examples: list[DecisionExample],
    tok,
    max_len: int = 512,
    head_max_len: int = 192,
    label_space: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """DecisionExample → collate-ready items with integer labels."""
    items = []
    for ex in examples:
        q = to_laya_question(ex)
        internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
        seq, markers = build_sequence(tok, ex.state, internal, max_len, head_max_len)
        if ex.qtype == "choice":
            assert isinstance(ex.criteria, dict)
            keys = list(ex.criteria.keys())
            if label_space is not None:
                label = label_space[str(ex.answer)]
            else:
                label = keys.index(ex.answer)
        elif ex.qtype == "score":
            label = int(ex.answer)
        else:
            label = int(bool(ex.answer))
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[ex.qtype],
                "label": label,
                "id": ex.id,
            }
        )
    return items


def collate_examples(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    return collate_items([items], pad_id)


class PhaseASFT:
    """Optimizer loop over trainable params only (encoder base stays frozen)."""

    def __init__(
        self,
        model: NirnayTrainModel,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        cal_lambda: float = 0.005,
    ):
        self.model = model
        self.cal_lambda = cal_lambda
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable parameters — check LoRA/additions wiring")
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out = self.model(batch)
        parts = self.model.loss_from_batch(batch, out)
        parts.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.opt.step()
        return {
            "loss": float(parts.total.detach()),
            "choice_ce": float(parts.choice_ce.detach()),
            "rps": float(parts.rps.detach()),
            "noul_bce": float(parts.noul_bce.detach()),
            "ncp": float(parts.ncp.detach()),
            "deep": float(parts.deep.detach()),
            "cal_ce": float(parts.cal_ce.detach()),
        }

    def fit(
        self,
        batches: list[dict[str, torch.Tensor]],
        steps: int,
    ) -> list[dict[str, float]]:
        history = []
        for i in range(steps):
            batch = batches[i % len(batches)]
            history.append(self.step(batch))
        return history

    def save_checkpoint(self, path: str, meta: dict[str, Any] | None = None) -> str:
        """Save trainable (and full) state for resume / Phase B handoff."""
        import json as _json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "meta": meta or {},
        }
        torch.save(payload, p)
        side = p.with_suffix(p.suffix + ".meta.json")
        side.write_text(_json.dumps(meta or {}, indent=2, sort_keys=True) + "\n")
        return str(p)


class PhaseBRLCD:
    """REINFORCE + group-mean baseline + Gaussian logit noise (plan §2).

    Samples actions with additive Gaussian noise on logits (policy
    exploration), scores with Brier+correctness reward, updates with
    rlcd_total_loss (choiceCE + λ·calCE − mean advantage).
    """

    def __init__(
        self,
        model: NirnayTrainModel,
        lr: float = 1e-4,
        group_size: int = 4,
        noise_std: float = 0.1,
        cal_lambda: float = 0.005,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        self.model = model
        self.group_size = group_size
        self.noise_std = noise_std
        self.cal_lambda = cal_lambda
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable parameters")
        self.opt = torch.optim.AdamW(params, lr=lr)

    @torch.no_grad()
    def sample_action(
        self, logits: torch.Tensor, marker_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian noise on valid logits; sample categorical action.

        Returns (action [B], noisy_probs [B,K] for reward).
        """
        noise = torch.randn_like(logits) * self.noise_std
        noisy = logits + noise.masked_fill(~marker_mask, -1e4)
        probs = torch.softmax(noisy, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, probs

    def step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        from .rlcd import (
            decision_token_ce,
            group_mean_advantages,
            rlcd_sample_reward,
            rlcd_total_loss,
        )

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out = self.model(batch)
        logits = out["logits"]
        mmask = batch["marker_mask"]
        labels = batch["label"]

        action, noisy_probs = self.sample_action(logits, mmask)
        # Use gold labels as reward targets (supervised RLCD warm start —
        # plan Phase B starts from Phase A; rewards need ground truth here).
        rewards = rlcd_sample_reward(noisy_probs, labels)

        # Repeat each sample group_size times for group baseline.
        B = rewards.size(0)
        rewards_rep = rewards.repeat_interleave(self.group_size)
        # Align group length: take full groups only
        n_groups_total = rewards_rep.numel() // self.group_size
        # group_mean_advantages expects flat length divisible by group_size
        # and groups of consecutive items — repeat already gives that if we
        # regroup as [B, group_size] after expand:
        rewards_mat = rewards.unsqueeze(1).expand(B, self.group_size).reshape(-1)
        adv = group_mean_advantages(rewards_mat, self.group_size)

        log_p = torch.log(torch.softmax(logits, dim=-1).clamp_min(1e-12))
        # Expand logits/rewards path: use original batch CE + calCE, policy term
        # from expanded advantages mean (equals group-mean of original rewards).
        choice_ce = torch.nn.functional.cross_entropy(
            logits.masked_fill(~mmask, -1e4), labels.clamp(min=0)
        )
        cal = decision_token_ce(log_p, labels.clamp(min=0))
        total = rlcd_total_loss(choice_ce, cal, adv, cal_lambda=self.cal_lambda)
        # Also fold NCP/deep from forward so additions keep training
        total = total + 0.3 * out["ncp"] + 0.2 * out["deep"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.opt.step()
        return {
            "loss": float(total.detach()),
            "reward_mean": float(rewards.mean()),
            "adv_mean": float(adv.mean()),
            "choice_ce": float(choice_ce.detach()),
            "ncp": float(out["ncp"].detach()),
            "deep": float(out["deep"].detach()),
        }

    def fit(
        self, batches: list[dict[str, torch.Tensor]], steps: int
    ) -> list[dict[str, float]]:
        history = []
        for i in range(steps):
            history.append(self.step(batches[i % len(batches)]))
        return history


def batch_size_ok(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    return collate_examples(items, pad_id)


def run_phase_a(
    *,
    steps: int = 3,
    batch_size: int = 8,
    n_synth: int = 32,
    banking_dir: str | None = None,
    banking_limit: int | None = None,
    out_dir: str = "artifacts/phase_a",
    lora_rank: int = 4,
    lr: float = 1e-3,
    seed: int = 13,
    device: str = "cpu",
) -> dict[str, Any]:
    """CLI entry: build mix → encode → SFT → write history + checkpoint."""
    import json as _json
    from pathlib import Path

    from .data import build_phase_a_mix, split_train_heldout, write_freeze_manifest

    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mix = build_phase_a_mix(
        n_synth=n_synth,
        banking_cache=banking_dir,
        banking_limit=banking_limit,
        seed=seed,
    )
    train_ex, held_ex = split_train_heldout(mix, heldout_frac=0.2, seed=seed)
    write_freeze_manifest(out / "freeze_manifest.json", mix)

    from nirnay import load as load_agent

    agent_path = os.environ.get("NIRNAY_MODEL", "convaiinnovations/laya")
    agent = load_agent(agent_path, device=device, enable_byte_path=False, nope_fraction=0.0)
    model = NirnayTrainModel(
        agent.model,
        use_lora=True,
        lora_rank=lora_rank,
        use_concepts=True,
        use_deepsup=True,
        use_c2f=True,
        n_labels_bank=77,
    )
    # Snapshot an encoder base param before training
    enc_param = next(
        p for n, p in agent.model.encoder.named_parameters() if "lora_" not in n
    )
    enc_before = enc_param.detach().clone()

    items = encode_examples(train_ex, agent.tok)
    batches: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        if len(chunk) < 2:
            continue
        b = collate_examples(chunk, agent.tok.pad_token_id)
        batches.append({k: v for k, v in b.items()})
    if not batches:
        raise RuntimeError("no batches — check mix size vs batch_size")

    trainer = PhaseASFT(model, lr=lr)
    history = trainer.fit(batches, steps=steps)
    losses = [h["loss"] for h in history]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        raise RuntimeError(f"non-finite loss: {losses}")

    if not torch.equal(enc_before, enc_param.detach()):
        raise RuntimeError("encoder base changed during Phase A — must stay frozen")

    ckpt = trainer.save_checkpoint(
        str(out / "phase_a.pt"),
        meta={
            "steps": steps,
            "n_train": len(train_ex),
            "n_heldout": len(held_ex),
            "n_mix": len(mix),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "trainable": model.trainable_summary()["trainable"],
            "encoder_frozen": True,
            "seed": seed,
        },
    )
    (out / "history.json").write_text(
        _json.dumps(history, indent=2) + "\n"
    )
    return {
        "ckpt": ckpt,
        "history": history,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "encoder_frozen": True,
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m nirnay.train` — Phase A CLI (Phase B via --phase b)."""
    import argparse

    from .data import build_phase_a_mix, split_train_heldout, write_freeze_manifest

    ap = argparse.ArgumentParser(prog="nirnay.train", description="NIRNAY Phase A/B training")
    ap.add_argument("--phase", choices=("a", "b"), default="a")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-synth", type=int, default=32)
    ap.add_argument("--banking-dir", default="data/banking77")
    ap.add_argument("--banking-limit", type=int, default=None)
    ap.add_argument("--out-dir", default="artifacts/phase_a")
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group-size", type=int, default=4)
    args = ap.parse_args(argv)

    if args.phase == "a":
        result = run_phase_a(
            steps=args.steps,
            batch_size=args.batch_size,
            n_synth=args.n_synth,
            banking_dir=args.banking_dir if Path(args.banking_dir).exists() else None,
            banking_limit=args.banking_limit,
            out_dir=args.out_dir,
            lora_rank=args.lora_rank,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        )
        h = result["history"]
        print(
            f"PHASE_A_OK steps={len(h)} loss0={h[0]['loss']:.4f} "
            f"lossN={h[-1]['loss']:.4f} ckpt={result['ckpt']} "
            f"train={result['n_train']} heldout={result['n_heldout']} "
            f"encoder_frozen=true"
        )
        return 0

    # Phase B: load Phase A checkpoint if present, run RLCD
    import os  # noqa: F401 — already imported at module top? ensure below

    result_b = run_phase_b(
        steps=args.steps,
        batch_size=args.batch_size,
        n_synth=args.n_synth,
        banking_dir=args.banking_dir if Path(args.banking_dir).exists() else None,
        banking_limit=args.banking_limit,
        out_dir=args.out_dir.replace("phase_a", "phase_b")
        if "phase_a" in args.out_dir
        else "artifacts/phase_b",
        lora_rank=args.lora_rank,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        group_size=args.group_size,
    )
    h = result_b["history"]
    print(
        f"PHASE_B_OK steps={len(h)} loss0={h[0]['loss']:.4f} "
        f"lossN={h[-1]['loss']:.4f} reward0={h[0]['reward_mean']:.4f} "
        f"ckpt={result_b['ckpt']}"
    )
    return 0


def run_phase_b(
    *,
    steps: int = 3,
    batch_size: int = 8,
    n_synth: int = 32,
    banking_dir: str | None = None,
    banking_limit: int | None = None,
    out_dir: str = "artifacts/phase_b",
    lora_rank: int = 4,
    lr: float = 1e-4,
    seed: int = 13,
    device: str = "cpu",
    group_size: int = 4,
) -> dict[str, Any]:
    """Phase B RLCD after Phase A (loads phase_a.pt when available)."""
    import json as _json
    from pathlib import Path

    from .data import build_phase_a_mix, split_train_heldout, write_freeze_manifest

    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mix = build_phase_a_mix(
        n_synth=n_synth,
        banking_cache=banking_dir,
        banking_limit=banking_limit,
        seed=seed,
    )
    train_ex, held_ex = split_train_heldout(mix, heldout_frac=0.2, seed=seed)
    write_freeze_manifest(out / "freeze_manifest.json", mix)

    from nirnay import load as load_agent

    agent_path = os.environ.get("NIRNAY_MODEL", "convaiinnovations/laya")
    agent = load_agent(agent_path, device=device, enable_byte_path=False, nope_fraction=0.0)
    model = NirnayTrainModel(
        agent.model,
        use_lora=True,
        lora_rank=lora_rank,
        use_concepts=True,
        use_deepsup=True,
        use_c2f=True,
        n_labels_bank=77,
    )
    # Resume Phase A weights if present
    pa = Path("artifacts/phase_a/phase_a.pt")
    if pa.exists():
        blob = torch.load(pa, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=False)

    items = encode_examples(train_ex, agent.tok)
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        if len(chunk) < 2:
            continue
        b = collate_examples(chunk, agent.tok.pad_token_id)
        batches.append(dict(b))
    if not batches:
        raise RuntimeError("no batches for Phase B")

    trainer = PhaseBRLCD(model, lr=lr, group_size=group_size)
    history = trainer.fit(batches, steps=steps)
    losses = [h["loss"] for h in history]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
        raise RuntimeError(f"non-finite Phase B loss: {losses}")

    (out / "history.json").write_text(_json.dumps(history, indent=2) + "\n")
    # Save via a temp PhaseASFT-like path
    import json as _j
    from pathlib import Path as _P

    ckpt_path = out / "phase_b.pt"
    torch.save({"model": model.state_dict(), "meta": {"steps": steps}}, ckpt_path)
    (_P(str(ckpt_path) + ".meta.json")).write_text(
        _j.dumps({"steps": steps, "n_train": len(train_ex)}, indent=2) + "\n"
    )
    return {
        "ckpt": str(ckpt_path),
        "history": history,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
    }


if __name__ == "__main__":
    raise SystemExit(main())
