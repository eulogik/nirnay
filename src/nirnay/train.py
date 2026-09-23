"""Phase A SFT stack (plan §2): LoRA + additions on the Laya backbone.

NirnayTrainModel reimplements DecisionModel.forward with:
  - LoRA-wrapped encoder linears (base frozen)
  - Concept bottleneck after encoder hidden states
  - Deep supervision hooks at layers 4/8/12
  - Optional coarse-to-fine auxiliary for 77-way choice

Losses follow `losses.assemble_plan_loss` exactly.
"""

from __future__ import annotations

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
