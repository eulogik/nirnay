"""Phase A SFT stack (plan §2): LoRA + additions on the Laya backbone.

NirnayTrainModel reimplements DecisionModel.forward with:
  - LoRA-wrapped encoder linears (base frozen)
  - Concept bottleneck after encoder hidden states
  - Deep supervision hooks at layers 4/8/12
  - Optional coarse-to-fine auxiliary for 77-way choice

Losses follow `losses.assemble_plan_loss` exactly.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.common import DecisionModel, QTYPES, build_sequence, collate_items, render_options

from .bytes import ByteFusion, BytePathConfig, encode_bytes
from .concepts import ConceptBottleneck, ConceptConfig
from .coarse2fine import CoarseToFine
from .data import BANKING77_LABELS, DecisionExample, to_laya_question
from .deepsup import DEEP_LAYERS, DeepSupervision
from .losses import CAL_LAMBDA_DEFAULT, LossParts, plan_loss_from_batch
from .lora import apply_lora, count_frozen, count_trainable

DEEPSUP_NUM_LABELS = 80  # ≥77 Banking77 + margin; probes output scores over fixed head
DEFAULT_MAX_BYTES = 512
CHECKPOINT_FORMAT_VERSION = 1


def _write_json_atomic(path: Path, payload: Any) -> None:
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
        use_byte_path: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        n_labels_bank: int = 77,
        concept_config: ConceptConfig | None = None,
    ):
        super().__init__()
        self.base = base
        self.use_concepts = use_concepts
        self.use_deepsup = use_deepsup
        self.use_c2f = use_c2f
        self.use_byte_path = use_byte_path
        self.max_bytes = int(max_bytes)
        self.lora_rank = int(lora_rank)
        self.n_labels_bank = int(n_labels_bank)

        hidden = int(base.encoder.config.hidden_size)
        if use_lora:
            self.lora_wrapped = apply_lora(base.encoder, rank=lora_rank)
        else:
            self.lora_wrapped = 0

        self.byte_fusion: Optional[ByteFusion] = None
        if use_byte_path:
            self.byte_fusion = ByteFusion(
                BytePathConfig(encoder_hidden=hidden, max_bytes=self.max_bytes)
            )

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
        """Freeze all encoder params except LoRA A/B; freeze unsupported act head."""
        for name, p in self.base.encoder.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        for p in self.base.act_head.parameters():
            p.requires_grad_(False)

    def trainable_summary(self) -> dict[str, int]:
        return {
            "trainable": count_trainable(self),
            "frozen": count_frozen(self),
            "lora_wrapped": self.lora_wrapped,
            "byte_fusion": self.byte_fusion is not None,
            "coarse_to_fine": self.c2f is not None,
            "act_head_frozen": not any(
                p.requires_grad for p in self.base.act_head.parameters()
            ),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self._captured.clear()
        ids = batch["input_ids"]
        att = batch["attention_mask"]
        mpos = batch["marker_pos"]
        mmask = batch["marker_mask"]
        qtype = batch["qtype"]

        if self.byte_fusion is None:
            encoded = self.base.encoder(input_ids=ids, attention_mask=att)
        else:
            embeddings = self.base.encoder.embeddings.tok_embeddings(ids)
            embeddings = self.byte_fusion(
                embeddings, batch.get("byte_ids"), batch.get("byte_mask")
            )
            encoded = self.base.encoder(
                inputs_embeds=embeddings,
                attention_mask=att,
            )
        h = encoded.last_hidden_state
        ncp = h.new_zeros(())
        codes = None
        if self.concepts is not None:
            h, info = self.concepts(h, attention_mask=att)
            ncp = info["ncp_loss"]
            codes = info["codes"]

        h = h + self.base.type_emb(qtype)[:, None, :]
        if self.base.head is not None:
            pad = ~att.bool()
            for layer in self.base.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        markers = torch.gather(h, 1, idx)
        logits = self.base.scorer(markers).squeeze(-1).float()
        logits = logits.masked_fill(~mmask, -1e4)

        c2f_eligible = torch.zeros_like(qtype, dtype=torch.bool)
        c2f_probs = logits.new_zeros((logits.size(0), self.c2f.num_labels if self.c2f else 0))
        if self.c2f is not None and logits.size(1) >= self.c2f.num_labels:
            if "c2f_mask" in batch:
                c2f_eligible = batch["c2f_mask"].to(torch.bool)
            else:
                c2f_eligible = (qtype == 0) & (
                    mmask.sum(dim=-1) == self.c2f.num_labels
                )
            labels = batch.get("label")
            if labels is not None and bool(labels.ge(0).any()):
                c2f_eligible &= labels.ge(0) & labels.lt(self.c2f.num_labels)
            rows = torch.nonzero(c2f_eligible, as_tuple=False).flatten()
            if rows.numel() > 0:
                row_hidden = h.index_select(0, rows)
                row_mask = att.index_select(0, rows).to(h.dtype).unsqueeze(-1)
                if "state_mask" in batch:
                    sm = batch["state_mask"].index_select(0, rows)
                    sm = sm.to(h.dtype).unsqueeze(-1)
                    sm = torch.where(sm.sum(dim=1, keepdim=True) > 0, sm, row_mask)
                    pool_mask = sm
                else:
                    pool_mask = row_mask
                query = (row_hidden * pool_mask).sum(dim=1) / pool_mask.sum(
                    dim=1
                ).clamp_min(1.0)
                # Stage-1: dense [MASK]-scorer top-k candidates (plan §1
                # "retrieve top-20 from 77+"). Gold teacher-forcing applies
                # only while training — eval/serve measure honest retrieval.
                dense = logits.index_select(0, rows)
                k = min(self.c2f.top_k, dense.size(1))
                cand = dense.topk(k, dim=-1).indices
                gold = None
                if (
                    self.training
                    and labels is not None
                    and bool((labels.index_select(0, rows) >= 0).all())
                ):
                    gold = labels.index_select(0, rows)
                    present = (cand == gold.unsqueeze(-1)).any(dim=-1, keepdim=True)
                    cand = torch.where(
                        present, cand, torch.cat([cand[:, :-1], gold.unsqueeze(-1)], dim=-1)
                    )
                bias = dense.gather(1, cand)
                c2f_out = self.c2f(
                    query, gold, candidates=cand, logit_bias=bias
                )
                full_probs = c2f_out["full_probs"]
                c2f_probs = c2f_probs.index_copy(0, rows, full_probs)
                row_logits = torch.log(full_probs.clamp_min(1e-8))
                if row_logits.size(1) < logits.size(1):
                    row_logits = F.pad(row_logits, (0, logits.size(1) - row_logits.size(1)), value=-1e4)
                else:
                    row_logits = row_logits[:, : logits.size(1)]
                replacement = logits.new_full(logits.shape, -1e4).index_copy(0, rows, row_logits)
                logits = torch.where(c2f_eligible.unsqueeze(-1), replacement, logits)

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
        deep_weighted = h.new_zeros(())
        deep_parts: dict[int, torch.Tensor] = {}
        if self.deepsup is not None and "label" in batch:
            labels = batch["label"]
            targets = labels.clamp(min=0, max=DEEPSUP_NUM_LABELS - 1)
            valid = labels >= 0
            if valid.any():
                captured = {
                    i: self._captured[i]
                    for i in DEEP_LAYERS
                    if i in self._captured
                }
                if len(captured) == len(self.deepsup.layers):
                    selected = {i: captured[i][valid] for i in captured}
                    _, deep_parts = self.deepsup(
                        selected,
                        targets[valid],
                        attention_mask=att[valid],
                    )
                    deep = torch.stack(list(deep_parts.values())).mean()
                    deep_weighted = self.deepsup.weight * deep

        return {
            "logits": logits,
            "act_logits": act_logits,
            "ncp": ncp,
            "codes": codes,
            "deep": deep,
            "deep_weighted": deep_weighted,
            "deep_parts": deep_parts,
            "c2f_eligible": c2f_eligible,
            "c2f_probs": c2f_probs,
            "captured": dict(self._captured),
        }

    def loss_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        out: dict[str, Any],
        cal_lambda: float = CAL_LAMBDA_DEFAULT,
    ) -> LossParts:
        relational = batch.get("relational")
        if relational is not None and not torch.is_tensor(relational):
            relational = torch.as_tensor(
                relational, dtype=out["logits"].dtype, device=out["logits"].device
            )
        return plan_loss_from_batch(
            logits=out["logits"],
            labels=batch["label"],
            marker_mask=batch["marker_mask"],
            qtype=batch["qtype"],
            ncp=out["ncp"],
            deep=out["deep"],
            relational=relational,
            cal_lambda=cal_lambda,
        )

    def model_config(self) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "lora_rank": self.lora_rank,
            "use_lora": self.lora_wrapped > 0,
            "use_concepts": self.concepts is not None,
            "use_deepsup": self.deepsup is not None,
            "use_c2f": self.c2f is not None,
            "use_byte_path": self.byte_fusion is not None,
            "max_bytes": self.max_bytes,
            "n_labels_bank": self.n_labels_bank,
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: state[name].detach().cpu() for name in names if name in state}

    def load_checkpoint_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload.get("model", payload)
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"unexpected checkpoint keys: {result.unexpected_keys[:5]}"
            )
        required = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        missing = sorted(required.intersection(result.missing_keys))
        if missing:
            raise RuntimeError(f"checkpoint missing trainable keys: {missing[:5]}")
        return dict(payload.get("meta", {}))

    @classmethod
    def from_checkpoint(
        cls,
        base: DecisionModel,
        payload: dict[str, Any],
        *,
        device: str | torch.device,
    ) -> tuple["NirnayTrainModel", dict[str, Any]]:
        meta = dict(payload.get("meta", {}))
        config = dict(meta.get("model_config", {}))
        model = cls(
            base,
            use_lora=bool(config.get("use_lora", True)),
            lora_rank=int(config.get("lora_rank", 8)),
            use_concepts=bool(config.get("use_concepts", True)),
            use_deepsup=bool(config.get("use_deepsup", True)),
            use_c2f=bool(config.get("use_c2f", True)),
            use_byte_path=bool(config.get("use_byte_path", True)),
            max_bytes=int(config.get("max_bytes", DEFAULT_MAX_BYTES)),
            n_labels_bank=int(config.get("n_labels_bank", 77)),
        )
        model.load_checkpoint_payload(payload)
        model.to(torch.device(device))
        return model, meta

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
    max_bytes: int = DEFAULT_MAX_BYTES,
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
        raw_state = ex.state if isinstance(ex.state, str) else str(ex.state)
        byte_ids, byte_mask = encode_bytes(
            raw_state.encode("utf-8", errors="replace"), max_len=max_bytes
        )
        seps = [i for i, t in enumerate(seq) if t == tok.sep_token_id]
        if len(seps) >= 3:
            s0, s1 = seps[-2] + 1, seps[-1]
        elif len(seps) == 2:
            s0, s1 = seps[-1] + 1, len(seq)
        else:
            s0, s1 = 0, len(seq)
        state_mask = [0] * len(seq)
        for i in range(s0, min(s1, len(seq))):
            state_mask[i] = 1
        c2f_eligible = (
            ex.source == "banking77"
            and ex.qtype == "choice"
            and isinstance(ex.criteria, dict)
            and tuple(ex.criteria.keys()) == BANKING77_LABELS
        )
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[ex.qtype],
                "label": label,
                "id": ex.id,
                "byte_ids": byte_ids[0],
                "byte_mask": byte_mask[0],
                "state_mask": state_mask,
                "c2f_eligible": c2f_eligible,
                "c2f_label": label if c2f_eligible else -1,
            }
        )
    return items


def collate_examples(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    result = collate_items([items], pad_id)
    if result is None:
        raise ValueError("cannot collate an empty decision batch")
    byte_len = max(
        (int(item["byte_ids"].numel()) for item in items if "byte_ids" in item),
        default=0,
    )
    if byte_len:
        byte_ids = torch.zeros((len(items), byte_len), dtype=torch.long)
        byte_mask = torch.zeros((len(items), byte_len), dtype=torch.long)
        for row, item in enumerate(items):
            if "byte_ids" not in item:
                continue
            values = item["byte_ids"].to(torch.long)
            mask = item.get("byte_mask")
            mask = (
                torch.ones_like(values)
                if mask is None
                else mask.to(torch.long)
            )
            n = min(byte_len, values.numel())
            byte_ids[row, :n] = values[:n]
            byte_mask[row, :n] = mask[:n]
        result["byte_ids"] = byte_ids
        result["byte_mask"] = byte_mask
    if all("state_mask" in item for item in items):
        width = result["input_ids"].size(1)
        state_mask = torch.zeros((len(items), width), dtype=torch.bool)
        for row, item in enumerate(items):
            values = torch.as_tensor(item["state_mask"], dtype=torch.bool)
            n = min(width, values.numel())
            state_mask[row, :n] = values[:n]
        result["state_mask"] = state_mask
    c2f_mask = torch.tensor(
        [bool(item.get("c2f_eligible", False)) for item in items], dtype=torch.bool
    )
    c2f_label = torch.tensor(
        [int(item.get("c2f_label", -1)) for item in items], dtype=torch.long
    )
    result["c2f_mask"] = c2f_mask
    result["c2f_label"] = c2f_label
    return result


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    mps = getattr(torch, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        get_state = getattr(mps, "get_rng_state", None)
        if callable(get_state):
            state["mps"] = get_state()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    mps = getattr(torch, "mps", None)
    mps_state = state.get("mps")
    if mps_state is not None and mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        set_state = getattr(mps, "set_rng_state", None)
        if callable(set_state):
            set_state(mps_state)


class PhaseASFT:
    """Optimizer loop over trainable params only (encoder base stays frozen)."""

    def __init__(
        self,
        model: NirnayTrainModel,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        cal_lambda: float = 0.005,
        pretrained_lr: float | None = None,
        concepts_lr: float | None = None,
    ):
        self.model = model
        self.cal_lambda = cal_lambda
        self.pretrained_lr = pretrained_lr
        self.concepts_lr = concepts_lr
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named:
            raise ValueError("no trainable parameters — check LoRA/additions wiring")
        # Pretrained laya parts (head/scorer/type_emb/embeddings) are sensitive
        # to Adam sign-drift: noisy-choice gradients diffuse them ~lr/step and
        # the readout collapses when displacement crosses ~0.1-0.2 (2026-09-25
        # bisection, cliff at steps 100-150). Fresh additions keep the full lr.
        # concepts (VQ codebook/commitment) blow up at full lr (ncp spike 22 at
        # step150) — a gentler group keeps the codebook stable.
        pretrained_prefixes = ("base.head.", "base.scorer.", "base.type_emb")

        def is_pretrained(n: str) -> bool:
            return n.startswith(pretrained_prefixes) or n.startswith("base.encoder.embeddings")

        groups = []
        if pretrained_lr is None and concepts_lr is None:
            groups = [{"params": [p for _, p in named], "lr": lr}]
        else:
            pre_bucket, concepts_bucket, rest = [], [], []
            for n, p in named:
                if pretrained_lr is not None and is_pretrained(n):
                    pre_bucket.append(p)
                elif concepts_lr is not None and n.startswith("concepts."):
                    concepts_bucket.append(p)
                else:
                    rest.append(p)
            if pre_bucket:
                groups.append({"params": pre_bucket, "lr": pretrained_lr or lr})
            if concepts_bucket:
                groups.append({"params": concepts_bucket, "lr": concepts_lr or lr})
            if rest:
                groups.append({"params": rest, "lr": lr})
        self.opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)

    def step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out = self.model(batch)
        parts = self.model.loss_from_batch(batch, out, cal_lambda=self.cal_lambda)
        if not bool(torch.isfinite(parts.total)):
            raise FloatingPointError(f"non-finite Phase A loss: {float(parts.total.detach())}")
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
            "deep_weighted": float(
                out.get("deep_weighted", parts.deep.detach() * 0.2).detach()
            ),
            "relational": float(parts.relational.detach()),
            "cal_ce": float(parts.cal_ce.detach()),
        }

    def fit(
        self,
        batches: list[dict[str, torch.Tensor]],
        steps: int,
        log_every: int = 50,
        *,
        start_step: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        checkpoint_meta: dict[str, Any] | None = None,
        history_path: str | Path | None = None,
        history_prefix: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        if not batches:
            raise ValueError("no batches — check mix size vs batch_size")
        if start_step < 0 or start_step > steps:
            raise ValueError("start_step must be between 0 and steps")
        history: list[dict[str, float]] = []
        t0 = time.perf_counter()
        executed = 0
        for step in range(start_step, steps):
            batch = batches[step % len(batches)]
            h = self.step(batch)
            history.append(h)
            executed += 1
            current = step + 1
            if checkpoint_path and checkpoint_every and (
                current % checkpoint_every == 0 or current == steps
            ):
                meta = dict(checkpoint_meta or {})
                meta["completed_steps"] = current
                if history_path:
                    history_file = Path(history_path)
                    history_file.parent.mkdir(parents=True, exist_ok=True)
                    _write_json_atomic(
                        history_file, list(history_prefix or []) + history
                    )
                self.save_checkpoint(str(checkpoint_path), meta=meta)
            if log_every and (
                current % log_every == 0 or step == start_step or current == steps
            ):
                elapsed = max(time.perf_counter() - t0, 1e-9)
                sec_per_step = elapsed / max(executed, 1)
                steps_per_sec = 1.0 / sec_per_step
                eta = (steps - current) * sec_per_step
                print(
                    f"step={current}/{steps} loss={h['loss']:.4f} "
                    f"choice_ce={h['choice_ce']:.4f} deep={h['deep']:.4f} "
                    f"ncp={h['ncp']:.4f} sec/step={sec_per_step:.3f} "
                    f"steps/s={steps_per_sec:.3f} eta={eta:.0f}s",
                    flush=True,
                )
        return history

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = self.model.load_checkpoint_payload(payload)
        if payload.get("optimizer") is not None:
            self.opt.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload.get("rng_state"))
        return meta

    def save_checkpoint(self, path: str, meta: dict[str, Any] | None = None) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.trainable_state_dict(),
            "optimizer": self.opt.state_dict(),
            "rng_state": _capture_rng_state(),
            "meta": meta or {},
        }
        temporary = p.with_name(p.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, p)
        side = p.with_suffix(p.suffix + ".meta.json")
        _write_json_atomic(side, meta or {})
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

    def sample_action(
        self, logits: torch.Tensor, marker_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian noise on valid logits; sample categorical action."""
        noise = torch.randn_like(logits) * self.noise_std
        noisy = logits + noise.masked_fill(~marker_mask, -1e4)
        probs = torch.softmax(noisy, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, probs

    def step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        from .losses import masked_cal_ce
        from .rlcd import (
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
        group_logits = logits.repeat_interleave(self.group_size, dim=0)
        group_mask = mmask.repeat_interleave(self.group_size, dim=0)
        group_labels = labels.repeat_interleave(self.group_size, dim=0)

        action, noisy_probs = self.sample_action(group_logits, group_mask)
        rewards = rlcd_sample_reward(noisy_probs, group_labels).detach()
        advantages = group_mean_advantages(rewards, self.group_size)
        log_probs = torch.log_softmax(
            group_logits.masked_fill(~group_mask, -1e4), dim=-1
        )
        choice_ce = torch.nn.functional.cross_entropy(
            logits.masked_fill(~mmask, -1e4),
            labels.clamp(min=0, max=logits.size(-1) - 1),
        )
        cal = masked_cal_ce(
            logits,
            labels.clamp(min=0, max=logits.size(-1) - 1),
            mmask,
        )
        total = rlcd_total_loss(
            choice_ce,
            cal,
            advantages,
            log_probs=log_probs,
            actions=action,
            cal_lambda=self.cal_lambda,
        )
        total = total + 0.3 * out["ncp"] + 0.2 * out["deep"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite Phase B loss: {float(total.detach())}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.opt.step()
        return {
            "loss": float(total.detach()),
            "reward_mean": float(rewards.mean()),
            "adv_mean": float(advantages.mean()),
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


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Safety net (2026-09-26): the 7000-step run trained past an eval collapse
# (step ~6431) and only failed at final eval — ~1h wasted plus a stale
# phase_a.pt. Probes bound the damage and leave phase_a_best.pt behind.
PROBE_ABORT_FACTOR = 0.5   # acc < 0.5 × best → collapse
PROBE_ABORT_MIN_BEST = 0.1 # never abort before any competence was seen
PROBE_NCP_SPIKE = 10.0     # healthy ncp ≈ 0-2; spike storms hit 4-31


class PhaseACollapse(RuntimeError):
    """Probe abort: eval collapsed mid-run; phase_a_best.pt holds the peak."""


@torch.no_grad()
def probe_checkpoint(
    model: "NirnayTrainModel",
    items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Read-only heldout probe: dense accuracy + VQ usage telemetry.

    Accuracy mirrors scripts/eval_checkpoint.py (rows label>=0 and >=2
    markers, argmax over valid markers only) so probe and final eval are the
    same metric on different splits. Usage stats come from forward codes
    (valid tokens only): unique codes and mode fraction per chunk — mode
    ≈ 0.97 precedes the collapse (healthy ≈ 0.27-0.49).
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    counts: list[torch.Tensor] | None = None
    if getattr(model, "concepts", None) is not None:
        k = model.concepts.config.codes_per_chunk
        n = model.concepts.config.num_chunks
        counts = [torch.zeros(k, dtype=torch.long) for _ in range(n)]
    try:
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            if len(chunk) < 2:
                continue
            batch = collate_examples(chunk, pad_id)
            batch = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            out = model(batch)
            logits = out["logits"].float()
            mmask = batch["marker_mask"]
            labels = batch["label"]
            for row in range(labels.shape[0]):
                label = int(labels[row])
                if label < 0:
                    continue
                count = int(mmask[row].sum())
                if count < 2:
                    continue
                total += 1
                if int(logits[row, :count].argmax()) == label:
                    correct += 1
            codes = out.get("codes")
            if codes is not None and counts is not None:
                att = batch["attention_mask"].bool()
                for i in range(codes.shape[-1]):
                    valid_codes = codes[:, :, i][att].detach().cpu()
                    counts[i] += torch.bincount(
                        valid_codes.flatten(), minlength=counts[i].numel()
                    )
    finally:
        if was_training:
            model.train()
    result: dict[str, Any] = {"acc": correct / max(total, 1), "n": total}
    if counts is not None:
        totals = [int(c.sum()) for c in counts]
        result["usage_unique"] = [int((c > 0).sum()) for c in counts]
        result["usage_mode_frac"] = [
            float(c.max()) / max(int(c.sum()), 1) for c in counts
        ]
        result["usage_tokens"] = totals[0] if totals else 0
    return result


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
    checkpoint_every: int = 250,
    resume: bool = False,
    overwrite: bool = False,
    log_every: int = 50,
    freeze_lora: bool = False,
    freeze_byte: bool = False,
    pretrained_lr: float | None = None,
    concepts_lr: float | None = None,
    probe_every: int = 0,
    probe_size: int = 256,
    code_usage_weight: float = 0.01,
    code_usage_temp: float = 0.1,
) -> dict[str, Any]:
    """CLI entry: build mix → encode → SFT → write history + checkpoint.

    probe_every > 0 splits training into segments and runs a heldout probe
    (dense acc + VQ usage) after each: saves phase_a_best.pt on improvement
    and raises PhaseACollapse on eval collapse / ncp spike (safety net for
    the 2026-09-26 death, which was only visible at final eval).
    """
    import json as _json

    from .data import build_phase_a_mix, split_train_heldout, write_freeze_manifest

    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if probe_every < 0:
        raise ValueError("probe_every must be non-negative")
    if probe_every and probe_size < 8:
        raise ValueError("probe_size must be at least 8 when probing")
    torch.manual_seed(seed)
    device = resolve_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "phase_a.pt"
    if checkpoint_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"{checkpoint_path} exists; pass resume=True or overwrite=True"
        )
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

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
    agent = load_agent(
        agent_path,
        device=device,
        enable_byte_path=False,
        nope_fraction=1.0 / 3.0,
    )
    model = NirnayTrainModel(
        agent.model,
        use_lora=True,
        lora_rank=lora_rank,
        use_concepts=True,
        use_deepsup=True,
        use_c2f=True,
        use_byte_path=True,
        max_bytes=DEFAULT_MAX_BYTES,
        n_labels_bank=77,
    )
    model.to(torch.device(device))
    if freeze_lora:
        for name, p in model.base.encoder.named_parameters():
            if "lora_" in name:
                p.requires_grad_(False)
    if freeze_byte and getattr(model, "byte_fusion", None) is not None:
        for p in model.byte_fusion.parameters():
            p.requires_grad_(False)
    if model.concepts is not None:
        # loss-only knobs (not in state_dict); mirrored into checkpoint meta
        model.concepts.config.code_usage_weight = code_usage_weight
        model.concepts.config.code_usage_temp = code_usage_temp
    enc_param = next(
        p for n, p in agent.model.encoder.named_parameters() if "lora_" not in n
    )
    enc_before = enc_param.detach().clone()

    items = encode_examples(train_ex, agent.tok, max_bytes=DEFAULT_MAX_BYTES)
    # Seeded shuffle: deterministic across resumes; kills label-homogeneous
    # batches (mix order is label-sorted → unshuffled batches were single-label).
    random.Random(seed).shuffle(items)
    dev = torch.device(device)
    batches: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        if len(chunk) < 2:
            continue
        batch = collate_examples(chunk, agent.tok.pad_token_id)
        batches.append(
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        )
    if not batches:
        raise RuntimeError("no batches — check mix size vs batch_size")

    trainer = PhaseASFT(model, lr=lr, pretrained_lr=pretrained_lr, concepts_lr=concepts_lr)
    start_step = 0
    prior_history: list[dict[str, float]] = []
    if resume:
        prior_meta = trainer.load_checkpoint(checkpoint_path)
        start_step = int(prior_meta.get("completed_steps", prior_meta.get("steps", 0)))
        history_path = out / "history.json"
        if history_path.exists():
            prior_history = list(_json.loads(history_path.read_text()))
        if len(prior_history) > start_step:
            prior_history = prior_history[:start_step]
    if start_step > steps:
        raise ValueError(f"checkpoint completed {start_step} steps, beyond target {steps}")

    checkpoint_meta = {
        "steps": steps,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "n_mix": len(mix),
        "trainable": model.trainable_summary()["trainable"],
        "encoder_frozen": True,
        "freeze_lora": freeze_lora,
        "freeze_byte": freeze_byte,
        "pretrained_lr": pretrained_lr,
        "concepts_lr": concepts_lr,
        "seed": seed,
        "nope_fraction": 1.0 / 3.0,
        "code_usage_weight": code_usage_weight,
        "code_usage_temp": code_usage_temp,
        "model_config": model.model_config(),
    }
    history: list[dict[str, float]] = []
    if probe_every > 0:
        labeled = [
            it
            for it in encode_examples(held_ex, agent.tok, max_bytes=DEFAULT_MAX_BYTES)
            if int(it.get("label", -1)) >= 0
        ]
        random.Random(seed + 7).shuffle(labeled)
        probe_items = labeled[:probe_size]
        if len(probe_items) < 8:
            raise RuntimeError(f"probe set too small: {len(probe_items)}")
        probe_path = out / "probes.json"
        probes: list[dict[str, Any]] = []
        if probe_path.exists():
            probes = list(_json.loads(probe_path.read_text()))
        best_path = out / "phase_a_best.pt"
        best_acc = max((float(p.get("acc", -1.0)) for p in probes), default=-1.0)
        seg_start = start_step
        while seg_start < steps:
            seg_end = min(seg_start + probe_every, steps)
            seg_hist = trainer.fit(
                batches,
                steps=seg_end,
                log_every=log_every,
                start_step=seg_start,
                checkpoint_path=checkpoint_path,
                checkpoint_every=checkpoint_every,
                checkpoint_meta=checkpoint_meta,
                history_path=out / "history.json",
                history_prefix=prior_history + history,
            )
            history.extend(seg_hist)
            probe = probe_checkpoint(
                model, probe_items, agent.tok.pad_token_id, dev,
                batch_size=min(batch_size, 4),
            )
            ncp_window = [h["ncp"] for h in seg_hist]
            probe["step"] = seg_end
            probe["ncp_mean"] = sum(ncp_window) / max(len(ncp_window), 1)
            probes.append(probe)
            _write_json_atomic(probe_path, probes)
            print(
                f"probe step={seg_end} acc={probe['acc']:.4f} n={probe['n']} "
                f"ncp_mean={probe['ncp_mean']:.4f} "
                f"usage_unique={probe.get('usage_unique')} "
                f"mode_frac={[round(m, 3) for m in probe.get('usage_mode_frac', [])]}",
                flush=True,
            )
            if probe["acc"] > best_acc:
                best_acc = probe["acc"]
                trainer.save_checkpoint(
                    str(best_path),
                    meta={
                        **checkpoint_meta,
                        "completed_steps": seg_end,
                        "probe_acc": best_acc,
                        "probe_step": seg_end,
                    },
                )
            if best_acc >= PROBE_ABORT_MIN_BEST and (
                probe["acc"] < PROBE_ABORT_FACTOR * best_acc
            ):
                raise PhaseACollapse(
                    f"step {seg_end}: probe acc {probe['acc']:.4f} < "
                    f"{PROBE_ABORT_FACTOR}x best {best_acc:.4f}; "
                    f"best checkpoint kept at {best_path}"
                )
            if probe["ncp_mean"] > PROBE_NCP_SPIKE:
                raise PhaseACollapse(
                    f"step {seg_end}: ncp_mean {probe['ncp_mean']:.2f} > "
                    f"{PROBE_NCP_SPIKE}; best checkpoint kept at {best_path}"
                )
            seg_start = seg_end
    else:
        history = trainer.fit(
            batches,
            steps=steps,
            log_every=log_every,
            start_step=start_step,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            checkpoint_meta=checkpoint_meta,
            history_path=out / "history.json",
            history_prefix=prior_history,
        )
    full_history = prior_history + history
    losses = [h["loss"] for h in full_history]
    if not losses or any(not math.isfinite(x) for x in losses):
        raise RuntimeError(f"non-finite or empty loss history: {losses[:3]}")

    if not torch.equal(enc_before, enc_param.detach()):
        raise RuntimeError("encoder base changed during Phase A — must stay frozen")

    meta = {
        "steps": steps,
        "completed_steps": steps,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "n_mix": len(mix),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "trainable": model.trainable_summary()["trainable"],
        "encoder_frozen": True,
        "freeze_lora": freeze_lora,
        "freeze_byte": freeze_byte,
        "pretrained_lr": pretrained_lr,
        "concepts_lr": concepts_lr,
        "seed": seed,
        "nope_fraction": 1.0 / 3.0,
        "code_usage_weight": code_usage_weight,
        "code_usage_temp": code_usage_temp,
        "model_config": model.model_config(),
    }
    _write_json_atomic(out / "history.json", full_history)
    ckpt = trainer.save_checkpoint(str(checkpoint_path), meta=meta)
    summary = model.trainable_summary()
    model.remove_hooks()
    return {
        "ckpt": ckpt,
        "history": full_history,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "encoder_frozen": True,
        "model_config": model.model_config(),
        "trainable": summary["trainable"],
        "byte_fusion": summary["byte_fusion"],
        "coarse_to_fine": summary["coarse_to_fine"],
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m nirnay.train` — Phase A CLI (Phase B via --phase b)."""
    import argparse

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
    ap.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--phase-a-path", default=None)
    ap.add_argument(
        "--freeze-lora",
        action="store_true",
        help="diagnostic: keep encoder LoRA frozen (ablation 2026-09-25)",
    )
    ap.add_argument(
        "--freeze-byte",
        action="store_true",
        help="diagnostic: freeze byte-fusion params (ablation 2026-09-25)",
    )
    ap.add_argument(
        "--pretrained-lr",
        type=float,
        default=None,
        help="lr for pretrained laya parts (head/scorer/type_emb/embeddings); "
        "default None = same as --lr",
    )
    ap.add_argument(
        "--concepts-lr",
        type=float,
        default=None,
        help="lr for concepts (VQ) params; default None = same as --lr",
    )
    ap.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="heldout probe cadence in steps; >0 enables best-checkpoint "
        "saving and collapse abort (safety net, 2026-09-26); 0 = disabled",
    )
    ap.add_argument("--probe-size", type=int, default=256)
    ap.add_argument("--code-usage-weight", type=float, default=0.01)
    ap.add_argument("--code-usage-temp", type=float, default=0.1)
    args = ap.parse_args(argv)

    if args.phase == "a":
        try:
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
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                overwrite=args.overwrite,
                log_every=args.log_every,
                freeze_lora=args.freeze_lora,
                freeze_byte=args.freeze_byte,
                pretrained_lr=args.pretrained_lr,
                concepts_lr=args.concepts_lr,
                probe_every=args.probe_every,
                probe_size=args.probe_size,
                code_usage_weight=args.code_usage_weight,
                code_usage_temp=args.code_usage_temp,
            )
        except PhaseACollapse as exc:
            print(f"PHASE_A_ABORT {exc}", flush=True)
            return 1
        h = result["history"]
        print(
            f"PHASE_A_OK steps={len(h)} loss0={h[0]['loss']:.4f} "
            f"lossN={h[-1]['loss']:.4f} ckpt={result['ckpt']} "
            f"train={result['n_train']} heldout={result['n_heldout']} "
            f"encoder_frozen=true"
        )
        return 0

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
        phase_a_path=args.phase_a_path,
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
    phase_a_path: str | None = None,
) -> dict[str, Any]:
    """Phase B RLCD after Phase A (loads phase_a.pt when available)."""
    import json as _json

    from .data import build_phase_a_mix, split_train_heldout, write_freeze_manifest

    if steps < 1:
        raise ValueError("steps must be positive")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    torch.manual_seed(seed)
    device = resolve_device(device)
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
    agent = load_agent(
        agent_path,
        device=device,
        enable_byte_path=False,
        nope_fraction=1.0 / 3.0,
    )
    if phase_a_path is None:
        phase_a = out.parent / "phase_a" / "phase_a.pt"
    else:
        phase_a = Path(phase_a_path)
    phase_a_meta: dict[str, Any] = {}
    if phase_a.exists():
        payload = torch.load(phase_a, map_location="cpu", weights_only=False)
        model, phase_a_meta = NirnayTrainModel.from_checkpoint(
            agent.model, payload, device=device
        )
    else:
        model = NirnayTrainModel(
            agent.model,
            use_lora=True,
            lora_rank=lora_rank,
            use_concepts=True,
            use_deepsup=True,
            use_c2f=True,
            use_byte_path=True,
            max_bytes=DEFAULT_MAX_BYTES,
            n_labels_bank=77,
        )
        model.to(torch.device(device))

    items = encode_examples(train_ex, agent.tok, max_bytes=model.max_bytes)
    random.Random(seed).shuffle(items)
    dev = torch.device(device)
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        if len(chunk) < 2:
            continue
        batch = collate_examples(chunk, agent.tok.pad_token_id)
        batches.append(
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        )
    if not batches:
        raise RuntimeError("no batches for Phase B")

    trainer = PhaseBRLCD(model, lr=lr, group_size=group_size)
    history = trainer.fit(batches, steps=steps)
    losses = [h["loss"] for h in history]
    if not losses or any(not math.isfinite(x) for x in losses):
        raise RuntimeError(f"non-finite or empty Phase B loss: {losses[:3]}")

    _write_json_atomic(out / "history.json", history)
    ckpt_path = out / "phase_b.pt"
    meta = {
        "steps": steps,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "seed": seed,
        "phase_a": str(phase_a) if phase_a.exists() else None,
        "phase_a_meta": phase_a_meta,
        "model_config": model.model_config(),
    }
    payload = {
        "model": model.trainable_state_dict(),
        "optimizer": trainer.opt.state_dict(),
        "rng_state": _capture_rng_state(),
        "meta": meta,
    }
    temporary = ckpt_path.with_name(ckpt_path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, ckpt_path)
    _write_json_atomic(ckpt_path.with_suffix(ckpt_path.suffix + ".meta.json"), meta)
    model.remove_hooks()
    return {
        "ckpt": str(ckpt_path),
        "history": history,
        "n_train": len(train_ex),
        "n_heldout": len(held_ex),
        "phase_a_loaded": phase_a.exists(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
