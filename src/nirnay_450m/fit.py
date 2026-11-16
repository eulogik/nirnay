"""Light fine-tuning for your own labels.

Freezes the encoder and concept stack, trains the choice scorer on a JSONL
of (state, label) pairs, fits a temperature on a heldout split, and writes
a checkpoint that `Classifier(checkpoint=..., temps_path=...)` can load.

This is deliberately narrow: it adapts the readout, it does not retrain the
backbone. That is what makes it minutes on a laptop instead of hours.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .classifier import DEFAULT_BASE_MODEL, DEFAULT_REPO, Classifier, normalise_labels
from .data import DecisionExample
from .eval import evaluate_logits
from .losses import choice_ce
from .temps import fit_temperature, temp_bucket, write_temps
from .train import collate_examples, encode_examples

DEFAULT_INSTRUCTION = "Classify the state into exactly one of the labels."


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _infer_labels(rows: list[dict[str, Any]]) -> dict[str, str]:
    labels = sorted({str(r["label"]) for r in rows})
    if len(labels) < 2:
        raise ValueError("need at least two distinct labels")
    return {lab: lab.replace("_", " ") for lab in labels}


def _split(
    rows: list[dict[str, Any]], seed: int, heldout_frac: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_held = max(1, int(round(len(group) * heldout_frac))) if len(group) >= 2 else 0
        held.extend(group[:n_held])
        train.extend(group[n_held:])
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def _examples(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    instructions: str,
) -> list[DecisionExample]:
    out = []
    for i, row in enumerate(rows):
        label = str(row["label"])
        if label not in labels:
            raise ValueError(f"label {label!r} is not in the labels file")
        out.append(
            DecisionExample(
                id=f"custom-{i:05d}",
                source="custom",
                qtype="choice",
                state=row["state"],
                instructions=row.get("instructions", instructions),
                answer=label,
                criteria=labels,
            )
        )
    return out


def _batches(items, tok, pad_id, device, batch_size):
    batches = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        if not chunk:
            continue
        batch = collate_examples(chunk, pad_id)
        batches.append({
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        })
    return batches


def _collect_logits(model, batches) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            out = model(batch, compute_aux=False)
            logits = out["logits"].detach().float().cpu().numpy()
            mm = batch["marker_mask"].detach().cpu().numpy()
            lb = batch["label"].detach().cpu().numpy()
            for r, label in enumerate(lb):
                if int(label) < 0:
                    continue
                count = int(mm[r].sum())
                if count < 2:
                    continue
                rows.append(logits[r, :count])
                targets.append(int(label))
    width = max((x.shape[0] for x in rows), default=0)
    padded = np.full((len(rows), width), -1e4, dtype=np.float32)
    for i, values in enumerate(rows):
        padded[i, : values.shape[0]] = values
    return padded, np.asarray(targets, dtype=np.int64)


def fit_custom(
    data: str | Path,
    *,
    labels: Any = None,
    out: str | Path = "nirnay_custom.pt",
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 13,
    device: str | None = None,
    base_checkpoint: str | Path | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    repo: str = DEFAULT_REPO,
    instructions: str = DEFAULT_INSTRUCTION,
    heldout_frac: float = 0.2,
) -> dict[str, Any]:
    rows = _read_jsonl(data)
    label_map = normalise_labels(labels) if labels is not None else _infer_labels(rows)
    if len(label_map) < 2:
        raise ValueError("need at least two labels")
    train_rows, held_rows = _split(rows, seed=seed, heldout_frac=heldout_frac)
    if not train_rows:
        raise ValueError("no training rows after split")

    from .agent import NirnayAgent

    ckpt = str(base_checkpoint) if base_checkpoint else Classifier(model=repo).resolve_checkpoint()
    agent = NirnayAgent(
        base_model,
        device=device,
        checkpoint_path=ckpt,
        enable_byte_path=False,
    )
    model = agent.train_model
    assert model is not None

    train_items = encode_examples(
        _examples(train_rows, label_map, instructions), agent.tok,
        max_bytes=model.max_bytes,
    )
    held_items = encode_examples(
        _examples(held_rows, label_map, instructions), agent.tok,
        max_bytes=model.max_bytes,
    ) if held_rows else []
    train_batches = _batches(train_items, agent.tok, agent.tok.pad_token_id, agent.device, batch_size)
    held_batches = _batches(held_items, agent.tok, agent.tok.pad_token_id, agent.device, batch_size)
    if not train_batches:
        raise ValueError("no training batches")

    before_logits, before_targets = (
        _collect_logits(model, held_batches) if held_batches else (None, None)
    )
    before_metrics = (
        evaluate_logits(before_logits, before_targets) if before_logits is not None else None
    )

    save_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.base.scorer.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [p for p in model.base.scorer.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    model.eval()  # frozen encoder: eval mode avoids MPS SDPA dropout and is correct here
    best_state = None
    best_acc = -1.0
    eval_every = max(10, steps // 10)
    last_loss = 0.0
    for step in range(1, steps + 1):
        batch = train_batches[(step - 1) % len(train_batches)]
        opt.zero_grad(set_to_none=True)
        fit_out = model(batch, compute_aux=False)
        loss = choice_ce(fit_out["logits"], batch["label"], batch["marker_mask"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite fit loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.base.scorer.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        last_loss = float(loss.detach())
        if held_batches and step % eval_every == 0:
            logits, targets = _collect_logits(model, held_batches)
            acc = float(evaluate_logits(logits, targets).accuracy)
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.base.scorer.state_dict().items()
                }
    if best_state is not None:
        model.base.scorer.load_state_dict(best_state)

    after_logits, after_targets = (
        _collect_logits(model, held_batches) if held_batches else (None, None)
    )
    after_metrics = (
        evaluate_logits(after_logits, after_targets) if after_logits is not None else None
    )

    temp = None
    temps_path = None
    if after_logits is not None and len(after_targets) > 0:
        onehot = np.zeros((len(after_targets), after_logits.shape[1]), dtype=np.float64)
        onehot[np.arange(len(after_targets)), after_targets] = 1.0
        temp = float(fit_temperature(after_logits, onehot))
        temps_path = Path(out).with_suffix(".temps.json")
        write_temps(temps_path, {temp_bucket("choice", len(label_map)): temp})

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    payload = {
        "model": {n: state[n].detach().cpu() for n in save_names},
        "meta": {
            "custom": True,
            "labels": label_map,
            "steps": steps,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "n_train": len(train_rows),
            "n_heldout": len(held_rows),
            "heldout_accuracy_before": float(before_metrics.accuracy) if before_metrics else None,
            "heldout_accuracy_after": float(after_metrics.accuracy) if after_metrics else None,
            "best_heldout_accuracy": best_acc if best_state is not None else None,
            "final_loss": last_loss,
            "temperature": temp,
            "base_checkpoint": ckpt,
            "base_model": base_model,
            "nope_fraction": 1.0 / 3.0,
            "model_config": model.model_config(),
        },
    }
    temporary = out_path.with_name(out_path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(out_path)
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(payload["meta"], indent=2) + "\n"
    )

    return {
        "checkpoint": str(out_path),
        "temps": str(temps_path) if temps_path else None,
        "labels": label_map,
        "before": before_metrics.as_dict() if before_metrics else None,
        "after": after_metrics.as_dict() if after_metrics else None,
        "n_train": len(train_rows),
        "n_heldout": len(held_rows),
        "temperature": temp,
    }
