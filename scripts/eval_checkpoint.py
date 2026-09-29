from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained NIRNAY checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/banking77")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise SystemExit("batch-size must be at least 2")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from nirnay.agent import NirnayAgent
    from nirnay.data import load_banking77_csv
    from nirnay.eval import evaluate_logits
    from nirnay.train import collate_examples, encode_examples

    data_dir = Path(args.data_dir)
    test_path = data_dir / "test.csv"
    if not test_path.exists():
        raise SystemExit(f"test data not found: {test_path}")
    examples = load_banking77_csv(test_path, split="test")
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("no evaluation examples")

    device = resolve_device(args.device)
    agent = NirnayAgent(
        device=device,
        checkpoint_path=str(checkpoint),
        enable_byte_path=False,
    )
    if agent.train_model is None:
        raise SystemExit("checkpoint did not load a trained model")
    items = encode_examples(
        examples,
        agent.tok,
        max_bytes=agent.train_model.max_bytes,
    )
    all_logits: list[np.ndarray] = []
    all_targets: list[int] = []
    max_options = 0
    agent.train_model.eval()
    with torch.no_grad():
        for start in range(0, len(items), args.batch_size):
            batch = collate_examples(
                items[start : start + args.batch_size], agent.tok.pad_token_id
            )
            batch = {
                key: (value.to(agent.device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            output = agent.train_model(batch)
            logits = output["logits"].detach().float().cpu().numpy()
            marker_mask = batch["marker_mask"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            for row, label in enumerate(labels):
                if int(label) < 0:
                    continue
                count = int(marker_mask[row].sum())
                if count < 2:
                    raise SystemExit(f"invalid marker count: {count}")
                all_logits.append(logits[row, :count])
                all_targets.append(int(label))
                max_options = max(max_options, count)

    width = max_options
    padded = np.full((len(all_logits), width), -1e4, dtype=np.float32)
    for row, values in enumerate(all_logits):
        padded[row, : values.shape[0]] = values
    metrics = evaluate_logits(padded, np.asarray(all_targets, dtype=np.int64))
    result = {
        "checkpoint": str(checkpoint),
        "data": str(test_path),
        "device": device,
        **metrics.as_dict(),
    }
    output_path = Path(args.out) if args.out else checkpoint.with_name("eval_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(
        f"EVAL_CHECKPOINT_OK n={metrics.n} accuracy={metrics.accuracy:.4f} "
        f"brier={metrics.brier:.4f} ece_raw={metrics.ece_raw:.4f} "
        f"ece_fitted={metrics.ece_fitted:.4f} temperature={metrics.temperature:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
