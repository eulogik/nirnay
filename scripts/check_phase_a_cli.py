"""G17: Phase A CLI — freeze manifest + finite history + frozen encoder ckpt."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import os

    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "phase_a"
        from nirnay.train import main as train_main

        rc = train_main(
            [
                "--phase", "a",
                "--steps", "3",
                "--batch-size", "6",
                "--n-synth", "24",
                "--banking-dir", "data/banking77",
                "--banking-limit", "48",
                "--out-dir", str(out_dir),
                "--lora-rank", "4",
                "--seed", "7",
            ]
        )
        if rc != 0:
            print(f"PHASE_A_CLI_FAIL exit={rc}", file=sys.stderr)
            return 1

        man = out_dir / "freeze_manifest.json"
        hist = out_dir / "history.json"
        ckpt = out_dir / "phase_a.pt"
        meta = out_dir / "phase_a.pt.meta.json"
        for p in (man, hist, ckpt, meta):
            if not p.exists() or p.stat().st_size < 10:
                print(f"PHASE_A_CLI_FAIL missing_or_empty={p.name}", file=sys.stderr)
                return 1

        m = json.loads(man.read_text())
        if m.get("n", 0) < 24 or not m.get("hashes"):
            print(f"PHASE_A_CLI_FAIL manifest_n={m.get('n')}", file=sys.stderr)
            return 1

        history = json.loads(hist.read_text())
        if len(history) < 3:
            print(f"PHASE_A_CLI_FAIL history_len={len(history)}", file=sys.stderr)
            return 1
        losses = [h["loss"] for h in history]
        if any(x != x or x in (float("inf"), float("-inf")) for x in losses):
            print(f"PHASE_A_CLI_FAIL nonfinite={losses}", file=sys.stderr)
            return 1

        mm = json.loads(meta.read_text())
        if not mm.get("encoder_frozen"):
            print("PHASE_A_CLI_FAIL meta_encoder_frozen", file=sys.stderr)
            return 1
        if mm.get("trainable", 0) <= 0:
            print(f"PHASE_A_CLI_FAIL trainable={mm.get('trainable')}", file=sys.stderr)
            return 1

        import torch

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model" not in blob or not blob["model"]:
            print("PHASE_A_CLI_FAIL ckpt_keys", file=sys.stderr)
            return 1
        ckpt_bytes = ckpt.stat().st_size

    print(
        f"PHASE_A_CLI_OK steps={len(history)} loss0={losses[0]:.4f} "
        f"lossN={losses[-1]:.4f} freeze_n={m['n']} ckpt_bytes={ckpt_bytes} "
        f"encoder_frozen=true trainable={mm['trainable']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
