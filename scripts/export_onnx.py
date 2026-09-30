"""Export a trained NIRNAY checkpoint to ONNX (+ parity + latency report).

Serving graph only (eval mode, no labels): logits + act_logits out.
Verifies numerics across input classes (batch sizes, lengths, c2f
eligible/ineligible rows) and reports CPU latency vs the torch baseline.
Writes <out>.report.json next to the artifact.

Example:
  uv run python scripts/export_onnx.py --checkpoint artifacts/phase_b/phase_b.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export NIRNAY checkpoint to ONNX")
    p.add_argument("--checkpoint", default="artifacts/phase_b/phase_b.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--skip-verify", action="store_true")
    return p


def main() -> int:
    import torch

    args = build_parser().parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    out = Path(args.out) if args.out else ckpt.with_suffix(".onnx")

    import os

    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from nirnay.agent import NirnayAgent
    from nirnay.data import load_banking77_csv
    from nirnay.train import collate_examples, encode_examples

    agent = NirnayAgent(
        device=args.device, checkpoint_path=str(ckpt), enable_byte_path=False
    )
    model = agent.train_model
    model.eval()

    class ServeWrapper(torch.nn.Module):
        # Pads marker slots to 77 (the c2f bank width). Exact: padded slots
        # are mask-filled to -1e4 everywhere downstream and c2f eligibility
        # counts only real markers (verified zero-contribution in parity).
        # Sequence stays native length: the byte path interpolates patches
        # to the sequence length, so padding tokens would perturb every
        # position (measured 7.2 max abs). Only batch is dynamic; bytes are
        # always 512 by construction.
        N_SLOTS = 77

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(
            self, input_ids, attention_mask, marker_pos, marker_mask,
            qtype, byte_ids, byte_mask, state_mask, c2f_mask,
        ):
            # Branch-free pad (possibly empty): no Python `if` on shapes,
            # dtype-preserving via new_zeros.
            B = marker_pos.size(0)
            marker_pos = torch.cat(
                [marker_pos,
                 marker_pos.new_zeros((B, self.N_SLOTS - marker_pos.size(1)))], dim=1)
            marker_mask = torch.cat(
                [marker_mask, marker_mask.new_zeros((B, self.N_SLOTS - marker_mask.size(1)))], dim=1)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "marker_pos": marker_pos,
                "marker_mask": marker_mask,
                "qtype": qtype,
                "byte_ids": byte_ids,
                "byte_mask": byte_mask,
                "state_mask": state_mask,
                "c2f_mask": c2f_mask,
            }
            out = self.m(batch, compute_aux=False)
            return out["logits"], out["act_logits"]

    wrapped = ServeWrapper(model).eval()
    test = load_banking77_csv("data/banking77/test.csv", split="test")
    items = encode_examples(test[:20], agent.tok, max_bytes=model.max_bytes)
    from nirnay.data import synthetic_policy_examples

    items += encode_examples(
        synthetic_policy_examples(n=32, seed=7), agent.tok,
        max_bytes=model.max_bytes)

    def to_batch(idxs, device="cpu"):
        chunk = [items[i] for i in idxs]
        b = collate_examples(chunk, agent.tok.pad_token_id)
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in b.items()
        }

    ex = to_batch([0, 1, 2, 3])
    # Pad the example to the caps so static dims pin at serving maxima
    # (batch stays symbolic; nothing legal exceeds these).
    need = 512 - ex["input_ids"].size(1)
    if need > 0:
        ex["input_ids"] = torch.cat(
            [ex["input_ids"], ex["input_ids"].new_zeros((4, need))], dim=1)
        ex["attention_mask"] = torch.cat(
            [ex["attention_mask"], ex["attention_mask"].new_zeros((4, need))], dim=1)
        ex["state_mask"] = torch.cat(
            [ex["state_mask"], ex["state_mask"].new_zeros((4, need))], dim=1)
    ex_args = (
        ex["input_ids"], ex["attention_mask"], ex["marker_pos"],
        ex["marker_mask"], ex["qtype"], ex["byte_ids"], ex["byte_mask"],
        ex["state_mask"], ex["c2f_mask"],
    )
    print("exporting (torch.export + onnx, opset %d) ..." % args.opset, flush=True)
    try:
        from torch.export import Dim

        batch = Dim("batch", max=32)
        seq = Dim("seq", max=512)
        # slots static at 77: the wrapper pads markers (verified exact).
        # bytes static at 512: encode_bytes always pads to max_len.
        ep_shapes = {
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "marker_pos": {0: batch},
            "marker_mask": {0: batch},
            "qtype": {0: batch},
            "byte_ids": {0: batch},
            "byte_mask": {0: batch},
            "state_mask": {0: batch, 1: seq},
            "c2f_mask": {0: batch},
        }
        ep = torch.export.export(wrapped, ex_args, dynamic_shapes=ep_shapes)
        torch.onnx.export(
            ep,
            f=str(out),
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask",
                         "qtype", "byte_ids", "byte_mask", "state_mask", "c2f_mask"],
            output_names=["logits", "act_logits"],
            opset_version=args.opset,
        )
        mode = "dynamo-dynamic"
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ONNX export failed ({type(e).__name__}: {str(e)[:500]})")
    print(f"EXPORT_OK path={out} mode={mode} bytes={out.stat().st_size}")

    if args.skip_verify:
        return 0

    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    def pad_feed(b):
        # Pad markers to the artifact's static 77 slots; sequence stays native.
        out = dict(b)
        B = b["input_ids"].size(0)
        pm = 77 - b["marker_pos"].size(1)
        out["marker_pos"] = torch.cat(
            [b["marker_pos"], b["marker_pos"].new_zeros((B, pm))], dim=1)
        out["marker_mask"] = torch.cat(
            [b["marker_mask"], b["marker_mask"].new_zeros((B, pm))], dim=1)
        return {n: out[n].cpu().numpy() for n in in_names}

    def run_ort(idxs):
        b = to_batch(idxs)
        return sess.run(None, pad_feed(b))

    worst_abs, agree, total = 0.0, 0, 0
    classes = {"b4": [0, 1, 2, 3], "b1": [4], "b2": [5, 6], "b8a": list(range(8)),
               "b8b": list(range(8, 16)), "b4long": [16, 17, 18, 19],
               "syn_choice": [20, 23], "syn_score": [21, 24],
               "syn_noul": [22, 25], "syn_mix": [20, 21, 22, 23]}
    per_class = {}
    # Leg 1: raw torch forward (compute_aux on, unpadded) vs wrapped
    # (padded, aux off) must match: proves pad-exactness + flag safety.
    leg1_worst = 0.0
    leg1_agree, leg1_total = 0, 0
    with torch.no_grad():
        for name, idxs in classes.items():
            b = to_batch(idxs)
            raw = model(b)["logits"].float().cpu().numpy()
            wrap = wrapped(*[b[n] for n in in_names])[0].float().cpu().numpy()
            w = raw.shape[1]
            leg1_worst = max(leg1_worst, float(np.abs(raw - wrap[:, :w]).max()))
            mm = b["marker_mask"].cpu().numpy()
            lb = b["label"].cpu().numpy()
            for r in range(lb.shape[0]):
                if int(lb[r]) < 0:
                    continue
                c = int(mm[r].sum())
                if c < 2:
                    continue
                leg1_total += 1
                if int(raw[r, :c].argmax()) == int(wrap[r, :c].argmax()):
                    leg1_agree += 1
    print(f"LEG1 raw-vs-wrapped max_abs={leg1_worst:.2e} argmax={leg1_agree}/{leg1_total}")
    with torch.no_grad():
        for name, idxs in classes.items():
            b = to_batch(idxs)
            ref = wrapped(*[b[n] for n in in_names])[0].float().cpu().numpy()
            got = run_ort(idxs)[0]
            d = float(np.abs(ref - got).max())
            mm = b["marker_mask"].cpu().numpy()
            lb = b["label"].cpu().numpy()
            a = t = 0
            for r in range(lb.shape[0]):
                if int(lb[r]) < 0:
                    continue
                c = int(mm[r].sum())
                if c < 2:
                    continue
                t += 1
                if int(ref[r, :c].argmax()) == int(got[r, :c].argmax()):
                    a += 1
            per_class[name] = {"max_abs": d, "agree": f"{a}/{t}"}
            worst_abs = max(worst_abs, d)
            agree += a
            total += t
    print(f"PARITY max_abs={worst_abs:.2e} argmax_agree={agree}/{total}")
    for name, stats in per_class.items():
        print(f"  {name}: {stats}")

    # latency: batch-1 banking row, same protocol as the torch measurement
    one = to_batch([0])
    feed1 = pad_feed(one)
    for _ in range(5):
        sess.run(None, feed1)
    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        sess.run(None, feed1)
    ort_ms = (time.perf_counter() - t0) / N * 1000.0
    with torch.no_grad():
        for _ in range(5):
            wrapped(*[one[n] for n in in_names])
        t0 = time.perf_counter()
        for _ in range(N):
            wrapped(*[one[n] for n in in_names])
    torch_ms = (time.perf_counter() - t0) / N * 1000.0
    print(f"LATENCY ort_cpu={ort_ms:.1f}ms torch_cpu={torch_ms:.1f}ms")
    report = {
        "checkpoint": str(ckpt),
        "artifact": str(out),
        "export_mode": mode,
        "opset": args.opset,
        "parity_max_abs": worst_abs,
        "argmax_agree": f"{agree}/{total}",
        "per_class": per_class,
        "ort_cpu_ms": ort_ms,
        "torch_cpu_ms": torch_ms,
    }
    rep_path = out.with_suffix(out.suffix + ".report.json")
    rep_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"REPORT_OK {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
