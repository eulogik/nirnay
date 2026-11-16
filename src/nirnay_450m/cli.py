"""Command line interface.

    nirnay classify "My card was charged twice" --labels banking77
    nirnay classify --input data.jsonl --output out.jsonl --labels labels.json
    nirnay serve --port 8000
    nirnay doctor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str | None, rows: Iterable[dict[str, Any]]) -> None:
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = open(target, "w", encoding="utf-8")
    else:
        fh = sys.stdout
    try:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if path:
            fh.close()


def _classify(args: argparse.Namespace) -> int:
    from .classifier import Classifier

    clf = Classifier(
        model=args.repo,
        checkpoint=args.checkpoint,
        device=args.device,
        base_model=args.base_model,
    )

    top_k = args.top_k or None
    if args.input:
        def results() -> Iterable[dict[str, Any]]:
            for record in _read_jsonl(args.input):
                decision = clf.classify(
                    record["state"],
                    args.labels,
                    instructions=record.get("instructions", args.instructions),
                    threshold=args.threshold,
                    top_k=top_k,
                )
                yield {**record, **decision.as_dict()}

        _write_jsonl(args.output, results())
        return 0

    text = args.text
    if text is None:
        if sys.stdin.isatty():
            print("Provide text, --input, or pipe text on stdin.", file=sys.stderr)
            return 2
        text = sys.stdin.read().strip()
    if not text:
        print("Empty input.", file=sys.stderr)
        return 2

    decision = clf.classify(
        text,
        args.labels,
        instructions=args.instructions,
        threshold=args.threshold,
        top_k=top_k,
    )
    if args.json:
        print(json.dumps(decision.as_dict(), ensure_ascii=False))
    else:
        print(f"{decision.choice} ({decision.confidence:.2f})")
        if decision.escalated:
            print("escalate: true")
        for label, prob in decision.top[1:]:
            print(f"  {label} {prob:.3f}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    if args.checkpoint:
        os.environ["NIRNAY_CHECKPOINT"] = str(Path(args.checkpoint).expanduser())
    if args.device:
        os.environ["NIRNAY_DEVICE"] = args.device
    if args.base_model:
        os.environ["NIRNAY_MODEL"] = args.base_model
    if args.repo:
        os.environ["NIRNAY_REPO"] = args.repo

    import uvicorn

    from .server import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _doctor(_: argparse.Namespace) -> int:
    import platform

    import torch

    from . import __version__
    from .classifier import DEFAULT_REPO, Classifier

    print(f"nirnay-450m {__version__}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available: {bool(mps and mps.is_available())}")
    print(f"model repo: {DEFAULT_REPO}")
    try:
        print(f"checkpoint: {Classifier().resolve_checkpoint()}")
    except Exception as exc:  # noqa: BLE001
        print(f"checkpoint: unavailable ({type(exc).__name__}: {exc})")
    print(f"hf home: {os.environ.get('HF_HOME', 'default')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nirnay",
        description="Local calibrated decisions with NIRNAY 450M.",
    )
    parser.add_argument("--version", action="version", version="nirnay-450m")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("classify", help="classify one state or a JSONL batch")
    c.add_argument("text", nargs="?", help="state to classify (omit to read stdin)")
    c.add_argument("--labels", default="banking77",
                   help='"banking77", a JSON file, or inline JSON')
    c.add_argument("--input", help="JSONL with state/instructions rows")
    c.add_argument("--output", help="JSONL output path (default stdout)")
    c.add_argument("--instructions", default=None)
    c.add_argument("--threshold", type=float, default=None,
                   help="mark escalated when confidence is below this")
    c.add_argument("--top-k", type=int, default=0)
    c.add_argument("--json", action="store_true")
    c.add_argument("--checkpoint", default=None)
    c.add_argument("--device", default=None)
    c.add_argument("--repo", default="eulogik/nirnay-450m")
    c.add_argument("--base-model", default="convaiinnovations/laya")
    c.set_defaults(func=_classify)

    s = sub.add_parser("serve", help="Jev-compatible HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--repo", default=None)
    s.add_argument("--base-model", default=None)
    s.set_defaults(func=_serve)

    d = sub.add_parser("doctor", help="print environment and model status")
    d.set_defaults(func=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
