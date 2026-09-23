"""G16: real Banking77 CSVs → §2 examples, ≥10k, 77 labels, hash freeze, 80/20."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / "data" / "banking77"


def main() -> int:
    from nirnay.data import (
        BANKING77_LABELS,
        freeze_hashes,
        load_banking77_dir,
        split_train_heldout,
        to_laya_record,
    )

    if not BANK_DIR.exists():
        print(
            f"BANKING77_DATA_FAIL missing={BANK_DIR} "
            "(aria2c PolyAI-LDN banking_data train.csv+test.csv)",
            file=sys.stderr,
        )
        return 1

    examples = load_banking77_dir(BANK_DIR)
    if len(examples) < 10_000:
        print(f"BANKING77_DATA_FAIL n={len(examples)} need>=10000", file=sys.stderr)
        return 1

    answers = {str(e.answer) for e in examples}
    missing = set(BANKING77_LABELS) - answers
    if missing:
        print(
            f"BANKING77_DATA_FAIL missing_labels={len(missing)} e.g. {sorted(missing)[:5]}",
            file=sys.stderr,
        )
        return 1
    if any(e.qtype != "choice" for e in examples):
        print("BANKING77_DATA_FAIL non_choice", file=sys.stderr)
        return 1
    if any(e.source != "banking77" for e in examples):
        print("BANKING77_DATA_FAIL bad_source", file=sys.stderr)
        return 1

    h1 = freeze_hashes(examples)
    h2 = freeze_hashes(examples)
    if h1 != h2 or len(h1) != len(examples):
        print("BANKING77_DATA_FAIL hash_unstable", file=sys.stderr)
        return 1
    # Content-addressed: flip one state
    ex0 = examples[0]
    h0 = h1[ex0.id]
    ex0.state = ex0.state + "x"
    if freeze_hashes([ex0])[ex0.id] == h0:
        print("BANKING77_DATA_FAIL hash_not_content", file=sys.stderr)
        return 1
    ex0.state = ex0.state[:-1]

    train, held = split_train_heldout(examples, heldout_frac=0.2, seed=13)
    t_ids = {e.id for e in train}
    h_ids = {e.id for e in held}
    if t_ids & h_ids:
        print("BANKING77_DATA_FAIL overlap", file=sys.stderr)
        return 1
    if len(train) + len(held) != len(examples):
        print("BANKING77_DATA_FAIL split_size", file=sys.stderr)
        return 1
    frac = len(held) / len(examples)
    if not (0.15 <= frac <= 0.25):
        print(f"BANKING77_DATA_FAIL frac={frac:.3f}", file=sys.stderr)
        return 1

    banned = {"jev", "jevbench", "distill", "api_output"}
    for e in examples[:20]:
        blob = str(to_laya_record(e)).lower()
        if any(b in blob for b in banned):
            print(f"BANKING77_DATA_FAIL banned id={e.id}", file=sys.stderr)
            return 1

    print(
        f"BANKING77_DATA_OK n={len(examples)} labels=77 train={len(train)} "
        f"heldout={len(held)} hashes_stable=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
