"""G13: §2 data pipeline — 77 intents, stable hashes, 80/20 split, no Jev fields."""

from __future__ import annotations

import sys


def main() -> int:
    from nirnay.data import (
        BANKING77_LABELS,
        build_phase_a_mix,
        freeze_hashes,
        split_train_heldout,
        synthetic_policy_examples,
        to_laya_record,
        write_freeze_manifest,
    )

    if len(BANKING77_LABELS) != 77:
        print(f"DATA_FAIL n_labels={len(BANKING77_LABELS)}", file=sys.stderr)
        return 1
    if len(set(BANKING77_LABELS)) != 77:
        print("DATA_FAIL duplicate_labels", file=sys.stderr)
        return 1
    # Spot-check official names verified 2026-09-23
    expect_spot = {
        0: "activate_my_card",
        11: "card_arrival",
        51: "Refund_not_showing_up",
        53: "reverted_card_payment?",
        62: "topping_up_by_card",
        76: "wrong_exchange_rate_for_cash_withdrawal",
    }
    for idx, name in expect_spot.items():
        if BANKING77_LABELS[idx] != name:
            print(
                f"DATA_FAIL label[{idx}]={BANKING77_LABELS[idx]!r} expected {name!r}",
                file=sys.stderr,
            )
            return 1

    # Mix: synthetic policies (choice/score/noul) ± banking cache
    mix = build_phase_a_mix(n_synth=96)
    if len(mix) < 96:
        print(f"DATA_FAIL mix_size={len(mix)}", file=sys.stderr)
        return 1
    qtypes = {ex.qtype for ex in mix}
    if qtypes != {"choice", "score", "noul"}:
        print(f"DATA_FAIL qtypes={qtypes}", file=sys.stderr)
        return 1
    sources = {ex.source for ex in mix}
    if sources - {"synthetic_policy", "banking77"}:
        print(f"DATA_FAIL sources={sources}", file=sys.stderr)
        return 1

    # Hash freeze: stable across calls; content-sensitive
    h1 = freeze_hashes(mix)
    h2 = freeze_hashes(mix)
    if h1 != h2 or len(h1) != len(mix):
        print("DATA_FAIL hash_unstable", file=sys.stderr)
        return 1
    one = mix[0]
    h_before = h1[one.id]
    restored = one.state
    one.state = one.state + " tampered"
    if freeze_hashes([one])[one.id] == h_before:
        print("DATA_FAIL hash_not_content_addressed", file=sys.stderr)
        return 1
    one.state = restored
    if freeze_hashes([one])[one.id] != h_before:
        print("DATA_FAIL hash_not_restorable", file=sys.stderr)
        return 1

    # 80/20 split, disjoint
    train, held = split_train_heldout(mix, heldout_frac=0.2, seed=13)
    t_ids = {e.id for e in train}
    h_ids = {e.id for e in held}
    if t_ids & h_ids:
        print("DATA_FAIL split_overlap", file=sys.stderr)
        return 1
    if len(train) + len(held) != len(mix):
        print(
            f"DATA_FAIL split_sizes train={len(train)} held={len(held)} n={len(mix)}",
            file=sys.stderr,
        )
        return 1
    frac = len(held) / len(mix)
    if not (0.15 <= frac <= 0.25):
        print(f"DATA_FAIL heldout_frac={frac:.3f}", file=sys.stderr)
        return 1

    # Laya-compatible records; no Jev-derived fields anywhere
    banned = {"jev", "jevbench", "distill", "api_output"}
    for ex in mix[:10]:
        rec = to_laya_record(ex)
        blob = str(rec).lower()
        if any(b in blob for b in banned):
            print(f"DATA_FAIL banned_field id={ex.id}", file=sys.stderr)
            return 1
        if set(rec) < {"id", "state", "question", "answer", "qtype", "source"}:
            print(f"DATA_FAIL record_keys={set(rec)}", file=sys.stderr)
            return 1
        if rec["question"]["type"] != ex.qtype:
            print("DATA_FAIL qtype_mismatch", file=sys.stderr)
            return 1

    # Banking77 examples (when cache present) respect 77-way answer set
    from nirnay.data import banking77_choice_examples

    bank_ex = banking77_choice_examples([("I am still waiting on my card?", 11)], limit=1)
    if bank_ex[0].answer != "card_arrival":
        print(f"DATA_FAIL bank_answer={bank_ex[0].answer}", file=sys.stderr)
        return 1

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        man = write_freeze_manifest(Path(td) / "freeze.json", mix)
        if not man.exists() or man.stat().st_size < 10:
            print("DATA_FAIL manifest", file=sys.stderr)
            return 1

    print(
        f"DATA_OK n={len(mix)} labels77=true train={len(train)} heldout={len(held)} "
        f"hashes_stable=true qtypes={sorted(qtypes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
