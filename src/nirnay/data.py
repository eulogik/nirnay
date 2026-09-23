"""§2 data pipeline: decision examples, hash freeze, 80/20 split.

Sources per plan §2 (no Jev outputs ever): Banking77 + synthetic policies.
Additional public sets (AG News, SST-2, …) plug in via `source=` records when
present; this module owns the DecisionExample schema, official Banking77
intent list (CC-BY-4.0, verified 2026-09-23), content hashing, and the
train/heldout split used for temperature fitting.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

# Official Banking77 intent names (HF PolyAI/banking77 / banking77 README, verified 2026-09-23).
BANKING77_LABELS: tuple[str, ...] = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

QType = Literal["choice", "score", "noul"]


@dataclass
class DecisionExample:
    """One labeled decision: state + typed question + gold answer."""

    id: str
    source: str  # "banking77" | "synthetic_policy" | other §2 public set
    qtype: QType
    state: str
    instructions: str
    answer: Any  # choice: label str; score: int level; noul: bool
    criteria: Any = None  # choice: dict[str, desc]; score: list[str]; noul: optional dict
    meta: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "id": self.id,
                "source": self.source,
                "qtype": self.qtype,
                "state": self.state,
                "instructions": self.instructions,
                "answer": self.answer,
                "criteria": self.criteria,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_hashes(examples: Sequence[DecisionExample]) -> dict[str, str]:
    """Stable id → sha256 content hash map (Kev/JevBench freeze discipline)."""
    out: dict[str, str] = {}
    for ex in examples:
        if ex.id in out:
            raise ValueError(f"duplicate example id: {ex.id}")
        out[ex.id] = ex.content_hash()
    return out


def split_train_heldout(
    examples: Sequence[DecisionExample],
    heldout_frac: float = 0.2,
    seed: int = 13,
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Deterministic 80/20 by id; disjoint; heldout used only for temp fit."""
    if not (0.0 < heldout_frac < 1.0):
        raise ValueError("heldout_frac must be in (0, 1)")
    ids = sorted({ex.id for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_held = max(1, int(round(len(ids) * heldout_frac)))
    held_ids = set(ids[:n_held])
    heldout = [ex for ex in examples if ex.id in held_ids]
    train = [ex for ex in examples if ex.id not in held_ids]
    train_ids = {ex.id for ex in train}
    held_ids2 = {ex.id for ex in heldout}
    if train_ids & held_ids2:
        raise AssertionError("train/heldout id overlap")
    if len(train_ids) + len(held_ids2) != len(ids):
        raise AssertionError("split dropped or duplicated ids")
    return train, heldout


def banking77_choice_examples(
    texts: Sequence[tuple[str, int]],
    limit: int | None = None,
) -> list[DecisionExample]:
    """(text, label_idx) pairs → 77-way choice decisions (stage-1/2 training)."""
    labels = list(BANKING77_LABELS)
    criteria = {lab: lab.replace("_", " ") for lab in labels}
    examples: list[DecisionExample] = []
    for i, (text, label_idx) in enumerate(texts):
        if limit is not None and i >= limit:
            break
        if not (0 <= label_idx < len(labels)):
            raise ValueError(f"banking77 label {label_idx} out of range")
        examples.append(
            DecisionExample(
                id=f"banking77-{i:05d}",
                source="banking77",
                qtype="choice",
                state=text,
                instructions="Classify the banking intent of the user message.",
                answer=labels[label_idx],
                criteria=criteria,
                meta={"label_idx": label_idx},
            )
        )
    return examples


def synthetic_policy_examples(n: int = 64, seed: int = 7) -> list[DecisionExample]:
    """Programmatic policy pairs (§2: synthetic policies ~10k in full run).

    Here a compact deterministic slice: choice + score + noul, each with
    verifiable gold answers for smoke SFT and ECE fixtures.
    """
    rng = random.Random(seed)
    examples: list[DecisionExample] = []
    depts = {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }
    for i in range(n):
        kind = i % 3
        if kind == 0:
            keyword, gold = rng.choice(
                [
                    ("refund the duplicate", "billing"),
                    ("server is down", "technical"),
                    ("enterprise pricing", "sales"),
                    ("change my display name", "other"),
                ]
            )
            examples.append(
                DecisionExample(
                    id=f"synth-choice-{i:04d}",
                    source="synthetic_policy",
                    qtype="choice",
                    state=f"Customer write-up: please {keyword} as soon as possible.",
                    instructions="Which department should handle this request?",
                    answer=gold,
                    criteria=depts,
                )
            )
        elif kind == 1:
            level = rng.randint(0, 4)
            examples.append(
                DecisionExample(
                    id=f"synth-score-{i:04d}",
                    source="synthetic_policy",
                    qtype="score",
                    state=f"Policy case {i}: severity marker {level} of 5 in the log.",
                    instructions="How urgent is this request?",
                    answer=level,
                    criteria=["not urgent", "low", "medium", "high", "critical deadline"],
                )
            )
        else:
            yes = rng.random() < 0.5
            examples.append(
                DecisionExample(
                    id=f"synth-noul-{i:04d}",
                    source="synthetic_policy",
                    qtype="noul",
                    state=(
                        "Please reverse the charge for invoice 4411."
                        if yes
                        else "Thanks for the update, no action needed."
                    ),
                    instructions="Does the user explicitly request a refund?",
                    answer=yes,
                    criteria={"false": "no explicit refund ask", "true": "explicit refund ask"},
                )
            )
    return examples


def load_banking77_cache(path: str | Path) -> list[DecisionExample]:
    """Load cached Banking77 JSONL: lines {text, label} or {text, label_idx}.

    Optional — full data lands via HF `PolyAI/banking77` (aria2c/hf) on the
    training box; gates use synthetic + label-name checks without the corpus.
    """
    path = Path(path)
    pairs: list[tuple[str, int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        idx = row.get("label_idx", row.get("label"))
        if isinstance(idx, str):
            idx = BANKING77_LABELS.index(idx)
        pairs.append((row["text"], int(idx)))
    return banking77_choice_examples(pairs)


def build_phase_a_mix(
    n_synth: int = 96,
    banking_cache: str | Path | None = None,
    banking_limit: int | None = None,
    seed: int = 13,
) -> list[DecisionExample]:
    """§2 mix available locally: synthetic policies (+ Banking77 if cached)."""
    examples = synthetic_policy_examples(n=n_synth, seed=seed)
    if banking_cache is not None and Path(banking_cache).exists():
        examples.extend(load_banking77_cache(banking_cache)[:banking_limit])
    return examples


def to_laya_question(ex: DecisionExample) -> dict[str, Any]:
    q: dict[str, Any] = {"type": ex.qtype, "instructions": ex.instructions}
    if ex.criteria is not None:
        q["criteria"] = ex.criteria
    return q


def to_laya_record(ex: DecisionExample) -> dict[str, Any]:
    return {
        "id": ex.id,
        "state": ex.state,
        "question": to_laya_question(ex),
        "answer": ex.answer,
        "qtype": ex.qtype,
        "source": ex.source,
    }


def write_freeze_manifest(
    path: str | Path, examples: Sequence[DecisionExample]
) -> Path:
    """Write frozen id→hash manifest (publish with training runs)."""
    path = Path(path)
    manifest = {
        "n": len(examples),
        "hashes": freeze_hashes(examples),
        "sources": sorted({ex.source for ex in examples}),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
