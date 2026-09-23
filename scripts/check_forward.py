"""G1 helper is inline; G2: load model, one multi-question forward, valid dists."""

from __future__ import annotations

import sys


def main() -> int:
    from nirnay import load

    agent = load(
        "convaiinnovations/laya",
        device="cpu",  # M4 local; T4 path is Kaggle
        enable_byte_path=True,
        nope_fraction=1.0 / 3.0,
    )
    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge",
        "body": "We were billed twice for March. Please refund the duplicate today.",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical deadline"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    }
    out = agent.system_one(state, questions)
    answers = out.get("answers") or {}

    def check_choice(ans: dict, keys: list[str]) -> bool:
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            return False
        if set(probs.keys()) != set(keys):
            return False
        vals = [float(v) for v in probs.values()]
        if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            return False
        return abs(sum(vals) - 1.0) <= 0.01

    ok = True
    d = answers.get("department") or {}
    ok = ok and d.get("type") == "choice" and check_choice(
        d, ["billing", "technical", "sales", "other"]
    )
    s = answers.get("urgency") or {}
    ok = ok and s.get("type") == "score" and check_choice(s, ["0", "1", "2"])
    n = answers.get("refund") or {}
    noul = n.get("noul")
    ok = ok and n.get("type") == "noul" and noul is not None and 0.0 <= float(noul) <= 1.0

    nope = agent.nope_report
    ok = ok and bool(nope.get("params_unchanged"))

    if ok:
        print(
            f"FORWARD_OK distributions=valid nope={nope.get('method')} "
            f"masked={nope.get('masked')} heads={nope.get('num_heads')}"
        )
        return 0
    print(f"FORWARD_FAIL answers={answers} nope={nope}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
