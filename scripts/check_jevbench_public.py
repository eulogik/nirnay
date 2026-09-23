"""G7: JevBench public suite against local /v1/systemone.

Runs datasets/public/{original,easy,hard}.jsonl through the typesafe adapter
wire format against an in-process TestClient (or NIRNAY_ENDPOINT if set).
Reports accuracy + schema validity on public items only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "third_party" / "jevbench" / "datasets" / "public"


def load_tasks() -> list[dict]:
    tasks = []
    for name in ("original.jsonl", "easy.jsonl", "hard.jsonl"):
        p = PUB / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_question(task: dict) -> dict:
    q = {"type": task["question"]["type"], "instructions": task["question"]["instructions"]}
    if task["question"].get("criteria") is not None:
        q["criteria"] = task["question"]["criteria"]
    return q


def extract_probs(task: dict, ans: dict) -> dict | None:
    qtype = task["question"]["type"]
    labels = task["labels"]
    try:
        if qtype == "noul":
            p = float(ans["noul"])
            if not (0.0 <= p <= 1.0):
                return None
            return {"no": 1.0 - p, "yes": p}
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            return None
        # Normalize keys to label strings
        if qtype == "score":
            # keys are level indices as strings; labels are ordered levels
            out = {}
            for i, lab in enumerate(labels):
                out[lab] = float(probs.get(str(i), probs.get(i, 0.0)))
        else:
            out = {str(k): float(v) for k, v in probs.items()}
            if set(out) != set(map(str, labels)):
                # try direct
                if set(out) != set(labels):
                    return None
        s = sum(out.values())
        if s <= 0:
            return None
        if abs(s - 1.0) > 0.02:
            return None
        return {k: v / s for k, v in out.items()}
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    os.environ.setdefault("NIRNAY_DEVICE", "cpu")
    endpoint = os.environ.get("NIRNAY_ENDPOINT")

    if endpoint:
        import urllib.request

        def ask(state, question):
            body = json.dumps(
                {"state": state, "model": "nirnay-1", "questions": {"decision": question}}
            ).encode()
            req = urllib.request.Request(
                endpoint.rstrip("/") + "/v1/systemone",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
    else:
        from fastapi.testclient import TestClient

        from nirnay.server import app

        client_holder = {}

        def ask(state, question):
            if "c" not in client_holder:
                client_holder["c"] = TestClient(app)
                client_holder["c"].__enter__()
            r = client_holder["c"].post(
                "/v1/systemone",
                json={"state": state, "model": "nirnay-1", "questions": {"decision": question}},
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()

    tasks = load_tasks()
    if not tasks:
        print("JEVbench_PUBLIC_FAIL no_tasks", file=sys.stderr)
        return 1

    n = 0
    correct = 0
    schema_ok = 0
    errors = 0
    # Cap for local CPU runtime; full suite on Kaggle/H100.
    limit = int(os.environ.get("NIRNAY_JEVBENCH_LIMIT", "60"))
    for task in tasks[:limit]:
        n += 1
        q = build_question(task)
        try:
            out = ask(task["state"], q)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"task {task.get('id')} error: {e}", file=sys.stderr)
            continue
        ans = (out.get("answers") or {}).get("decision")
        if not isinstance(ans, dict):
            errors += 1
            continue
        probs = extract_probs(task, ans)
        if probs is None:
            errors += 1
            continue
        schema_ok += 1
        # accuracy: argmax vs expected (labels ordered as in task)
        labels = [str(x) for x in task["labels"]]
        best = max(labels, key=lambda lab: probs.get(lab, probs.get(str(lab), 0.0)))
        # map expected
        exp = str(task["expected"])
        # For score, expected may be level value — compare via labels list index
        if task["question"]["type"] == "score":
            # expected is often the label string; fall back to argmax label equality
            pass
        if best == exp or exp in probs and best == exp:
            correct += 1
        else:
            # also accept if expected has highest prob under alternate key
            if exp in probs and probs[exp] == max(probs.values()):
                correct += 1

    if n == 0:
        print("JEVbench_PUBLIC_FAIL zero", file=sys.stderr)
        return 1
    schema_validity = schema_ok / n
    acc = correct / n
    if schema_validity > 0.99 and errors == 0:
        print(
            f"JEVbench_PUBLIC_OK schema_validity={schema_validity:.3f} "
            f"accuracy={acc:.3f} n={n} limit={limit}"
        )
        return 0
    print(
        f"JEVbench_PUBLIC_FAIL schema_validity={schema_validity:.3f} "
        f"accuracy={acc:.3f} n={n} errors={errors}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
