"""G5: server /v1/systemone schema (in-process TestClient, no network port needed)."""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Force small/local path: still full Laya checkpoint.
    os.environ.setdefault("NIRNAY_MODEL", "convaiinnovations/laya")
    os.environ.setdefault("NIRNAY_DEVICE", "cpu")

    from fastapi.testclient import TestClient

    from nirnay.server import app

    with TestClient(app) as client:
        h = client.get("/healthz")
        if h.status_code != 200 or not h.json().get("ok"):
            print(f"SERVER_SCHEMA_FAIL health={h.status_code} {h.text}", file=sys.stderr)
            return 1

        body = {
            "state": "We were billed twice. Please refund the duplicate today.",
            "model": "nirnay-1",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {
                        "billing": "invoices, payments, refunds",
                        "technical": "bugs, outages",
                        "other": "everything else",
                    },
                }
            },
        }
        r = client.post("/v1/systemone", json=body)
        if r.status_code != 200:
            print(f"SERVER_SCHEMA_FAIL status={r.status_code} {r.text[:500]}", file=sys.stderr)
            return 1
        data = r.json()
        ans = (data.get("answers") or {}).get("decision")
        if not isinstance(ans, dict) or ans.get("type") != "choice":
            print(f"SERVER_SCHEMA_FAIL answers={data}", file=sys.stderr)
            return 1
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != {"billing", "technical", "other"}:
            print(f"SERVER_SCHEMA_FAIL probs={probs}", file=sys.stderr)
            return 1
        vals = [float(v) for v in probs.values()]
        if any(v < 0 or v > 1 for v in vals) or abs(sum(vals) - 1.0) > 0.01:
            print(f"SERVER_SCHEMA_FAIL sum={sum(vals)} vals={vals}", file=sys.stderr)
            return 1
        if ans.get("choice") not in probs:
            print(f"SERVER_SCHEMA_FAIL choice={ans.get('choice')}", file=sys.stderr)
            return 1

        # noul + score shapes
        body2 = {
            "state": "The account shows a duplicate charge on invoice 4411.",
            "questions": {
                "decision": {
                    "type": "noul",
                    "instructions": "Is a refund requested?",
                }
            },
        }
        r2 = client.post("/v1/systemone", json=body2)
        a2 = ((r2.json() or {}).get("answers") or {}).get("decision") or {}
        if r2.status_code != 200 or a2.get("type") != "noul":
            print(f"SERVER_SCHEMA_FAIL noul={r2.status_code} {a2}", file=sys.stderr)
            return 1
        p = float(a2.get("noul", -1))
        if not (0.0 <= p <= 1.0):
            print(f"SERVER_SCHEMA_FAIL noul_range={p}", file=sys.stderr)
            return 1

    print("SERVER_SCHEMA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
