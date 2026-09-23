"""G12: full G1–G7 regression after Days 15–35 modules land.

Runs the seven Days 1–14 check scripts sequentially in this interpreter.
Requires model-cache env when heavy gates run (set by the G12 CHECK line):
  HF_HOME, HF_HUB_OFFLINE=1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fast structural first, model-loading last (matches AGENTS.md order).
CHECKS: list[tuple[str, list[str], str]] = [
    (
        "G1",
        [sys.executable, "-c", "import nirnay; import nirnay.model; import nirnay.server; print('nirnay import ok')"],
        "nirnay import ok",
    ),
    ("G3", [sys.executable, str(ROOT / "scripts" / "check_byte_path.py")], "BYTE_PATH_OK"),
    ("G4", [sys.executable, str(ROOT / "scripts" / "check_nope_mask.py")], "NOPE_OK"),
    ("G6", [sys.executable, str(ROOT / "scripts" / "check_temps_refit.py")], "TEMPS_OK"),
    ("G2", [sys.executable, str(ROOT / "scripts" / "check_forward.py")], "FORWARD_OK"),
    ("G5", [sys.executable, str(ROOT / "scripts" / "check_server_schema.py")], "SERVER_SCHEMA_OK"),
    ("G7", [sys.executable, str(ROOT / "scripts" / "check_jevbench_public.py")], "JEVbench_PUBLIC_OK"),
]


def main() -> int:
    # Defensive defaults when invoked outside the approved CHECK line.
    os.environ.setdefault("HF_HOME", "/Volumes/KIOXIA 1TB/huggingface_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NIRNAY_DEVICE", "cpu")

    failed: list[str] = []
    for name, cmd, expect in CHECKS:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=420,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            print(f"REGRESSION_FAIL {name} timeout", file=sys.stderr)
            failed.append(name)
            continue
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or expect not in (proc.stdout or ""):
            tail = combined.strip().splitlines()[-3:] if combined.strip() else []
            print(
                f"REGRESSION_FAIL {name} exit={proc.returncode} "
                f"expect={expect!r} tail={tail}",
                file=sys.stderr,
            )
            failed.append(name)

    if failed:
        print(f"REGRESSION_FAIL failed={failed}", file=sys.stderr)
        return 1

    print(f"REGRESSION_OK checks={len(CHECKS)} ids={','.join(c[0] for c in CHECKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
