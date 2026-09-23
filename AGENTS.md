# AGENTS.md

## Repo status

- Scaffolded for plan §5 Days 1–14 + Days 15–35 entry: Python package `src/nirnay/**` (incl. concepts/deepsup/rlcd/coarse2fine), gate scripts `scripts/check_*.py` (G1–G12), `GATES.md` (**12/12 met** with automatic evidence), `pyproject.toml` + `uv.lock`. Git `main` → private **https://github.com/eulogik/nirnay** (root commit 1047a72; push with `git push origin main`). No lint/typecheck/CI yet — do not invent them.
- `MEMORY.md` holds session state, verified facts, and open risks. Read it before starting work.

## Source of truth

- `NIRNAY-Breakthrough-Plan.md` defines the product: frozen v1 architecture (§1), training recipe (§2), pre-registered benchmarks (§3), 90-day build plan (§5), kill gates (§6).
- Paper/data cutoff: sources through **Sept 19, 2026** (Cheap Verifiers arXiv:2609.01345 v2 verified Sept 23); plan verified/amended **Sept 23, 2026** (amendment log at file end).
- If the plan conflicts with a freshly verified external source (arXiv, HuggingFace, JevBench): re-verify, fix the plan, append to its amendment log. Do not silently diverge.

## Non-negotiable (plan §0, §6)

- **License:** Apache-2.0. Fork only Apache-2.0 sources (Laya, pico-type).
- **Jev is closed:** compare only to published third-party numbers; never distill from Jev API outputs; never publish a Jev column as our own measurement.
- **Metrics:** always report raw **and** fitted ECE; keep zero-shot and fine-tuned results in separate columns.
- **Delivered error (2026-09-23):** never count a metric computed through our own act/escalate/abstain head as delivered-error evidence; delivered-error rates need an external ground-truth sample audit. Flywheel labels from external outcomes only — never verifier-rejected-tail self-training (plan §3/§6/§7, arXiv:2609.01345).
- **v1 scope frozen (plan §1):** NoPE head-masking only; GLA+RoPE split is v1.1 (120M distill target). Params ~450M = 421M Laya fork (LoRA) + ~30M additions. Do not add components absent from §1 without updating the plan first.
- **Benchmarks (§3) are pre-registered:** no cherry-picking; publish pass/fail even on miss.

## Verification discipline

- Citations and benchmark numbers in the plan have been wrong and fixed; see the Sept 22 amendment log. Fetch/spot-check arXiv IDs and leaderboard rows before relying on them.
- Gate runner: unlazy `gate-check.mjs`. Approvals live under `~/.unlazy/approved` (outside repo). EXPECT must be a literal output prefix — comparison operators do not match.
- Model weights: Laya 421M in HF cache (`HF_HOME=/Volumes/KIOXIA 1TB/huggingface_cache`, blob sha256 `891102d372688fc2…`). Load offline with `HF_HUB_OFFLINE=1`. Download method of record: `aria2c -x 8 -s 8` (flaky network; parallel curl/hf hub stalled).

## Commands

Canonical order (from repo root, `uv` managed venv):

```sh
uv sync
uv run python -c "import nirnay; import nirnay.model; import nirnay.server; print('nirnay import ok')"   # G1
uv run python scripts/check_byte_path.py    # G3
uv run python scripts/check_nope_mask.py    # G4
uv run python scripts/check_temps_refit.py  # G6
uv run python scripts/check_forward.py      # G2 (needs model cache + HF_HUB_OFFLINE=1)
uv run python scripts/check_server_schema.py # G5
uv run python scripts/check_jevbench_public.py # G7 (NIRNAY_JEVBENCH_LIMIT=60 default)
uv run python scripts/check_concepts.py      # G8
uv run python scripts/check_deepsup.py       # G9
uv run python scripts/check_rlcd.py          # G10
uv run python scripts/check_coarse_to_fine.py # G11
HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_regression.py # G12 (runs G1–G7)
```

Gate ledger:

```sh
export HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1
node /Users/eulogikdeveloper/.agents/skills/unlazy/scripts/gate-check.mjs --status GATES.md
node /Users/eulogikdeveloper/.agents/skills/unlazy/scripts/gate-check.mjs --approve GATES.md --timeout 600
node /Users/eulogikdeveloper/.agents/skills/unlazy/scripts/gate-check.mjs GATES.md --timeout 600
```

No lint/typecheck command yet (nothing configured). Tests: no suite; the `scripts/check_*.py` files are the verification suite.
