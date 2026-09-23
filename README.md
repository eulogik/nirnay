# NIRNAY

**निर्णय — decision.**  
A System 1.5 model for calibrated decisions: *Jev speed by default, thinks longer only when uncertain, remembers across calls, reads bytes — not tokens. Open weights. Apache-2.0.*

> One parallel pass over `state + questions` → probability distributions you can ship.  
> No text generation. No black-box API. No cherry-picked benchmarks.

---

## Why

Production systems do not need another chatbot. They need answers like:

- *Which team owns this incident?*
- *Is this charge a duplicate?*
- *How urgent is this request — 1 to 5?*

Today you choose between a fast, closed, poorly calibrated API and a slow, expensive LLM that still overconfidence-wrongs its way through hard slices. **NIRNAY is the third path:** a small, open decision head that is fast enough for inline routing, calibrated enough to trust at the threshold, and honest enough to abstain when it does not know.

| | Jev (closed) | Laya (Apache-2.0 fork base) | **NIRNAY** |
|---|---|---|---|
| Interface | distributions only | distributions | distributions + abstain |
| Weights | ✗ closed | ✓ open | ✓ open |
| Byte-level / all scripts | ✗ text | ✗ BPE shred (Khmer 0.00@0.95) | ✓ byte path + router |
| Calibration reported | miscalibrated (unconfirmed ECE) | raw **and** fitted | raw **and** fitted, always |
| Memory / multi-turn | stateless | stateless | session memory (in plan) |
| Independent eval | JevBench 74.4 | JevBench 54.4 | *pre-registered targets, publish miss or hit* |

Benchmarks are **pre-registered** in the plan (§3). We publish pass/fail even on a miss.

---

## What is in the box (v1 scaffold)

```
bytes → coding-rate patches → Laya 421M fork (NoPE head-masking)
      → concept bottleneck (product-VQ + MoME slots)
      → hypercube sparse wiring + SGDR block router
      → parallel heads: choice | score | noul (+ relational, abstain)
      → per-(type, count) temperatures → calibrated distribution
      → halting gate (loop ≤4× only when uncertain)
```

| Module | Plan ref | Role |
|---|---|---|
| `nirnay.bytes` | §1.1 | ByteEmbed → conv stack → encoder patches (no tokenizer) |
| `nirnay.nope` | §1.2 | NoPE mask on 1/3 global heads — zero new params |
| `nirnay.concepts` | §1.3 | Product-VQ 128-dim / chunk-4 / 32-codes, MoME M=4, NCP loss |
| `nirnay.hypercube` · `nirnay.sgdr` | §1.4–5 | Sparse wiring + training-free long-range blocks |
| `nirnay.data` · `lora` · `losses` · `train` | §2 | Hash-frozen mix, LoRA, plan loss (0.3/0.2/λ), Phase A SFT |
| `nirnay.deepsup` | §1 | Aux CE at encoder layers 4 / 8 / 12 (`0.2 · L_deep`) |
| `nirnay.rlcd` | §1.8, §2 | RLCD++: bounded Brier reward, calCE, group-mean baseline |
| `nirnay.coarse2fine` | §1.3 | Stage-1 top-20 retrieve → stage-2 pointer (77-way) |
| `nirnay.temps` | §2 | Per-bucket temperature fit → `temperature_by_options.json` |
| `nirnay.agent` · `nirnay.server` | §4 | Laya-fork runtime + Jev-compatible `POST /v1/systemone` |

Full architecture, training recipe, kill gates, and 90-day plan:  
**[`NIRNAY-Breakthrough-Plan.md`](./NIRNAY-Breakthrough-Plan.md)**

---

## Quickstart

```bash
git clone https://github.com/eulogik/nirnay.git
cd nirnay
uv sync --extra dev

# package + model sanity
uv run python -c "import nirnay; print(nirnay.__version__)"

# Jev-compatible server
uv run nirnay-server    # POST http://localhost:8000/v1/systemone
```

Drop-in for any typesafe-style client:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "nirnay-1",
  "state": "We were billed twice for March. Please refund the duplicate.",
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "Which department handles this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "other": "everything else"
      }
    }
  }
}'
```

Answer shape: `answers.decision.{choice|probabilities}` — valid distributions, every question, one pass.

---

## Verify

Every claim below is machine-checked. The ledger lives in [`GATES.md`](./GATES.md) (**15 / 15 met** with automatic evidence).

```bash
# structural gates (no model download)
uv run python scripts/check_byte_path.py       # G3  byte → patches → encoder
uv run python scripts/check_nope_mask.py       # G4  NoPE active, params unchanged
uv run python scripts/check_temps_refit.py     # G6  temps ∈ [0.5, 5.0]
uv run python scripts/check_concepts.py        # G8  VQ codes, gate, NCP, ≤17M
uv run python scripts/check_deepsup.py         # G9  layers 4/8/12 only, 0.2×CE
uv run python scripts/check_rlcd.py            # G10 Brier ∈ [0,1], calCE, baseline
uv run python scripts/check_coarse_to_fine.py  # G11 gold in top-20, valid pointer

# model gates (need Laya weights in HF cache; offline via HF_HUB_OFFLINE=1)
uv run python scripts/check_forward.py         # G2  one pass, valid dists
uv run python scripts/check_server_schema.py   # G5  /v1/systemone schema
uv run python scripts/check_jevbench_public.py # G7  public suite accuracy
uv run python scripts/check_regression.py      # G12 runs G1–G7 end-to-end
uv run python scripts/check_data_pipeline.py  # G13 77 intents, hash freeze, 80/20
uv run python scripts/check_loss_assembly.py  # G14 exact 0.3 / 0.2 / λ coeffs
HF_HOME=… HF_HUB_OFFLINE=1 uv run python scripts/check_phase_a_smoke.py  # G15 Phase A SFT
```

Third-party references (Laya, JevBench, pico-type) are **not** vendored:

```bash
git clone https://github.com/convaiinnovations/laya        third_party/laya
git clone https://github.com/fstandhartinger/jevbench      third_party/jevbench
git clone https://github.com/eulogik/pico-type             third_party/pico-type
```

Gate runner (optional, same ledger):

```bash
node ~/.agents/skills/unlazy/scripts/gate-check.mjs --status GATES.md
```

---

## Status

| Phase (plan §5) | State |
|---|---|
| Days 1–14 — Fork + parity | **Done** — G1–G7 met (forward, byte path, NoPE, server, temps, JevBench public) |
| Days 15–35 — Calibrate + MoME | **Done through training stack** — concept/MoME, deep supervision, RLCD++, coarse-to-fine (G8–G11), full regression (G12), §2 data + plan loss + Phase A SFT smoke (G13–G15) → rented GPU for ECE/Banking77 gates |
| Days 36–60 — Think + remember | Planned (halting gate, session memory) |
| Days 61–90 — Harden + launch | Planned (32k sparse, ONNX, Space, JevBench submission) |

Honest open risk: public JevBench accuracy on the local CPU path is **0.600** (n=60 cap); the pre-registered milestone is ≥70 on the full frozen set — measured on held-out data before any launch claim.

**Never (contract):** distill from Jev · publish a Jev column as our own measurement · report fitted-only ECE · count an in-loop head metric as delivered-error · train on verifier-rejected-tail self-labels.

---

## Repository map

```
NIRNAY-Breakthrough-Plan.md   source of truth (architecture, §3 benchmarks, kill gates)
GATES.md                      machine-checked completion ledger (G1–G15)
src/nirnay/                   package: bytes, nope, concepts, deepsup, rlcd, …
scripts/check_*.py            one script per gate — the verification suite
AGENTS.md · MEMORY.md         agent/session operating notes
```

---

## License

**Apache-2.0** — fork chain is legal on purpose: [Laya](https://github.com/convaiinnovations/laya) and [pico-type](https://github.com/eulogik/pico-type) are both Apache-2.0. Jev is closed: we only ever compare against **published third-party numbers**, never distill its outputs.
