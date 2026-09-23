# MEMORY.md

Session state for the NIRNAY repo. Update when decisions, verified facts, or project state change.

## State (as of 2026-09-23)

- Git: branch `main` tracks **private** `https://github.com/eulogik/nirnay` (root `1047a72`; training stack pushed as `bfdedde`; Banking77+PhaseA/B+eval pending this commit). Identity: Team Eulogik &lt;developers@eulogik.com&gt;.
- Code: `src/nirnay/**` incl. `data,lora,losses,train,eval` + prior modules; `scripts/check_*.py` (G1–G19); Banking77 CC-BY-4.0 CSVs under `data/banking77/` (tracked). No lint/typecheck/CI.
- **GATES.md: 19/19 MET** (G1–G7 Days 1–14; G8–G12 entry modules; G13–G15 training stack; G16–G19 runnable path). Modules: `concepts,deepsup,rlcd,coarse2fine` + `data,lora,losses,train,eval`. Concept params 4,739,012 ≤17M. G15: 112 LoRA wraps, frozen encoder, hooks {4,8,12}, 3 steps 2.52→2.43. G16: Banking77 n=13083 labels=77 train=10466 heldout=2617. G17: Phase A CLI writes freeze manifest + history + 2GB ckpt, encoder_frozen=true, trainable=36.5M. G18: Phase B RLCD 3 steps loss 1.55→1.34 reward∈[0.42,0.65] adv_mean~0. G19: eval always emits raw+fitted ECE together.
- Plan amended **2026-09-23** (append-only, now 180 lines): Cheap Verifiers arXiv:2609.01345 clauses in §3 Rules (no act/escalate/abstain-head metric as delivered-error evidence), §6 Never (+external ground truth, +no rejected-tail self-labels), §7 flywheel (external labels only), Sources + amendment log. Impact: no change to G1–G7 or Day gates; constrains future operational delivered-error reporting and flywheel label pipeline.
- Model: Laya 421M (`421293827` params) in HF cache; sha256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`; offline load ~26s CPU. Warning: checkpoint ships temps outside [0.5,5] (choice:11+ clamped from 0.1006).
- Gate results of record: G2 `FORWARD_OK distributions=valid nope=patch… masked=True heads=16`; G5 `SERVER_SCHEMA_OK`; G7 `JEVbench_PUBLIC_OK schema_validity=1.000 accuracy=0.600 n=60 limit=60`.
- Public JevBench suite is only 231 items; full ≥70 claim needs held-out data (G7 uses limit 60).
- Next: rented-GPU Phase A/B toward §3 (fitted ECE ≤0.08, Banking77 ≥0.65). Local path ready: Banking77 CSVs at `data/banking77/`, CLI `python -m nirnay.train --phase a|b`. Remaining §2 data not yet local: AG News, SST-2, BoolQ, MNLI, ToxicChat, Enron, DAIR Emotion, MASSIVE slice, typed-decisions train split. Push when asked.

## Decisions (do not reverse casually)

- Standalone product: not VasoolAI, not AgentTrust; new repo, new weights, new benchmark.
- License Apache-2.0; fork sources: Laya + pico-type (both Apache-2.0).
- v1 architecture frozen (plan §1): bytes → coding-rate patches → Laya fork w/ NoPE masking → concept/MoME → hypercube + SGDR → parallel heads → per-bucket temps → halting ≤4x. GLA split deferred to v1.1.
- Param budget ~450M; the 120M encoder is a v1.1 distill/CPU-tier target only.
- Kill gates K1–K4 and §3 benchmark targets are pre-registered — treat as contracts.

## Verified facts (spot-check before reuse; may go stale)

- JevBench v1.3.0 (534 frozen decisions): Jev 74.4 > SemIf 73.1 > Laya 54.4.
- Laya typed-decisions: 0.766 acc, Brier 0.062, raw ECE 0.466 → fitted 0.081, Banking77 0.425, 32.8ms T4.
- SGDR = arXiv:2609.22884 (training-free block routing; 5.03× vs FlashAttn @128K, <3.4ms overhead).
- Kev = github.com/jaredpalmer/kev, Qwen3.5-gen 0.8B/4B/9B (Sept 21); unanswerable high-conf Kev-9B 0.05.
- Hypercube = arXiv:2609.18145; DAT = arXiv:2609.20530; MoRE = arXiv:2609.18176.
- Cheap Verifiers = arXiv:2609.01345 (v2, verified 2026-09-23): β 0.12→0.55 (0.5B→32B student), frontier verifier β≈0.05 escalates 46% hard-MATH vs 39% true error, corrective FT on verifier-rejected tail collapses student, in-loop dashboard 3% vs true delivered error 32%, conservation ε_∞≲q₀β₀.

## Open risks

- Days 1–14 full JevBench ≥70 gate (plan) vs measured **0.600 on public n=60** — public subset may not represent the 534-item frozen set; need held-out evaluation before claiming the milestone number.
- K3 (Banking77 <0.65) vs §3 target ≥0.80: intentional cushion, not a contradiction.
- ~35ms GPU latency, ≤100ms CPU-ONNX, and <$300 cost are targets — unmeasured.
- Jev calibration figures (ECE 0.107 etc.) still marked unconfirmed in plan §0.
- Shipped Laya temps have out-of-range bucket (`choice:11+=0.1006`); our G6 refit writes temps in [0.5,5] but agent currently relies on laya clamp warning — wire `temperature_by_options.json` into serving path before publishing calibrated numbers.
- Network to HF is flaky (parallel Python/curl stalled); `aria2c -x 8` is the reliable download path. Do not assume `hf download` works unattended.
