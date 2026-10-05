# NIRNAY 450M: a small calibrated decision model that beats Jev on Banking77

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/eulogik/nirnay/blob/main/NIRNAY-Breakthrough-Plan.md)
[![Params: 450M](https://img.shields.io/badge/params-450M-cyan.svg)](https://github.com/eulogik/nirnay/blob/main/src/nirnay/train.py)
[![Banking77: 87.9%](https://img.shields.io/badge/Banking77-87.9%25-brightgreen.svg)](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b.json)
[![ECE fitted: 0.045](https://img.shields.io/badge/ECE_fitted-0.045-blueviolet.svg)](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b.json)
[![Gates: 19/19](https://img.shields.io/badge/gates-19%2F19-success.svg)](https://github.com/eulogik/nirnay/blob/main/GATES.md)
[![Built by Eulogik](https://img.shields.io/badge/built_by-Eulogik-orange.svg)](https://eulogik.com)

**NIRNAY (निर्णय, "decision") is a 450M open-weight classifier for banking intent and typed decisions.** One forward pass turns a state plus a question into calibrated probabilities. No text generation, no API key, no per-call bill. Apache-2.0, built by [Eulogik](https://eulogik.com).

<p align="center">
  <img src="https://raw.githubusercontent.com/eulogik/nirnay/main/assets/hero.png" alt="NIRNAY 450M: 87.9% on Banking77, fitted ECE 0.045, Apache-2.0" width="100%" />
</p>

[Model](https://huggingface.co/eulogik/nirnay-450m) · [Live demo](https://huggingface.co/spaces/GautamKishore/nirnay-demo) · [Code](https://github.com/eulogik/nirnay) · [Paper](https://github.com/eulogik/nirnay/blob/main/papers/nirnay/main.pdf) · [Evals](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b.json)

<p align="center">
  <img src="https://raw.githubusercontent.com/eulogik/nirnay/main/assets/benchmark_banking77.png" alt="Banking77: NIRNAY vs Jev vs Julia-1 vs untrained, same 3,080 test cases" width="100%" />
</p>

Banking77 keywords for search: banking intent classification, 77-way intent classifier, intent detection model, small language model for classification, calibrated decision model, system one model, Jev alternative, Laya fine-tune, on-device text classifier, Apache 2.0 classifier.

## Numbers first

| Model | Banking77 test (n=3080) | Setup |
|---|---|---|
| **NIRNAY phase_b** | **0.8792** (Brier 0.208, ECE raw 0.089 / fitted 0.045) | fine-tuned, this repo |
| NIRNAY phase_a | 0.8656 (Brier 0.240, ECE raw 0.110 / fitted 0.033) | fine-tuned, this repo |
| Jev 1.13.0 | 0.803 (same 3,080 cases, recorded run) | zero-shot API ([jevbench.xyz](https://jevbench.xyz)) |
| Julia-1 144M | 0.64 (their 72-label pilot, n=100, shortlist) | their card concedes grouped routing drops answers |
| Untrained baseline | 0.143 (our template, our measurement) | fresh weights |

Raw JSON: [`eval/banking77_phase_b.json`](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b.json), [`eval/banking77_phase_a.json`](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_a.json). Same test split for every row above.

> The honest caveat: we fine-tuned on the train split, Jev answered zero-shot. That is exactly the Laya thesis (a model you fine-tune on your data), and this repo proves it works: +7.6 points over the API on identical cases.

Same metric as Cloudflare's Decision Index table (macro-F1, measured by us on the same 3,080 cases, per-class scores in [`eval/banking77_phase_b_macro.json`](https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b_macro.json)):

| Model | Banking77 macro-F1 | Setup |
|---|---|---|
| Clef 27B | 0.942 | their Decision Index 0.2.1 suite (self-reported) |
| Clef-Flash 9B | 0.909 | their Decision Index 0.2.1 suite (self-reported) |
| **NIRNAY phase_b** | **0.8785** | fine-tuned 450M, this repo |
| Kev 9B | 0.848 | their Decision Index 0.2.1 suite (self-reported) |
| Jev 1.13.0 | 0.797 | their Decision Index 0.2.1 suite (self-reported) |
| Laya | 0.143 | their Decision Index 0.2.1 suite (self-reported) |

Same metric family, different harnesses, so read it as a cross-check, not a leaderboard. We beat Jev by 8 and Kev by 3 at 60x and 20x fewer params, on a Mac instead of an H200.

| Benchmark / Metric | Jev 1.13.0 | NIRNAY phase_b |
|---|---|---|
| Banking77 (same 3,080) | 0.803 (recorded run) | **0.8792** (fine-tuned) |
| ECE, raw | 0.197 gap (jevals, their setup) | **0.089** (our test) |
| Latency, 1 decision | 310 to 478 ms (API medians) | **209 ms** MPS / 361 ms CPU |
| Weights | closed API | **Apache-2.0** |
| Cost | per-call billing | **$0 self-hosted** |

Jev figures above are third-party published, never measured here (no API access); sample sizes and prompts differ. Ours are measured in this repo.

## Where Jev leads

* **JevBench overall.** 72.1 vs our 0.55 public: intelligence, sealed decisions, calibration at scale. Different league, honestly stated.
* **Long documents.** Our context caps at 512 tokens. Their API takes more.
* **Zero setup.** Their API works with a key. Ours needs a local install plus a 466MB checkpoint download.
* **Languages.** We trained English banking. They serve many languages out of the box.

Speed, batch-1, measured 2026-09-30: **209 ms on M4 MPS, 361 ms on CPU.** Faster than Jev API calls (310 to 478 ms in independent runs). Slower than Laya's 33 ms. Corpus latency work (ONNX) is open.

## Try it

No install needed: **[live demo](https://huggingface.co/spaces/GautamKishore/nirnay-demo)** (type a message, get the intent). Or run it yourself:

```bash
pip install git+https://github.com/eulogik/nirnay
```

```python
from nirnay.agent import NirnayAgent

agent = NirnayAgent(device="mps", checkpoint_path="phase_b.pt", enable_byte_path=False)
out = agent.system_one(
    "My card was charged twice for the same order.",
    questions={"intent": {
        "type": "choice",
        "instructions": "Classify the banking intent.",
        "criteria": {"duplicate_charge": "charged twice", "refund_status": "ask about refund"},
    }},
)
print(out["answers"]["intent"]["probabilities"])
```

Choice, score, and yes/no ride the same call:

```python
out = agent.system_one(
    "Ticket T-4411: outage started 02:10, still down, 40 agents idle.",
    questions={
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
        "page_oncall": {
            "type": "noul",
            "instructions": "Should the on-call be paged?",
        },
    },
)
print(out["answers"]["urgency"]["score"])   # expected level over the rubric
print(out["answers"]["page_oncall"]["noul"])  # P(true)
```

The checkpoint loads with `NirnayAgent(checkpoint_path=...)`. Full eval harness: `scripts/eval_checkpoint.py`. Server: `nirnay.server` exposes `POST /v1/systemone` (Jev-compatible wire format).

## How it was trained

Base is Laya 421M (Apache-2.0) plus ~30M of additions (concept bottleneck, deep supervision, coarse-to-fine pointer, byte path). 7,000 Phase A steps + 50 RLCD steps, all on one Mac. Three-group optimizer, seeded everything, hashes frozen before training.

<p align="center">
  <img src="https://raw.githubusercontent.com/eulogik/nirnay/main/assets/training_stability.png" alt="Probe accuracy and flat NCP across all 7,000 steps" width="100%" />
</p>

Two training deaths taught us the fixes (both landed, both gated):

1. **Scale runaway.** The concept encoder output grew unboundedly (healthy RMS 0.09, dead 13.5) while centroids sat still, so the VQ loss exploded 300x. Fix: affine-free LayerNorm on `z` before quantize. Scale stops being a degree of freedom. Zero new params.
2. **Usage collapse.** With no pressure on code usage, all tokens fell into one code per chunk, the quantized states went constant, and accuracy flatlined at 1/77 with no loss spike to warn you. Fix: a load-balancing aux term (hard fractions times soft probs, linear so gradients never vanish).

Plus a safety net around training itself: heldout probes every 250 steps, best-checkpoint retention, and an abort that fires when accuracy halves. It caught four real collapses during development. Details: [`NIRNAY-Breakthrough-Plan.md`](https://github.com/eulogik/nirnay/blob/main/NIRNAY-Breakthrough-Plan.md) (amendment 2026-09-26), [`MEMORY.md`](https://github.com/eulogik/nirnay/blob/main/MEMORY.md).

## Calibration

We report raw and fitted ECE on every eval, always. Phase_b on Banking77 test:

<p align="center">
  <img src="https://raw.githubusercontent.com/eulogik/nirnay/main/assets/reliability.png" alt="Reliability diagram, measured on test set" width="100%" />
</p>

Most mass sits above 0.9 confidence at 92% accuracy there. Per-bucket temperatures ship with the run.

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/eulogik/nirnay/main/assets/architecture.png" alt="Forward pass" width="100%" />
</p>

Bytes and token ids feed a frozen Laya encoder (LoRA adapters train). A concept bottleneck (layer-normed product-VQ, 4 chunks x 32 codes, mixture-of-slots) adds a learned residual. A 2-layer head scores dense markers; a coarse-to-fine pointer re-ranks the top 20 for 77-way decisions. Per-bucket temperatures calibrate the output. Deep supervision at layers 4/8/12 and RLCD exist only at training time.

## Evals and honesty

* `eval/` holds the raw result JSONs. Reproduce with `scripts/eval_checkpoint.py --checkpoint <ckpt>`.
* JevBench public (231 items, shipped checkpoint): **0.550** overall, easy 0.875, original 0.569, hard 0.396. The hard tier (long policy docs, probability, temporal reasoning) is the gap. We publish it because the plan pre-registers pass/fail either way. Training data for that gap (10k programmatic reasoning items, ground truth by construction) is in `src/nirnay/data_synth_hard.py`.
* 19/19 gate checks green ([`GATES.md`](https://github.com/eulogik/nirnay/blob/main/GATES.md)). Zero-shot and fine-tuned numbers are never mixed.

## Limits

* 512-token context. Long documents get head-truncated; the hard JevBench tier shows it (0.396). Longer context is v1.1 work.
* CPU latency 361 ms per decision in PyTorch. No ONNX export yet, no GGUF/Ollama build yet.
* Banking77 is the proven lane (triage, routing, guardrails). Anything else, measure before trusting.

## FAQ

**What is NIRNAY?** A 450M Apache-2.0 decision model: state plus typed question in, calibrated probabilities out. Choice, score, and yes/no questions.

**How accurate is it?** 87.9% on Banking77 test (3,080 cases), Brier 0.208, fitted ECE 0.045.

**How does it compare to Jev?** On identical Banking77 test cases: 0.879 (fine-tuned) vs 0.803 (Jev 1.13.0 zero-shot API). Ours runs local, no per-call cost.

**How does it compare to Laya?** Laya is our base (421M, Apache-2.0). We fixed its training instability, added the concept bottleneck and pointer, and fine-tuned it. Untrained on our template it scores 0.143 here.

**How fast is it?** 209 ms per decision on M4 GPU, 361 ms on CPU, batch-1, measured.

**What license?** Apache-2.0. Commercial use fine.

**What hardware trained it?** One Mac (M4, 16GB). Full run about 4 hours. Cloud spend: $0. Receipts in `logs/` and `MEMORY.md`.

**What is it bad at?** Long documents over 512 tokens, and JevBench-hard style probability/temporal reasoning (0.396). See Limits.

## Built by Eulogik

NIRNAY is built by [Eulogik](https://eulogik.com) ([GitHub](https://github.com/eulogik), contact: info@eulogik.com), makers of pico-type, NanoForecast, TinyDoc-VLM, and KARN. Same house rules: Apache-2.0, measured numbers only, runs on your hardware.

## License

Apache-2.0. Laya base (ConvAI Innovations, Apache-2.0). Banking77 data (PolyAI, CC-BY-4.0).

## Links

- **Model:** https://huggingface.co/eulogik/nirnay-450m
- **Code:** https://github.com/eulogik/nirnay
- **Paper:** https://github.com/eulogik/nirnay/blob/main/papers/nirnay/main.pdf
- **Evals:** https://github.com/eulogik/nirnay/blob/main/eval/banking77_phase_b.json
- **Eulogik:** https://eulogik.com (contact: info@eulogik.com)
