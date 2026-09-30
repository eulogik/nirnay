# A 450M open model that beats Jev on Banking77 (with receipts)

NIRNAY phase_b scores **0.8792** on the full Banking77 test set (3,080 cases). Jev 1.13.0 scores 0.803 on the same cases (jevbench.xyz recorded run). Ours runs local in ~200ms, no API key, Apache-2.0.

The caveat up front, because it matters: we fine-tuned on the train split, Jev answered zero-shot. Same test cases, different preparation. That is the whole Laya thesis (a model you fine-tune on your data beats the API), and this is the receipt.

## What it is

450M params: a Laya 421M fork plus ~30M of additions (concept bottleneck, deep supervision, coarse-to-fine pointer, byte path). One forward pass: state plus typed question in, calibrated probabilities out. Choice, score, and yes/no. No text generation.

Fitted ECE 0.045 on test. Raw result JSONs ship in the repo under `eval/`. Reproduce with one script.

## Two ways training died, and the fixes

**Death 1: scale runaway.** The concept encoder output grew 150x during training (healthy RMS 0.09, dead 13.5) while the codebook sat still. The VQ loss spiked 300x and the run died at 1/77 accuracy. A directional test showed the task gradient itself climbs the scale while the codebook chases it. Fix: affine-free LayerNorm on `z` before quantize. Scale stops being a degree of freedom. Zero new params, old checkpoints still load.

**Death 2: usage collapse.** No loss spike at all. All tokens silently fell into one code per chunk, quantized states went constant, accuracy flatlined at 1/77. Fix: a load-balancing aux loss (hard assignment fractions times soft probs). It is linear in the soft term, so gradients flow even at collapse. An entropy version of the same idea failed A/B testing (too soft early, saturates late), so we replaced it.

**Safety net.** Heldout probes every 250 steps, best-checkpoint retention, and an abort that fires when accuracy halves. It caught four real collapses during development, including one that killed a 3,500-step run overnight and kept the best checkpoint automatically.

## Numbers

| Model | Banking77 test (n=3080) | Setup |
|---|---|---|
| NIRNAY phase_b | 0.8792, Brier 0.208, ECE fitted 0.045 | fine-tuned |
| NIRNAY phase_a | 0.8656 | fine-tuned |
| Jev 1.13.0 | 0.803, same 3,080 cases | zero-shot API |
| Untrained | 0.143 | fresh weights |

Speed, batch-1, measured: 209 ms on M4 GPU, 361 ms on CPU. 19/19 gate checks green. Trained on one Mac in about 4 hours.

## What it is bad at

JevBench public: 0.55 overall (easy 0.88, hard 0.40). Long policy docs, probability, and temporal reasoning beyond 512 tokens are the gap. We publish this because cherry-picking is worse than a miss. Training data for that gap (10k programmatic reasoning items) is already in the repo.

## Try it

Model: `eulogik/nirnay-450m` on Hugging Face. Code and eval harness: `github.com/eulogik/nirnay`. One-command install, checkpoint loads locally, no GPU needed (CPU works, just slower).

Built by [Eulogik](https://eulogik.com). Same house rules as pico-type and NanoForecast: Apache-2.0, measured numbers only, runs on your hardware.
