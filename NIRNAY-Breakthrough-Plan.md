# NIRNAY-1 — The System 1.5 Model: Jev/Laya Killer

> **One-liner:** Jev speed by default. Thinks longer only when uncertain. Remembers across calls. Reads bytes, not tokens. Open weights.
>
> **Status:** BREAKTHROUGH PLAN — Sept 22, 2026 (incorporating all papers through Sept 19)
> **Scope:** standalone. Not VasoolAI. Not AgentTrust. New repo, new weights, new benchmark.
> **License target:** Apache-2.0 (fork-legal: Laya + pico-type are both Apache-2.0; Jev is closed — never distill from its API outputs, only compare against published numbers)

---

## 0. What we know (facts, not vibes)

### Jev (TypeSafe AI, Sept 15 2026, closed)

* Interface: `state + {Choice ≤255 | Score 2-10 | Noul P(yes)}` → distribution + confidence, one parallel pass. No text out.
* Claimed: 70–500ms end-to-end, 40–200x faster, 193.6x / 444.6x peak on own 4 workflows, $0.042/M in / free out.
* Measured by third parties: single-Q ~1.7x vs cheap LLM (477ms vs 790ms); 6-Q batched ~100x (0.74s vs 74s). 6/7 defects vs Fable 7/7 at 25x faster / ~580x cheaper. Own accuracy 67.8% vs 74.1% Sol / 73.1% Opus (agreement with Astra+Fable avg, **not ground truth**).
* Training: RLCD, synthetic-only, undisclosed loss/arch. Best teardown hypothesis: causal decoder + prefix-KV + tree mask + pointer head.
* Hard limits: closed weights/API+waitlist, text/JSON only, 255 cap, ~32K/Q + 64K total, stateless, no generation, still miscalibrated off-distribution (ECE 0.107 vs 0.024 floor; Noul underconfident T=0.66, Choice/Score overconfident T=3.29/3.40; unanswerable 44.7% @0.74 — these calibration figures unconfirmed this pass). No Terms-of-Service benchmark ban found; policy still: all Jev numbers are third-party/published, never directly measured by open builders.
* Sources: typesafe.ai launch, TechCrunch 9/18, TrueStandard 9/19, Aman Kumar 16k calls, JevBench, Firecrawl/Vercel docs.

### Laya (ConvAI Innovations, Sept 19–22 2026, Apache-2.0)

* 3 ckpts: `laya` 421M ModernBERT-large 512ctx EN / `laya-multilingual` 322M mmBERT 1024ctx 100+ langs / `laya-typed-decisions` 421M 1024ctx fine-tuned. `pip install laya`, Router (script detect 0.09–0.73ms), Kaggle 2xT4 fine-tune ~4–5h.
* Arch (published, inspectable): bidirectional encoder + 2-layer head + per-option `[MASK]` scorer + type embedding + act/escalate head + temp per (type, option-count).
* Training (published RLCD): policy distribution + Gaussian logit noise → reward = proper scoring (log+spherical+RPS) → REINFORCE + group-mean baseline (GRPO-style) + TD(λ=1.0) multi-turn + soft-CE vs teacher. Human data, zero synthetic.
* Claim that matters: `laya-typed-decisions` **0.766 vs Jev 0.727 (+3.9pp)** on 2,000 decisions, Brier 0.062 vs 0.148 (2.4x), AG News 0.95 vs 0.91, Emotion 0.595 vs 0.48, ECE fitted 0.081 vs 0.246, latency 32.8ms T4 vs 236–276ms (7.8x), batched 7.2ms/q (20x).
* Caveats (author-stated, verified): win is **fine-tuned on train split vs Jev zero-shot** — base zero-shot 0.362 vs 0.318 random / 0.461 majority. Raw ECE 0.466→0.081 only after fitting (ships overconfident). Banking77 77-way **0.425 vs Jev 0.870** (192–256 tok head budget → 3–4 tok/option). Soft-acc trails 0.471 vs 0.580. Independent JevBench v1.3.0 (fstandhartinger/jevbench, 534 frozen decisions): **Jev 74.4 > SemIf 73.1 > Laya 54.4 (#33)** — on shared rubric Jev still leads by ~20 points; Laya's 512-tok budget cuts hard-tier states (hard acc 34.1% vs Jev 74.1%).
* Multilingual finding (critical): EN ckpt on Khmer 0.000@0.952, Armenian 0.05@0.885, Hindi 0.10@0.941 — BPE shreds non-Latin, confidence gives no warning. Router mandatory.
* Sources: huggingface.co/convaiinnovations/laya*, laya.convaiinnovations.com, BENCHMARKS.md, rlresearch.ai audit, JevBench, Kev README/cards (Qwen3.5-gen).

### pico-type (eulogik, US, June 2026 v0.1 → v0.2, Apache-2.0)

* 1.43–1.56M byte-level, no tokenizer, ≤1024 raw bytes → 7 fixed heads (coarse 12 / modality 8 / subtype 24 / code 62 / text 30 / MIME 90 / risk 6) in one pass. `ByteEmbed(256→96) → 3×Conv(k3,5,7) → 2×BiAttn(RoPE) → Pool(mean‖max‖std)→576d → Matryoshka heads (16/64/192/576)`.
* 9MB ONNX FP32, **~18ms CPU (README, verified)** — per-tier 3/5/12ms claims not confirmed this pass. CLI/MCP/Rust/WASM/Gradio. Trained M4 16GB: synth + 8,709 GitHub + 5,000 Wiki. Code 60.3% Heap (+57pp over synth-only), text-lang 98.3% Wiki (+79pp), synth heads 100%, real 20/21 (95.2%).
* Limits: fixed taxonomy only (no arbitrary schema), no RLCD/calibration (softmax max + thr 0.4/0.5), 1024B trunc, 8 code langs <50%, not semantic.
* Sources: github.com/eulogik/pico-type, HF eulogik/pico-type*, arXiv:2608.14658, eulogik.com/lab.

### Converging R&D (why now — all papers through Sept 19, 2026)

| Vector | Paper / artifact | Takeaway for NIRNAY |
|---|---|---|
| Latent concepts beat tokens | NCP-ArchPreview 8.9B/5.73T: 51.3% tokens to match OLMo, +2.45 macro, 17M adapter; LCM sentence-space | Concept bottleneck → break 255 cap, 2x efficiency |
| JEPA latent prediction | V-JEPA 2.1: dense loss (all tokens) + deep supervision (intermediate layers) + multimodal tokenizers, 2B SOTA grasp/nav | Loss at layers 4/8/12 + predict latent, ignore unpredictable detail |
| Test-time memory | Titans/MIRAS (Google Dec 2025): MLP as LTM + surprise gradient + momentum + forgetting, >2M ctx, beats GPT-4 BABILong small | Per-session memory, surprise-gated update — first System One with memory |
| Sparse pointers | Graph Machine (Sept 2): edges + referral, replace 75% dense, retrieve 2–4/4096 no loss drop | O(n) long-ctx on 1 GPU, coarse-to-fine 77+ options |
| **Rotating sparse wiring** | **Hypercube wiring (Sept 16): fixed sparse rotated across layers, 1/32 links, 2.4x faster, wider LR window, beats hybrid attention** | **Replace Graph Machine with hypercube wiring — simpler, faster, no referral overhead** |
| Adaptive depth | RecurTrace (Loop Mem Attn + halting, 56.9%@2.0 loops), Looped Flows (denoising-trained recurrence, 58.8% ARC-AGI-1), Looped Mamba+CHASE | 1 pass default, loop 1–4x iff uncertainty high = System 1.5 |
| Byte-level | BLT (8B/4T, +50% infer flop win), ByteFlow Net (coding-rate chunking beats BPE), Kathleen 469K no-attn/no-pretrain 92.4% AG News | Bytes + entropy/coding-rate patching → no BPE shred, all scripts |
| **Relational attention** | **Dual Attention Transformer (Sept 17): separates sensory vs structural attention, better data efficiency & OOD** | **Add relational head for structural reasoning (code/invoice/logic) alongside sensory head** |
| **Block-sparse SGDR** | **Semantic-Geometric Decoupled Routing (Sept 19): pre-RoPE pooling + offline geometric prior, 5x speedup, training-free** | **Use for 8k→sparse routing instead of Graph Machine — simpler, no training** |
| **MoRE / MoME / MoM** | **Mixture of Reused/Memory Experts (Sept 14–16): cross-layer expert sharing, 18x routing combos, context-aware slots, multiple memory states** | **Replace single Titans MLP with MoM-style multi-memory (4 states, top-2 activated) — eliminates interference** |
| **Modern Transformers Implicit Hybrids** | **Head-wise Hybrid (Sept 2): RFIS/RPD taxonomy, NoPE FA for global + LA for local, better extrapolation** | **Head-wise hybrid in encoder: global retrieval heads (NoPE) + local positional heads (RoPE)** |
| Calibrated RL | Rewarding Doubt (ICLR 2026 log-score RL), ConfTuner (tokenized Brier, proven proper), RLCR (Brier+correctness, ECE 0.37→0.03), non-hackable rewards spectrum, ACL 2026 decision-token CE (ECE −9pts, RLVR overconfident) | RLCD++ = Brier+correctness + decision-token CE + per-type temps; Brier bounded (safe), log unbounded (hackable) |
| Diffusion LMs | LLaDA-8B matches LLaMA3-8B, breaks reversal curse, but systemic overconfidence ECE+0.52 | Parallel ≠ calibrated; need decoding-loop defense |
| **ConvMem** | **Hierarchical convolution (Sept 9): training-free, log-tree reasoning, parallelizable, beats RL-trained** | **Use for session memory compression — no RL, just convolution tree** |

**Synthesis:** nobody has combined all twelve. Jev has 1.5/12. Laya has 3/12. pico-type has 2/12 (different axis). The white space is the full stack.

---

## 1. NIRNAY-1 — architecture (frozen for v1)

```
Session Memory (MoM-style: 4 independent memory states, top-2 activated, router + forget gate, ConvMem log-tree compression)
        ↑↓ (read 256d mixed summary + write surprise-gated, momentum 0.9, decay αt)
Bytes (≤32k, UTF-8, no tokenizer)
  → ByteEmbed(256→96) → Conv(k3,5,7) → Coding-Rate Patching (ByteFlow Top-K, not entropy)
  → Patch Encoder: forked Laya 421M encoder (ModernBERT-large, LoRA-tuned; NoPE head-masking only in v1 — GLA local layers move to the v1.1 120M encoder)
       • v1: NoPE masking on 1/3 global-retrieval heads (local heads keep Laya RoPE as-is)
       • v1.1 (120M distill): full 1/3 NoPE FA + 2/3 GLA+RoPE split
  → Concept Bottleneck (MoME-style: VQ 32×128 product, chunk 4, mixture-of-M slots, learned gate, NCP loss)
  → Sparse Routing (Hypercube wiring: fixed sparse rotated across layers, 1/32 links, log₂n layers to reach all)
       + SGDR block router for long-range (pre-RoPE semantic + offline geometric prior)
  → Parallel Heads (per Q, one pass):
       Choice pointer ([MASK] per option, softmax) | Score pointer (RPS) | Noul sigmoid
       + Relational head (structural: code AST / invoice schema / logic forms)
       + Act/Escalate + Abstain ("none-of-above / unanswerable")
  → Per-(type, count) temperature → calibrated distribution
  → Halting Gate (Looped Flows style): if max(p)<τ or margin<δ or ECE-risk high → loop concept module ≤4x
  → Deep supervision losses at enc layers 4/8/12 (V-JEPA trick)
```

**Why each piece (no ornament):**

1. **Byte-first + coding-rate patching** — fixes Laya Khmer/Hindi silent failure + Jev text-only. 256 vocab, all scripts/binaries. ByteFlow coding-rate > BLT entropy for boundary detection. Patching keeps 32k bytes ≈ 8k patches ≈ Laya 1024-tok cost.
2. **Head-wise Hybrid encoder** — Modern Transformers paper: RFIS/RPD proves global heads should be NoPE, local heads RoPE. v1 applies NoPE masking on global heads only (fork-safe, no new params); full NoPE/GLA split lands with the v1.1 120M encoder.
3. **Concept bottleneck + MoME** — fixes 255/20-option collapse. MoME: mixture-of-M slots per concept, learned gate → context-aware sense disambiguation (python language vs snake). Stage-1: pico-coarse retrieve top-20 from 77+; Stage-2: pointer choose. NCP aux loss gives efficiency + compositionality.
4. **Hypercube sparse wiring** — Sept 16 paper: fixed sparse rotated across layers reaches every position in log₂n layers with 1/32 links, 2.4x faster, wider LR window, beats hybrid attention. Simpler than Graph Machine referral, no training overhead.
5. **SGDR block router** — Sept 19 paper (arXiv:2609.22884, ID verified 2026-09-22): training-free block routing, pre-RoPE semantic pooling + offline geometric prior, 5.03× over FlashAttn **at 128K** with <3.4ms routing overhead — expect smaller gains at our 8k, measured in the Wk 1–2 probe. Use for long-range (8k patches) where hypercube log-depth still costs.
6. **Halting gate (Looped Flows)** — fixes Jev/Laya always-1-pass (fail hard) vs diffusion always-N (wasteful). Looped Flows trains recurrence with local denoising objectives → recurrent states transfer computation over time. 1 pass @ ~35ms GPU / ~120ms CPU-ONNX; hard → 2–3 passes. Target: +2–4pts on hard slice at 1.6x avg cost.
7. **MoM session memory** — fixes statelessness. MoM paper: 4 memory states, top-2 activated, router + forget gate eliminates interference. ConvMem: log-tree compression of session history via hierarchical convolution (training-free). First calibrated decider with multi-memory.
8. **RLCD++ (non-hackable)** — fixes raw ECE 0.466 + RLVR overconfidence. Reward = correctness + Brier (bounded, proved) + decision-token CE (uniform target when wrong, ACL 2026) + tokenized Brier for verbalized conf. Never pure log-loss alone (hackable, 2607.04332). Temps fitted per bucket, shipped + refit script.
9. **Matryoshka tiers + dual runtime** — 16/64/192/576d heads from one trunk. CPU-ONNX for L0/L1 (~40–120ms), GPU for full (~35ms). Jev-compatible `POST /v1/systemone` (like Kev) → drop-in `base_url` swap. Router: bytes→patch path always; script-detect only for legacy token fallback (delete in v2).

**Param budget v1:** ~450M total = 421M Laya encoder fork (LoRA-tuned, Kev pattern — full fine-tune only the ~30M additions: byte path ~2M + concept/MoME 17M + heads/memory/sparse ~13M). Fits 2×T4 Kaggle (Laya proved 421M there in 4–5h); 8k-ctx phases staged last with accumulation. The 120M 12L/768d encoder is the v1.1 distill/CPU-tier target, not v1. No foundation pretrain — fork + post-train (legal, Apache-2.0).

**What we explicitly do NOT build:** generative text head (use LLM for prose — cascade, not replacement), video/audio encoders v1 (tokenizer path reserved), 70B scale, closed API.

---

## 2. Training recipe (reproducible, 5h Kaggle)

* **Base:** fork `convaiinnovations/laya` encoder + head (Apache-2.0, 421M) + `eulogik/pico-type` byte-embed/conv blocks (Apache-2.0). LoRA on the encoder, full fine-tune on the ~30M additions (Laya proves 7k updates suffice; new blocks need 15k).
* **Data (no Jev outputs ever):** public sets (Banking77, AG News, SST-2, BoolQ, MNLI, ToxicChat, Enron, DAIR Emotion) + typed-decisions train split (1,200 cases/6k decisions, CC) + synthetic policies (programmatic pairs like Kev, ~10k) + multilingual MASSIVE slice (51 langs) + unanswerable augmentation (evidence-removed pairs — Kev-9B Qwen3.5-gen holds 0.05 high-conf; our target is parity-or-better via abstain head + RLCD). Total ~40k decisions. All hashes frozen pre-train (Kev/JevBench discipline).
* **Loss:** `L = L_choiceCE + L_RPS(score) + L_BCE(noul) + L_relational + 0.3*L_NCP + 0.2*L_deep + λ*L_calCE` where `L_calCE` = one-hot target if correct else uniform (ACL 2026, λ=0.001–0.01) + RL phase: REINFORCE + group-mean baseline, Gaussian logit noise, Brier+correctness reward (RLCR, bounded). TD(λ=1.0) for multi-turn prefixes.
* **Calibration:** fit 1 temp per (type, option-count bucket: 2,3-5,6-20,21+) on held-out 20%. Ship `temperature_by_options.json` + `refit.py`. Report raw AND fitted (never fitted-only like Laya headline).
* **Compute:** Phase A SFT 3h + Phase B RLCD 2h on 2xT4 free tier. Notebook publishes exact seeds, hashes, costs (Kev $95 transparency as bar).

---

## 3. Benchmarks to beat (pre-registered, no cherry-pick)

| Suite | Target v1 (must clear ALL) | Current SOTA to beat |
|---|---|---|
| typed-decisions test 2k (frozen) | **≥0.80 acc, Brier ≤0.05, raw ECE ≤0.12 / fitted ≤0.07** | Laya-ft 0.766/0.062/0.213; Jev pub 0.727/0.148/0.144 |
| Zero-shot held-out families | **≥0.60** (vs Laya base 0.36) | Laya 0.362, Jev 0.727 (zero-shot king — must close gap) |
| Banking77 77-way | **≥0.80** (vs Laya 0.425) via coarse-to-fine + MoME | Jev 0.87, Laya 0.425 |
| AG News / SST-2 / Enron | ≥0.95 / ≥0.88 / ≥0.99 | Laya 0.95, Kathleen 92.4% AG, Laya 0.993 spam |
| Multilingual 51-lang | ≥45/51 >3x random, **no silent-confident**: max-conf on unread script ≤0.5 OR router-correct ≥99% | Laya 45/51 but silent-confident failure; pico 98.2% text-lang |
| Unanswerable | ≤0.05 high-conf rate when evidence removed (parity-or-better: Kev-9B already holds 0.05) | Kev-9B 0.05 (best), Kev-8B 0.26, Jev 0.09 — transfer-v9 unknowables, Qwen3.5-gen Sept 21 |
| JevBench v1.3.0 (independent) | **≥78.0** (beat Jev 74.4) | Jev 74.4, SemIf 73.1, Laya 54.4, kev 0.6B 62.5 (Qwen3-gen; current Qwen3.5-gen 0.8B/4B/9B not yet on JevBench) |
| Latency | P50 1Q ≤35ms T4 GPU / ≤100ms CPU-ONNX base; 10Q batched ≤80ms GPU | Laya 32.8ms/72ms, pico ~18ms CPU (README), Jev ~250ms API |
| Selective @50% cov | ≥0.94 | Laya ToxicChat 0.931 |
| ARC-AGI-1 (Looped Flows metric) | ≥55% (vs 58.8% Looped Flows) | Looped Flows 58.8%, Jev N/A |

**Rules:** same items/prompts/seeds for all rows; publish per-item JSONL + timing logs + ground-truth audit (LargitData lesson: 3.8%→32.2% after pipeline fix). Never compare local-GPU to hosted-API without labeling (TrueStandard lesson). Report zero-shot and fine-tuned in separate columns (rlresearch.ai lesson). No metric computed through our own act/escalate/abstain head counts as delivered-error evidence; delivered-error rates require an external ground-truth sample audit (arXiv:2609.01345: in-loop dashboards read 3% while true delivered error hit 32%).

---

## 4. Viral GTM (why this spreads and Jev can't copy fast)

1. **Day-1 open everything:** weights + code + notebook + eval hashes, Apache-2.0. Jev can't (closed); Laya did (won goodwill) but hid zero-shot weakness — we publish both columns.
2. **Drop-in Jev API:** `POST /v1/systemone`, `typesafe-sdk` works with `base_url=localhost`. Kev proved the pattern; we add memory+abstain+relational as extra fields (backward-compat).
3. **Live Space + video:** HF Space (8 workflows + 51-lang router + ARC demo, ZeroGPU) + 30s video: 10-Q batch at 7ms/q vs Jev API spinner; Doom-10Hz clone but open + $0; Khmer/Hindi router demo (Laya silent-failure vs NIRNAY abstain); ARC-AGI-1 demo with halting gate visible.
4. **JevBench submission week 1:** independent score is the only number skeptics trust. Pre-register targets above; publish pass/fail even if miss (Kev honesty premium: 2.9k⭐ on frozen evals + published misses).
5. **Kaggle + Vercel:** `laya_finetune`-style notebook (free GPUs, 5h) + Vercel AI Gateway `evaluate` adapter (Jev got 13% paid teams in 24h via Gateway — ride same rail, $0 self-host angle).
6. **Name + narrative:** NIRNAY (निर्णय = decision) — System 1.5 story: "Jev gave up strings. We gave up tokens too — and taught it when to think twice, what to remember, and how to reason structurally." Hindi-rooted, global-pronounceable, meme-able. `pip install nirnay`.

---

## 5. Build plan (90 days, 1 builder + agents + rented GPUs)

* **Days 1–14 — Fork + parity:** Laya encoder + pico byte-embed wired, hypercube wiring + SGDR router, NoPE head-masking only, tree/[MASK] scorer unified, temps refit script, Jev-compat server. Gate: 1 pass, N-Q, valid dists, JevBench ≥70.
* **Days 15–35 — Calibrate + MoME:** concept bottleneck + MoME slots + deep supervision + RLCD++ + per-bucket temps. Gate: fitted ECE ≤0.08, Banking77 ≥0.65 via 2-stage + MoME.
* **Days 36–60 — Think + remember:** halting gate (Looped Flows, ≤4 loops) + MoM memory (4 states, top-2, ConvMem compression). Gate: hard-slice +3pts at ≤1.7x cost; session-consistency test (restriction persists across 5 turns — LargitData gap where Jev 84.4% vs Gemma 95%); ARC-AGI-1 ≥50%.
* **Days 61–90 — Harden + launch:** 32k sparse path, multilingual sweep, ONNX export (CPU ≤100ms), Space + notebook + JevBench + Gateway adapter. Gate: all §3 targets or publish miss + fix plan.

**Cost:** <$300 rented H100/T4 (Kev precedent: $95 Qwen3.5 fam, $228 weekend). No raise needed for v1.

---

## 6. Kill / pivot gates (pre-registered)

* K1: fitted ECE not <0.10 after RLCD++ + refit → kill concept/MoME/memory, ship Laya-fork + hypercube + temps only.
* K2: halting never triggers (<5%) or never helps (<+1pt) → lock to 1-pass, compete on bytes+calibration+hybrid only.
* K3: Banking77 stays <0.65 after 2-stage+MoME → cap schema at 20, document, target triage/guardrail GTM only.
* K4: JevBench <74 after 90d → pivot to pico-v2 lane (edge guardrail) where latency/moat clearer.
* Never: distill Jev outputs, publish Jev column as own measurement, report fitted-only ECE, count a metric computed through our own act/escalate/abstain head as delivered-error evidence, train on verifier-rejected-tail self-labels (2609.01345).

---

## 7. Moat (why copycats trail)

* **Data flywheel:** every decision → outcome → recalibration + multi-memory update. Laya's 17M adapter + our MoM session memory compound per deployment; structure clones (OpenJev/mini-jev/Kev in days) don't. Flywheel labels come from external outcomes/ground truth only — never from verifier-rejected-tail self-training (arXiv:2609.01345).
* **Calibration + multi-memory + bytes + structure triad:** any two are clonable in a week; all four + reproducible recipe + benchmark transparency is 6–12 months of grind.
* **Neutrality:** Jev-compatible + ONNX + MCP + Gateway — benefits from every agent stack, depends on none.
* **Relational head:** code AST / invoice schema / logic forms — vertical moat for VasoolAI-class workflows without being vertical.

---

### Sources (verify, don't trust)

Jev launch (typesafe.ai), TechCrunch 9/18, TrueStandard 9/19, Aman Kumar 16k, LargitData JevBench, Firecrawl/Vercel, KDnuggets 9/21, Laya HF + site + BENCHMARKS.md + PyPI 0.3.5, rlresearch.ai audit, Kev GitHub/HF Qwen3-gen (0.5B/0.6B/4B/8B) + Qwen3.5-gen (0.8B/4B/9B, Sept 21 update) + ExplainX correction, JevBench v1.3.0, pico-type GitHub/HF/arXiv:2608.14658, Titans arXiv:2501.00663 + MIRAS blog, Graph Machine arXiv:2609.02881, **Hypercube wiring arXiv:2609.18145**, ByteFlow arXiv:2603.03583, BLT arXiv:2412.09871 + BLT-D 2605.08044 + MBP 2608.15454, Kathleen arXiv:2604.07969, Carpathian byte-LM, **Relational Attention / DAT arXiv:2609.20530**, **SGDR arXiv:2609.22884**, **MoRE arXiv:2609.18176**, **MoME arXiv:2609.15126**, **MoM arXiv:2502.13685**, **Modern Transformers Implicit Hybrids arXiv:2609.02986**, **ConvMem arXiv:2609.10441**, **Looped Flows arXiv:2609.11801**, Rewarding Doubt ICLR 2026, ConfTuner arXiv:2508.18847, RL-confidence arXiv:2607.04332, CALM ACL 2026 findings-acl.610, RLCR arXiv:2507.16806, ConfidenceBench arXiv:2607.20526, NCP-ArchPreview arXiv:2609.10715, V-JEPA 2.1 arXiv:2603.14482, LLaDA survey arXiv:2508.10875, Cheap Verifiers arXiv:2609.01345 (v2, verified 2026-09-23).

*Amendment log: 2026-09-22 v2 breakthrough plan incorporating all Sept papers. Numbers above are third-party/published where marked; re-measure on own hardware before launch claims.*
*Verification 2026-09-22: JevBench scores corrected to v1.3.0 board (Jev 74.4 / SemIf 73.1 / Laya 54.4 — earlier 75.4/74.7/70.1 wrong); ToS §2.3(f) claim removed (not found); pico latency → ~18ms README (3/5/12ms unconfirmed); Jev calibration numbers (ECE 0.107 etc.) still unconfirmed. Laya BENCHMARKS.md, typesafe.ai, evals, arXiv abstracts verified this pass.*
*Verification 2026-09-22 (Kev refresh): repo now Qwen3.5-gen 0.8B/4B/9B (2.9k⭐; JevBench rows are older Qwen3-gen); unknowable-rate target reframed as parity-or-better (Kev-9B 0.05 vs Jev 0.09); Kev README + ExplainX piece verified this pass.*
*Fixes 2026-09-22 (review feedback): SGDR arXiv:2609.22884 verified correct — kept, with 128K qualifier; param tier resolved to fork-421M (~450M total, LoRA + staged 8k ctx; 120M encoder moved to v1.1 distill target) — Kaggle feasibility rests on Laya's own 2xT4 proof; GLA-in-v1 spillover removed from diagram/:87/Days 1–14 (v1 = NoPE masking only, full split is v1.1).*
*Amendment 2026-09-23: Cheap Verifiers arXiv:2609.01345 (v2, verified 2026-09-23) — §3: no act/escalate/abstain-head metric as delivered-error evidence; §6: +external-ground-truth and verifier-rejected-tail clauses; §7: flywheel labels external-only. Rationale: β 0.12→0.55, frontier verifier escalates 46% vs 39% true error, corrective FT on rejected tail collapses student, dashboard 3% vs true 32% (ε_∞≲q₀β₀).*
*Amendment 2026-09-26 (training-stability, evidence in MEMORY.md + artifacts/ab* + logs/ab*): §1 concept-bottleneck implementation detail — (i) z is affine-free layer-normed before product quantization (free-scale z ran away unboundedly: healthy z_rms 0.089 → 13.5 dead, ncp spike 323 → eval 1/77; LN quotients out scale, ncp structurally O(1), zero new params); (ii) NCP loss gains a code-usage balance term w·Σ_chunks clamp(K·f·P−1, ≥0) (MoE load-balancing form, defaults w=0.01 τ=0.1; usage skewed to one code → constant z_q → feature homogenisation → 1/77 even without any spike). Loss coefficients (§2) and all §1 components unchanged; G8/G14/G15/G17 + contract + durability re-green; 250-step shipped-config run matches pre-fix baseline (test dense 0.5218 vs 0.5195); run-3 (7000 steps) relaunched with heldout probes + collapse abort + best-checkpoint retention.*