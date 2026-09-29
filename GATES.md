# Gates: Days 1–14 parity (landed) + Days 15–35 entry: Calibrate + MoME modules (plan §5)

OWNS: src/nirnay/**, tests/**, scripts/**, pyproject.toml, uv.lock, GATES.md

Scope: Days 1–14: Laya-forked decision model with byte path, NoPE head-masking only, unified [MASK] scorer, per-bucket temps refit, and a Jev-compatible `/v1/systemone` server that returns valid distributions for N questions in one pass. Days 15–35: concept bottleneck + MoME, deep supervision at 4/8/12, RLCD++ losses, 2-stage coarse-to-fine, G1–G7 regression, training-stack wiring (§2 data pipeline, plan loss assembly, Phase A SFT smoke), then runnable training path — real Banking77 corpus, Phase A CLI checkpointing, Phase B RLCD smoke, held-out eval with raw+fitted ECE. Full milestones (fitted ECE ≤0.08, Banking77 ≥0.65) stay plan §5 contracts for rented-GPU sessions and are deliberately not gates here.

- [x] G1: Project installs and core package imports cleanly
  CHECK: uv run python -c "import nirnay; import nirnay.model; import nirnay.server; print('nirnay import ok')"
  EXPECT: nirnay import ok
  EVIDENCE: automatic-evidence=v1; definition-sha256=7acd432645e04fe3fb46afcf0b73efa58c464f00e54ba60dcc29a8de85fde2d7; exit=0; EXPECT=matched; output-sha256=7df0dfa8f97d03b85dae5a60342eca389227a5ed7ed39b340077cbf5be79e517; output-bytes=17; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G2: Laya checkpoint loads and one forward pass yields valid probability distributions (each choice/score dist sums to 1±0.01, all p in [0,1], noul in [0,1]) for a multi-question request
  CHECK: uv run python scripts/check_forward.py
  EXPECT: FORWARD_OK distributions=valid
  EVIDENCE: automatic-evidence=v1; definition-sha256=a0f1b75cfd3a38fea6562b3f864471a2b012728df388bc685992bf78300e14b1; exit=0; EXPECT=matched; output-sha256=382cbdc3ca5ad115ee734c90f64c3134670a3e1d5e0e545ef5d70c723bfa6f73; output-bytes=610; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G3: Byte path produces patches that feed the encoder (byte tensor → conv patches → encoder input_ids path exercised end-to-end)
  CHECK: uv run python scripts/check_byte_path.py
  EXPECT: BYTE_PATH_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=876cad5688efd1a1686833ed40db4f5bf72164b6b3431e82b757deae27212ab6; exit=0; EXPECT=matched; output-sha256=20cca144d9bc836015e82e6867836b740d6449627a88a162be4875a4b74f70ec; output-bytes=35; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G4: NoPE head-masking is active on the designated global-retrieval head fraction (v1: mask applied; local heads untouched) and does not change param count
  CHECK: uv run python scripts/check_nope_mask.py
  EXPECT: NOPE_OK masked_heads>0 params_unchanged=true
  EVIDENCE: automatic-evidence=v1; definition-sha256=5346725c8b9c6b69e4a8eb95d5116c9adc7845ffe518aa16001410c655032da3; exit=0; EXPECT=matched; output-sha256=4977a8b9182b1b6350b3fa2a9ffa84e76a20bba6facb3401e996cf4dd2a7e0b6; output-bytes=104; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G5: Jev-compatible server answers POST /v1/systemone with schema-valid answers for noul/choice/score (typesafe adapter field names: answers.decision.{noul|choice|probabilities})
  CHECK: uv run python scripts/check_server_schema.py
  EXPECT: SERVER_SCHEMA_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=7e34a3fd33b3f74f74539e59aabb04c1e7f86d143e930d83bfc96cdb44f44041; exit=0; EXPECT=matched; output-sha256=189ae9211b7aea51f3e4e47c76dbcf6a2760be221b40f188269286f668ecf89b; output-bytes=813; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G6: Per-(type, option-count) temperature refit script runs on held-out logits and writes temperature_by_options.json with finite temps in [0.5, 5.0]
  CHECK: uv run python scripts/check_temps_refit.py
  EXPECT: TEMPS_OK buckets_written=
  EVIDENCE: automatic-evidence=v1; definition-sha256=7dba9ae844013c55a86a82efeabbff3ae84ccf761a1009b9b3029b9cd26cb408; exit=0; EXPECT=matched; output-sha256=50deb1167efd25b1f65737b4f2259d2e20ab750517ddb727898bf361c67bbd7a; output-bytes=122; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G7: JevBench public suite (datasets/public) run against local server reports accuracy and schema validity without harness errors
  CHECK: uv run python scripts/check_jevbench_public.py
  EXPECT: JEVbench_PUBLIC_OK schema_validity=
  EVIDENCE: automatic-evidence=v1; definition-sha256=bf8bed61f6611b6e8cb82a2e88246f9ae28cce962fe81cfb8c3a7f9b2462a058; exit=0; EXPECT=matched; output-sha256=2b5e9fddc03c188ea7b82c8f31aa00eba86c77f1173de50e173e96815913670a; output-bytes=866; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G8: Concept bottleneck + MoME module: product-VQ codes land in [0, 32), slot gate rows sum to 1, NCP/VQ loss finite and >0, forward output shape matches hidden size, module params within the 17M concept/MoME budget
  CHECK: uv run python scripts/check_concepts.py
  EXPECT: CONCEPTS_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=10ede9bdf5750ab7f03667a9dea1d15e589dcb5b4dccd89752377d4cc6de3c27; exit=0; EXPECT=matched; output-sha256=300be3a99b297a581bd52eede044a9e78b4e64892d400b241ccd2b22f039cc8e; output-bytes=504; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G9: Deep supervision module computes finite weighted auxiliary losses from encoder states at layers 4/8/12 only (other layers ignored), total matches 0.2*L_deep weighting from plan §2
  CHECK: uv run python scripts/check_deepsup.py
  EXPECT: DEEPSUP_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=723eb54f7531a58d0bd41708b836e8927590b171ec3ba4c96d881af9e21524bb; exit=0; EXPECT=matched; output-sha256=fe109c76f25589c91c2d4437b9b17e2a5330349e6a226ec3a8c0bf4da4d25c85; output-bytes=468; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G10: RLCD++ loss suite: Brier reward in [0, 1], decision-token CE is one-hot when correct / uniform when wrong, REINFORCE group-mean advantages average ~0 within each group, combined loss finite
  CHECK: uv run python scripts/check_rlcd.py
  EXPECT: RLCD_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=37e13cb258d3a03c21ad9d00311fef96603f6753bf116698d91699c2ee280b1d; exit=0; EXPECT=matched; output-sha256=24e4f58c4b5c3bf63caf50ec4259feaaaad807d8ce2806973550e4fb1972365e; output-bytes=146; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G11: 2-stage coarse-to-fine scorer: stage-1 top-k retrieval includes the gold label for every fixture query, stage-2 choice distribution is valid (sums to 1, peaks at gold on separable fixtures), k defaults to ≤20 per plan §1
  CHECK: uv run python scripts/check_coarse_to_fine.py
  EXPECT: COARSE2FINE_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=73b0178ced573eb83cbed336b1fde7233518f7b5ea5f538c89c9380c96a629ce; exit=0; EXPECT=matched; output-sha256=7fee97b6a1b2c0b2366ad78370836f3f9d0914bac405c83572dedf369f4bb017; output-bytes=98; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G12: Full G1–G7 regression: all Days 1–14 check scripts still pass after Days 15–35 modules land (import, forward, byte path, NoPE, server schema, temps refit, JevBench public)
  CHECK: HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_regression.py
  EXPECT: REGRESSION_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=1873e2aea80be6504bd55101771564fc3fe04e574046f302defe55e5c3669d8c; exit=0; EXPECT=matched; output-sha256=dafc62c6f50950c6ba3413c8f906a0b47ca1ebc78d5b6080f4edfed85edd3289; output-bytes=48; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G13: §2 data pipeline: builds decision examples from synthetic policies + Banking77 (77 official intents), freezes stable SHA256 content hashes, splits 80/20 train/heldout with zero id overlap, formats Laya-compatible state/question/answer fields, no Jev-derived fields
  CHECK: uv run python scripts/check_data_pipeline.py
  EXPECT: DATA_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=2a82bac6c8dfc188329fe3199719294e5593fd938073ccdf49a71a7ba8c9029e; exit=0; EXPECT=matched; output-sha256=910980e4904e0c8ee5eac9fdc3e3dee4fd5aa8163482dfb82f5e6052b5c8ad30; output-bytes=101; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G14: Plan §2 loss assembly: total = choiceCE + RPS + noulBCE + relational + 0.3·NCP + 0.2·deep + λ·calCE with exact coefficients on fixed tensors (reproduces hand-computed sum), RPS/noul/choice components each finite as defined
  CHECK: uv run python scripts/check_loss_assembly.py
  EXPECT: LOSS_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=71c4c48e874a2a5f8e0d46021d67eab4c5acf6c8402094640aa63c195ad71e51; exit=0; EXPECT=matched; output-sha256=6bf52a24120ed6347a14dc4d65d4cc103c8324fe3097132100e2735ee4880b94; output-bytes=94; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G15: Phase A SFT smoke on Laya backbone: encoder base weights frozen, LoRA + concept/deepsup trainable, plan loss backward+step succeeds for ≥3 steps with finite loss, deepsup grads present at layers 4/8/12, NCP loss in graph
  CHECK: HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_phase_a_smoke.py
  EXPECT: PHASE_A_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=4df6139e56ab59cc418c017138dbf52c0cfb1d11af29491dc5a65d0369812d11; exit=0; EXPECT=matched; output-sha256=5d480a2b3ff449bd82b86be7d73442fa6822621758c058093383257e44dd0bd2; output-bytes=801; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G16: Real Banking77 corpus lands as §2 training data: ≥10000 examples, all 77 official intents present, stable content hashes, 80/20 split with zero id overlap, no Jev-derived fields
  CHECK: uv run python scripts/check_banking77_data.py
  EXPECT: BANKING77_DATA_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=172de36b37ecc1b9055bbc60346d0fea63741be4ebde0861fa87ebd9e90d9996; exit=0; EXPECT=matched; output-sha256=8b358caf761f02a80ef979902dd1f7fcb73170684c4a6c7e427f206c254f8782; output-bytes=80; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G17: Phase A CLI runs end-to-end on the local mix: writes freeze manifest, finite loss history JSON, and a trainable-state checkpoint; encoder base remains frozen
  CHECK: HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_phase_a_cli.py
  EXPECT: PHASE_A_CLI_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=6363a34005524df14a44b5e4dff6a26a8c2a784bb7871beb47259666e1648486; exit=0; EXPECT=matched; output-sha256=ea79001c55840c4370aa017f52221e7d1bd9b98ce7f5d4bfebaedac554610b34; output-bytes=877; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G18: Phase B RLCD smoke: Gaussian logit noise sampling, group-mean advantages sum ~0 per group, Brier+correctness rewards in [0,1], policy loss finite over ≥3 steps
  CHECK: uv run python scripts/check_phase_b_rlcd.py
  EXPECT: PHASE_B_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=a5dfa6aaaa35e7cccc4c28fbd33c0cfcee4c4a2b7a2ebe450d7cddbce65416e9; exit=0; EXPECT=matched; output-sha256=17c38a7b6ba1f790423f1ad68849cc09e6855fb56106ca62e2a86eb99d1642ba; output-bytes=506; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G19: Held-out eval reports accuracy, Brier, raw ECE, and fitted ECE together (never fitted-only) on a non-empty batch with finite metrics
  CHECK: HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_eval_ece.py
  EXPECT: EVAL_ECE_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=2b5a7cbc898d2e30e8c0adbeb7608be7e7a1de1b698053b272962afdda8765f8; exit=0; EXPECT=matched; output-sha256=eb5fc3c9aa6997ef2b31f99d1e116a6bf39cf90335f217018faed3e9c76ae3cc; output-bytes=120; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries
