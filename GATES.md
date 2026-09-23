# Gates: Days 1–14 parity (landed) + Days 15–35 entry: Calibrate + MoME modules (plan §5)

OWNS: src/nirnay/**, tests/**, scripts/**, pyproject.toml, uv.lock, GATES.md

Scope: Days 1–14: Laya-forked decision model with byte path, NoPE head-masking only, unified [MASK] scorer, per-bucket temps refit, and a Jev-compatible `/v1/systemone` server that returns valid distributions for N questions in one pass. Days 15–35 entry (this ledger's open gates): concept bottleneck + MoME module, deep supervision at encoder layers 4/8/12, RLCD++ loss suite (Brier + decision-token CE + REINFORCE group baseline), 2-stage coarse-to-fine scorer, and full G1–G7 regression after those modules land. Training milestones (fitted ECE ≤0.08, Banking77 ≥0.65) stay plan §5 contracts for rented-GPU sessions and are deliberately not gates here.

- [x] G1: Project installs and core package imports cleanly
  CHECK: uv run python -c "import nirnay; import nirnay.model; import nirnay.server; print('nirnay import ok')"
  EXPECT: nirnay import ok
  EVIDENCE: automatic-evidence=v1; definition-sha256=7acd432645e04fe3fb46afcf0b73efa58c464f00e54ba60dcc29a8de85fde2d7; exit=0; EXPECT=matched; output-sha256=7df0dfa8f97d03b85dae5a60342eca389227a5ed7ed39b340077cbf5be79e517; output-bytes=17; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G2: Laya checkpoint loads and one forward pass yields valid probability distributions (each choice/score dist sums to 1±0.01, all p in [0,1], noul in [0,1]) for a multi-question request
  CHECK: uv run python scripts/check_forward.py
  EXPECT: FORWARD_OK distributions=valid
  EVIDENCE: automatic-evidence=v1; definition-sha256=a0f1b75cfd3a38fea6562b3f864471a2b012728df388bc685992bf78300e14b1; exit=0; EXPECT=matched; output-sha256=1f26a11097517c0eb29480f79339b73a7286b44666158f50192be1c8a2bd653f; output-bytes=467; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

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
  EVIDENCE: automatic-evidence=v1; definition-sha256=7e34a3fd33b3f74f74539e59aabb04c1e7f86d143e930d83bfc96cdb44f44041; exit=0; EXPECT=matched; output-sha256=b7898f745fdd418c0d5384e9f4c2257c96b0e91abebf339746922a542a382d69; output-bytes=670; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G6: Per-(type, option-count) temperature refit script runs on held-out logits and writes temperature_by_options.json with finite temps in [0.5, 5.0]
  CHECK: uv run python scripts/check_temps_refit.py
  EXPECT: TEMPS_OK buckets_written=
  EVIDENCE: automatic-evidence=v1; definition-sha256=7dba9ae844013c55a86a82efeabbff3ae84ccf761a1009b9b3029b9cd26cb408; exit=0; EXPECT=matched; output-sha256=50deb1167efd25b1f65737b4f2259d2e20ab750517ddb727898bf361c67bbd7a; output-bytes=122; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G7: JevBench public suite (datasets/public) run against local server reports accuracy and schema validity without harness errors
  CHECK: uv run python scripts/check_jevbench_public.py
  EXPECT: JEVbench_PUBLIC_OK schema_validity=
  EVIDENCE: automatic-evidence=v1; definition-sha256=bf8bed61f6611b6e8cb82a2e88246f9ae28cce962fe81cfb8c3a7f9b2462a058; exit=0; EXPECT=matched; output-sha256=09319cce8fe2c59d1175bcb7ed864dfb964370dba00a5d4a44a36edb748c2790; output-bytes=723; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

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
  EVIDENCE: automatic-evidence=v1; definition-sha256=37e13cb258d3a03c21ad9d00311fef96603f6753bf116698d91699c2ee280b1d; exit=0; EXPECT=matched; output-sha256=e8a9140b1eff49c676eff5d6a01757783747a4be7a3aeb0ae111b5138a3c7c07; output-bytes=129; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G11: 2-stage coarse-to-fine scorer: stage-1 top-k retrieval includes the gold label for every fixture query, stage-2 choice distribution is valid (sums to 1, peaks at gold on separable fixtures), k defaults to ≤20 per plan §1
  CHECK: uv run python scripts/check_coarse_to_fine.py
  EXPECT: COARSE2FINE_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=73b0178ced573eb83cbed336b1fde7233518f7b5ea5f538c89c9380c96a629ce; exit=0; EXPECT=matched; output-sha256=7fee97b6a1b2c0b2366ad78370836f3f9d0914bac405c83572dedf369f4bb017; output-bytes=98; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries

- [x] G12: Full G1–G7 regression: all Days 1–14 check scripts still pass after Days 15–35 modules land (import, forward, byte path, NoPE, server schema, temps refit, JevBench public)
  CHECK: HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache" HF_HUB_OFFLINE=1 uv run python scripts/check_regression.py
  EXPECT: REGRESSION_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=1873e2aea80be6504bd55101771564fc3fe04e574046f302defe55e5c3669d8c; exit=0; EXPECT=matched; output-sha256=dafc62c6f50950c6ba3413c8f906a0b47ca1ebc78d5b6080f4edfed85edd3289; output-bytes=48; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/NIRNAY; path=afc7568fe86c/59 entries
