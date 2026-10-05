---
title: NIRNAY 450M banking intent
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: apache-2.0
models:
- eulogik/nirnay-450m
---

# NIRNAY 450M: banking intent, live

Open-weight banking intent classifier, 87.9% on Banking77 test (3,080 cases), fitted ECE 0.045. Type a customer message, get the intent plus calibrated probabilities. Runs on ZeroGPU, free tier.

Model: [eulogik/nirnay-450m](https://huggingface.co/eulogik/nirnay-450m). Code: [github.com/eulogik/nirnay](https://github.com/eulogik/nirnay). Built by [Eulogik](https://eulogik.com), Apache-2.0.

Fine-tune caveat: the model was fine-tuned on the Banking77 train split. Same test cases as the Jev 0.803 reference, which answered zero-shot.
