# v0.2 relaunch kit (2026-11-16)

The first launch led with a benchmark. This one leads with a working tool.
New facts to use, all verified:

- `pip install nirnay-450m` works (PyPI 0.2.1).
- `nirnay classify "text" --labels banking77` returns a calibrated result.
- `nirnay fit` adapts the readout to your own labels, with held-out
  accuracy printed before and after.
- Live demo: https://huggingface.co/spaces/GautamKishore/nirnay-demo
  (first request 12.4s, warm 1.8s).
- Custom labels work zero-shot: 0.9167 with original names, 0.8833 with
  neutral keys on a 10-label Banking77 subset (n=300).

Do not claim a fit accuracy gain unless the user's own run shows one.
The in-domain smoke test was saturated (0.95 -> 0.95).

## HN retry (one shot, new news, weekday 17:00 IST)

Title: `Show HN: NIRNAY 450M, a local classifier that learns your labels`

URL: `https://github.com/eulogik/nirnay`

First comment:

We shipped v0.2 of NIRNAY, a 450M open decision model. The first
release was a benchmark with a checkpoint. This one is a tool.

`pip install nirnay-450m`, then:

`nirnay classify "my card was charged twice" --labels banking77`

You get a calibrated answer, not text. If you have your own labels,
`nirnay fit` freezes the encoder, trains the choice readout on your
JSONL, early-stops on a held-out split, fits a temperature, and writes
a small checkpoint. It prints the held-out accuracy before and after.

Numbers, with the caveat: Banking77 test 0.8792 accuracy, 0.8785
macro-F1, fitted ECE 0.045 (n=3080). Same test cases through Jev
1.13.0: 0.803, zero-shot. We fine-tuned. Custom labels: 0.9167 with
original names, 0.8833 with neutral keys, n=300, 86-108ms on M4.

Limits: 512-token context, JevBench-hard 0.396, int8 quantized ONNX
flipped 4 of 24 decisions so it was rejected, not shipped.

Code and weights are Apache-2.0. Demo and docs in the repo. Happy to
answer anything about the quantization fixes or the fit path.

## Reddit r/LocalLLaMA (comment on the old thread or one new post)

NIRNAY 450M now has a real CLI and custom-label fine-tuning.

`pip install nirnay-450m`
`nirnay classify "my card was charged twice" --labels banking77`
`nirnay fit --data train.jsonl --labels labels.json --out my_model.pt`

The fit path freezes the encoder and trains the readout, so it runs in
minutes on a laptop, and it prints held-out accuracy before and after.
Banking77: 0.8792 accuracy, 0.8785 macro-F1, fitted ECE 0.045. Custom
10-label subset: 0.9167 with names, 0.8833 with neutral keys.
Apache-2.0, no API key. Demo: huggingface.co/spaces/GautamKishore/nirnay-demo

## X (4 posts, reply to the old thread)

1/4 NIRNAY 450M v0.2 is out. It is a tool now, not just a checkpoint.
`pip install nirnay-450m`
2/4 `nirnay classify "text" --labels banking77` returns calibrated
probabilities locally. No API key, no per-call bill.
3/4 `nirnay fit` adapts the readout to your labels, early-stops on a
held-out split, fits a temperature. Minutes on a laptop, not hours.
4/4 Numbers: 0.8792 Banking77, 0.8785 macro-F1, ECE 0.045. Custom
10-label: 0.9167. Apache-2.0. Demo: huggingface.co/spaces/GautamKishore/nirnay-demo

## LinkedIn

The first NIRNAY release was a benchmark. This one is a tool.

`pip install nirnay-450m`, then one command gets a calibrated answer
locally. If you have your own labels, `nirnay fit` adapts the readout
on your data and prints held-out accuracy before and after. Banking77:
87.9% accuracy, fitted ECE 0.045. Same test cases through the closed
API: 80.3%, zero-shot. Caveat stated: we fine-tuned.

For teams that cannot send tickets to an API, this is the shape that
works: open weights, local, calibrated, your labels. Repo and demo in
comments.

## Rules

- Link the working PyPI command, not the old git install.
- Link the Space before the model card.
- One HN retry only. If it dies, do not repost.
- Do not claim fit gains without a user-held-out number.
- Do not mention GGUF or Ollama as available.
- Do not submit to JevBench.
