# HN (post Thu 17:00 IST sharp)

## Title (pick one, do not edit after posting)

1. Show HN: 450M open decision model scores 87.9% on Banking77, trained on one Mac
2. Show HN: Fixing two training-collapse modes in quantized concept bottlenecks
3. Show HN: NIRNAY, an Apache-2.0 Jev alternative for intent classification

Recommendation: option 1. Numbers + small + open + trained-anywhere
is the HN bundle. Option 2 if the crowd skews research that morning.

## Comment body (post as first comment within 5 minutes)

Hey HN, we built NIRNAY, a 450M open-weight decision model for
classification-style work (banking intent, routing, urgency scoring).
One forward pass: state plus a typed question in, calibrated
probabilities out. No text generation.

Banking77 test (3,080 cases): 0.8792 accuracy, Brier 0.208, fitted
ECE 0.045. Same 3,080 cases through the Jev 1.13.0 API: 0.803.

Caveat first, since someone will check: we fine-tuned on the train
split, Jev answered zero-shot. That is the whole point of the project.
A model you fine-tune on your data beats the API on your data. +7.6
points on identical cases.

Two things broke during training and both fixes held through a 7,000
step run. First, the concept encoder output grew 150x while centroids
sat still, so the quantization loss exploded 300x. Fix: LayerNorm before
quantize, zero new params. Second, code usage collapsed to one code per
chunk with no loss spike at all, flatlining at 1/77 accuracy. Fix: a
load-balancing term over hard fractions times soft assignments.

Trained on one M4 Mac in about 4 hours. Zero cloud spend. Apache-2.0,
weights + code + raw eval JSONs all public:
github.com/eulogik/nirnay and huggingface.co/eulogik/nirnay-450m

Honest limits: 512-token context, JevBench-hard reasoning is weak
(0.396, published in the repo), CPU is 361ms per decision in PyTorch.
Happy to answer anything, especially on the quantization fixes.
