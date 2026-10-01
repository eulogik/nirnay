# X thread (post after HN is live, link the Dev.to post in 1/6)

1/6
We trained a 450M decision model on one Mac ($0 cloud). It scores
87.9% on Banking77 test. Jev scores 80.3% on the same 3,080 cases.
Apache-2.0, weights + code open. What broke along the way:

2/6
Caveat first: we fine-tuned, Jev answered zero-shot. Same test cases
though. Fine-tune beats API on your own data. That is the whole bet.

3/6
Death 1: the concept encoder output grew 150x while centroids sat
still. VQ loss spiked 300x. Fix: LayerNorm before quantize. Zero new
params.

4/6
Death 2: code usage silently collapsed to one code per chunk. No loss
spike. Accuracy flatlined at 1/77. Fix: load-balancing aux loss over
hard fractions times soft probs.

5/6
Numbers: 0.8792 acc, Brier 0.208, fitted ECE 0.045. 209ms on M4 GPU,
361ms CPU. 19/19 checks green. Paper draft + raw eval JSONs in repo.

6/6
Limits, stated plainly: 512-token context, JevBench-hard 0.396, no
ONNX/GGUF yet. Repo: github.com/eulogik/nirnay. Model:
huggingface.co/eulogik/nirnay-450m. Built by @eulogik.
