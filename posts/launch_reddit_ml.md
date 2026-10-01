# r/MachineLearning (post Thu 20:30 IST)

## Title

Fixing two training-collapse modes in product-quantized concept
bottlenecks (450M decision model, 87.9% Banking77)

## Body

We trained a 450M decision model (Laya fork + product-VQ concept
bottleneck + MoME + pointer) and hit two distinct collapses, both
ending at constant predictions. Isolating them took feature-path
probes and a directional-gradient test, so writing it up properly.

Collapse 1, scale runaway: encoder output RMS 0.089 healthy to 13.5
dead while centroids sat near 0.04, VQ loss spiking 300x. A unit step
along the task gradient raises scale (+455), along the NCP gradient
lowers it (-2980). Task-driven runaway with the codebook chasing.
Fix: affine-free LayerNorm on z before quantize. Zero new params,
old checkpoints still load.

Collapse 2, usage collapse: per-chunk dominant share to 0.97 with no
loss spike at all; commitment then homogenizes features. A run with
only fix 1 still dies. Fix: load-balancing aux w*sum(max(K*f*P - 1,0))
over detached hard fractions and soft assignments (linear in P, so
gradients never vanish). An entropy-deficit first attempt failed A/B
twice (too soft early, saturates late); reported as a negative result.

Plus process around it: heldout probes every 250 steps, best-checkpoint
retention, abort on halved accuracy. Caught four real collapses.

Result: 0.8792 Banking77 test (3,080), Brier 0.208, fitted ECE 0.045.
9-page draft with all tables and 19 verified refs in the repo under
papers/. Code + weights + eval JSONs Apache-2.0:
github.com/eulogik/nirnay, huggingface.co/eulogik/nirnay-450m
