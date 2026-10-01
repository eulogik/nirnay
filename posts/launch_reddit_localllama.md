# r/LocalLLaMA (post Thu 20:00 IST)

## Title

NIRNAY 450M: Apache-2.0 banking intent classifier, 87.9% on Banking77,
runs on CPU (trained on one Mac, $0 cloud)

## Body

Built a small open decision model for intent classification and routing.
450M params (Laya fork plus ~30M), one forward pass gives calibrated
probabilities, no text generation.

Banking77 test, 3,080 cases: **0.8792**, Brier 0.208, fitted ECE 0.045.
Same cases through Jev 1.13.0: 0.803. Caveat, stated plainly: we
fine-tuned, Jev answered zero-shot. Fine-tune beats API on your own
data, that is the thesis.

Runs local: 209ms on M4 GPU, 361ms on CPU, batch-1, PyTorch. No GGUF
or Ollama build yet (custom heads need converter work), so bring a
Python env for now.

```bash
pip install git+https://github.com/eulogik/nirnay
```

```python
from nirnay.agent import NirnayAgent
agent = NirnayAgent(device="cpu", checkpoint_path="phase_b.pt", enable_byte_path=False)
out = agent.system_one("My card was charged twice.", {"intent": {
    "type": "choice",
    "instructions": "Classify the banking intent.",
    "criteria": {lab: lab.replace("_", " ") for lab in BANKING77_LABELS}}})
```

(BANKING77_LABELS comes from nirnay.data; full snippet in the repo
README.)

Also in the repo: the two training collapses we hit and fixed (scale
runaway 150x, silent usage collapse to 1/77), a 9-page paper draft,
and every eval as raw JSON. JevBench-hard is weak (0.396, long docs),
published as-is.

Repo: github.com/eulogik/nirnay. Weights: huggingface.co/eulogik/nirnay-450m.
Apache-2.0. Built by Eulogik.
