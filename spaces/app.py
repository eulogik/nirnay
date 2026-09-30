"""NIRNAY Space demo: Banking77 intent classification with calibrated probs.

Deploy: new HF Space (Gradio SDK, CPU) with this file + requirements.txt,
HF_TOKEN secret set (model repo is private until launch). Local test:
NIRNAY_CHECKPOINT=/path/to/phase_b.pt python app.py
"""

from __future__ import annotations

import os

import torch

_MODEL = None
_LABELS = None


def _load():
    global _MODEL, _LABELS
    if _MODEL is not None:
        return _MODEL
    from nirnay.agent import NirnayAgent

    ckpt = os.environ.get("NIRNAY_CHECKPOINT")
    if ckpt is None or not os.path.exists(ckpt):
        from huggingface_hub import snapshot_download

        repo = os.environ.get("NIRNAY_REPO", "eulogik/nirnay-450m")
        local = snapshot_download(repo, allow_patterns=["phase_b.pt"])
        ckpt = os.path.join(local, "phase_b.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = NirnayAgent(device=device, checkpoint_path=ckpt, enable_byte_path=False)
    _MODEL = agent
    from nirnay.data import BANKING77_LABELS

    _LABELS = list(BANKING77_LABELS)
    return agent


def classify(message: str):
    message = (message or "").strip()
    if not message:
        return "Type a banking message first.", {}
    agent = _load()
    out = agent.system_one(
        message,
        {
            "intent": {
                "type": "choice",
                "instructions": "Classify the banking intent of the user message.",
                "criteria": {lab: lab.replace("_", " ") for lab in _LABELS},
            }
        },
    )
    ans = out["answers"]["intent"]
    probs = dict(ans["probabilities"])
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    best, conf = top[0]
    return f"{best} ({conf:.2f})", {k: round(float(v), 4) for k, v in top}


def build_demo():
    import gradio as gr

    with gr.Blocks(title="NIRNAY 450M: banking intent") as demo:
        gr.Markdown(
            "# NIRNAY 450M\n"
            "Open-weight banking intent classifier, 87.9% on Banking77 test. "
            "Type a message, get the intent plus calibrated probabilities. "
            "Built by [Eulogik](https://eulogik.com)."
        )
        msg = gr.Textbox(
            label="Customer message",
            value="My card was charged twice for the same order.",
            lines=3,
        )
        btn = gr.Button("Classify")
        pred = gr.Textbox(label="Top intent")
        top5 = gr.Label(label="Top 5 probabilities")
        btn.click(classify, inputs=msg, outputs=[pred, top5], api_name="predict")
        msg.submit(classify, inputs=msg, outputs=[pred, top5], api_name="predict")
    return demo


if __name__ == "__main__":
    _load()
    build_demo().launch(server_port=int(os.environ.get("PORT", "7860")))
