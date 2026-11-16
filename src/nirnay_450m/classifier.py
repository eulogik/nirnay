"""High-level, one-call classifier API.

The point is the five-minute experience:

    from nirnay_450m import Classifier
    clf = Classifier()
    result = clf("My card was charged twice", {
        "duplicate_charge": "charged twice",
        "refund_status": "asking about a refund",
    })
    print(result.choice, result.confidence)

Weights come from the public Hugging Face repo and are cached automatically.
Labels are just IDs mapped to descriptions. The model reads the descriptions,
so arbitrary label sets work without retraining (in-domain).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_REPO = "eulogik/nirnay-450m"
DEFAULT_CHECKPOINT = "phase_b.pt"
DEFAULT_BASE_MODEL = "convaiinnovations/laya"
DEFAULT_INSTRUCTION = "Classify the state into exactly one of the labels."


@dataclass
class Decision:
    """One calibrated choice."""

    choice: str
    confidence: float
    probabilities: dict[str, float]
    escalated: bool = False
    top: list[tuple[str, float]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "escalated": self.escalated,
            "top": [[k, round(v, 6)] for k, v in self.top],
        }

    def __str__(self) -> str:
        return f"{self.choice} ({self.confidence:.2f})"


def normalise_labels(labels: Any) -> dict[str, str]:
    """Accept a JSON file, a dict, a list, or the string "banking77"."""
    if labels is None:
        raise ValueError("labels are required")
    if isinstance(labels, str):
        if labels.strip().lower() in ("banking77", "banking-77"):
            from .data import BANKING77_LABELS

            return {lab: lab.replace("_", " ") for lab in BANKING77_LABELS}
        path = Path(labels)
        if not path.exists():
            raise FileNotFoundError(f"labels file not found: {path}")
        labels = json.loads(path.read_text())
    if isinstance(labels, Mapping):
        out: dict[str, str] = {}
        for key, value in labels.items():
            k = str(key).strip()
            if not k:
                raise ValueError("label IDs must be non-empty")
            desc = str(value).strip() if value is not None else ""
            out[k] = desc or k.replace("_", " ")
        if not out:
            raise ValueError("labels mapping is empty")
        return out
    if isinstance(labels, (list, tuple)):
        if not labels:
            raise ValueError("labels list is empty")
        return {str(x): str(x).replace("_", " ") for x in labels}
    raise TypeError(f"unsupported labels type: {type(labels).__name__}")


class Classifier:
    """Local calibrated classifier. Model loads once per instance."""

    def __init__(
        self,
        model: str = DEFAULT_REPO,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        base_model: str = DEFAULT_BASE_MODEL,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        enable_byte_path: bool = False,
    ):
        self.model = model
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.device = device
        self.base_model = base_model
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.token = token
        self.enable_byte_path = enable_byte_path
        self._agent = None

    def resolve_checkpoint(self) -> str:
        if self.checkpoint:
            path = Path(self.checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            return str(path)
        local = Path(self.model)
        if local.is_file():
            return str(local)
        if local.is_dir() and (local / DEFAULT_CHECKPOINT).exists():
            return str(local / DEFAULT_CHECKPOINT)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self.model,
            filename=DEFAULT_CHECKPOINT,
            cache_dir=self.cache_dir,
            token=self.token,
        )

    def _load(self):
        if self._agent is None:
            from .agent import NirnayAgent

            self._agent = NirnayAgent(
                self.base_model,
                device=self.device,
                checkpoint_path=self.resolve_checkpoint(),
                enable_byte_path=self.enable_byte_path,
                token=self.token,
            )
        return self._agent

    def classify(
        self,
        state: Any,
        labels: Any,
        instructions: str | None = None,
        *,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> Decision:
        """One state, one label set, one calibrated decision."""
        criteria = normalise_labels(labels)
        agent = self._load()
        out = agent.system_one(
            state,
            {
                "decision": {
                    "type": "choice",
                    "instructions": instructions or DEFAULT_INSTRUCTION,
                    "criteria": criteria,
                }
            },
        )
        ans = out["answers"]["decision"]
        probs = {k: float(v) for k, v in ans["probabilities"].items()}
        choice = ans["choice"]
        confidence = float(ans.get("confidence", probs.get(choice, 0.0)))
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if top_k:
            ranked = ranked[: max(1, int(top_k))]
        escalated = bool(threshold is not None and confidence < threshold)
        return Decision(
            choice=choice,
            confidence=confidence,
            probabilities=probs,
            escalated=escalated,
            top=ranked,
            raw=ans,
        )

    __call__ = classify

    def classify_many(
        self,
        records: Iterable[Mapping[str, Any]],
        labels: Any,
        instructions: str | None = None,
        *,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[Decision]:
        """Batch helper. Each record needs "state"; "instructions" is optional."""
        criteria = normalise_labels(labels)
        results: list[Decision] = []
        for record in records:
            state = record.get("state")
            if state is None:
                raise ValueError('every record needs a "state" field')
            results.append(
                self.classify(
                    state,
                    criteria,
                    instructions=record.get("instructions", instructions),
                    threshold=threshold,
                    top_k=top_k,
                )
            )
        return results


def classifier(**kwargs: Any) -> Classifier:
    """Convenience factory matching the docs."""
    return Classifier(**kwargs)
