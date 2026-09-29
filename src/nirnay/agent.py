"""High-level NIRNAY agent: Laya-fork runtime + v1 extensions."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from laya.common import QTYPES, build_sequence, confidence_from_probs, render_options

from .bytes import BytePath, BytePathConfig, encode_bytes
from .data import BANKING77_LABELS
from .nope import apply_nope_mask, param_count
from .temps import TEMP_MAX, TEMP_MIN, load_temps, temp_bucket


class NirnayAgent:
    """Decision runtime with optional trained NIRNAY checkpoint support."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        nope_fraction: float = 1.0 / 3.0,
        enable_byte_path: bool = True,
        temps_path: Optional[str] = None,
        checkpoint_path: str | Path | None = None,
        lora_rank: int = 8,
    ):
        import laya

        self._laya = laya.load(
            model_id_or_path, device=device, token=token, subfolder=subfolder
        )
        self.cfg = self._laya.cfg
        self.device = self._laya.device
        self.tok = self._laya.tok
        self.base_model = self._laya.model
        self.nope_report = apply_nope_mask(
            self.base_model.encoder, nope_fraction=nope_fraction
        )
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self._checkpoint_meta: dict[str, Any] = {}
        self.train_model: Any = None
        self.byte_path: Optional[BytePath] = None

        if self.checkpoint_path:
            from .train import NirnayTrainModel

            payload = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            self.train_model, self._checkpoint_meta = NirnayTrainModel.from_checkpoint(
                self.base_model, payload, device=self.device
            )
            checkpoint_nope = self._checkpoint_meta.get("nope_fraction")
            if checkpoint_nope is not None and not math.isclose(
                float(checkpoint_nope), nope_fraction, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"checkpoint NoPE fraction {checkpoint_nope} != runtime {nope_fraction}"
                )
            self.model = self.train_model
            self.train_model.eval()
            self.train_model.remove_hooks()
            if self.train_model.byte_fusion is not None:
                self.byte_path = self.train_model.byte_fusion.path
        else:
            self.model = self.base_model
            if enable_byte_path:
                hidden = int(getattr(self.model.encoder.config, "hidden_size", 1024))
                self.byte_path = BytePath(BytePathConfig(encoder_hidden=hidden))
                self.byte_path.to(self.device).eval()

        self.temps_override: dict[str, float] = {}
        if temps_path:
            self.load_temps(temps_path)
        self._param_count = param_count(self.model)
        self._lora_rank = lora_rank

    @property
    def param_count(self) -> int:
        return self._param_count

    def load_temps(self, path: str | Path) -> None:
        self.temps_override = {
            str(key): float(value) for key, value in load_temps(path).items()
        }
        for key, value in self.temps_override.items():
            if not math.isfinite(value) or not TEMP_MIN <= value <= TEMP_MAX:
                raise ValueError(f"temperature {key}={value} out of range")
        self._laya.temperature_by_options.update(self.temps_override)

    def encode_state_bytes(self, state: Union[str, dict, list]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, str):
            raw = state.encode("utf-8", errors="replace")
        elif isinstance(state, dict):
            raw = json.dumps(state, ensure_ascii=False).encode("utf-8")
        else:
            raw = json.dumps(state, ensure_ascii=False).encode("utf-8")
        max_len = self.byte_path.config.max_bytes if self.byte_path else 4096
        return encode_bytes(raw, max_len=max_len)

    @torch.no_grad()
    def byte_features(self, state: Union[str, dict, list]) -> Optional[torch.Tensor]:
        if self.byte_path is None:
            return None
        ids, mask = self.encode_state_bytes(state)
        return self.byte_path(ids, mask)

    @staticmethod
    def _to_internal(question: dict[str, Any]) -> dict[str, Any]:
        qtype = question["type"]
        criteria = question.get("criteria")
        if qtype == "choice" and isinstance(criteria, list):
            criteria = {str(i): None for i in range(len(criteria))}
        instructions = question["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": qtype, "ins": instructions, "crit": criteria}

    def _temperature(self, qtype: str, k: int) -> float:
        qt = QTYPES[qtype]
        default = self._laya.temperature[qt]
        return float(self.temps_override.get(temp_bucket(qtype, k), default))

    def _trained_system_one(
        self,
        state: Union[str, dict, list],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        from .train import collate_examples

        items = []
        qdefs: dict[str, dict[str, Any]] = {}
        for qid, raw_q in questions.items():
            q = self._to_internal(raw_q)
            qdefs[qid] = q
            seq, markers = build_sequence(
                self.tok,
                state,
                q,
                int(self.cfg.get("max_len", 512)),
                int(self.cfg.get("head_max_len", 192)),
            )
            option_count = len(render_options(q))
            if len(markers) != option_count:
                raise ValueError(
                    f"question {qid!r} options exceed head budget"
                )
            byte_ids, byte_mask = self.encode_state_bytes(state)
            c2f_eligible = (
                q["t"] == "choice"
                and isinstance(q.get("crit"), dict)
                and tuple(q["crit"].keys()) == BANKING77_LABELS
            )
            items.append(
                {
                    "ids": seq,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "byte_ids": byte_ids[0],
                    "byte_mask": byte_mask[0],
                    "c2f_eligible": c2f_eligible,
                    "c2f_label": -1,
                }
            )
        batch = collate_examples(items, self.tok.pad_token_id)
        batch = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        self.train_model.eval()
        with torch.no_grad():
            out = self.train_model(batch)
        logits = out["logits"].detach().float().cpu().numpy()
        act = torch.softmax(out["act_logits"].detach().float(), dim=-1).cpu().numpy()
        answers: dict[str, dict[str, Any]] = {}
        for row, qid in enumerate(questions):
            q = qdefs[qid]
            option_count = len(render_options(q))
            t_scale = self._temperature(q["t"], option_count)
            z = logits[row, :option_count] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            confidence = round(confidence_from_probs(p, option_count), 4)
            action = {"act_probability": round(float(act[row, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {
                        key: round(float(value), 4) for key, value in zip(keys, p)
                    },
                    "confidence": confidence,
                    "action": action,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(option_count) * p).sum()), 4),
                    "legend": {str(i): value for i, value in enumerate(q["crit"])},
                    "probabilities": {
                        str(i): round(float(value), 4) for i, value in enumerate(p)
                    },
                    "confidence": confidence,
                    "action": action,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": action,
                }
        return {
            "model": "nirnay-1",
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum().item()),
                "output_tokens": 0,
            },
        }

    @torch.no_grad()
    def system_one(
        self,
        state: Union[str, dict, list],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate typed questions in one parallel pass."""
        if self.train_model is not None:
            return self._trained_system_one(state, questions)
        return self._laya.system_one(state, questions)

    predict = system_one

    def report(self) -> dict[str, Any]:
        return {
            "params": self._param_count,
            "device": str(self.device),
            "nope": self.nope_report,
            "byte_path": self.byte_path is not None,
            "byte_fusion": bool(
                self.train_model is not None
                and self.train_model.byte_fusion is not None
            ),
            "coarse_to_fine": bool(
                self.train_model is not None and self.train_model.c2f is not None
            ),
            "checkpoint_loaded": self.train_model is not None,
            "checkpoint_path": self.checkpoint_path,
            "temps_override_buckets": sorted(self.temps_override.keys()),
        }


def load(
    model_id_or_path: str = "convaiinnovations/laya",
    device: Optional[str] = None,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
    **kwargs: Any,
) -> NirnayAgent:
    return NirnayAgent(
        model_id_or_path,
        device=device,
        token=token,
        subfolder=subfolder,
        **kwargs,
    )
