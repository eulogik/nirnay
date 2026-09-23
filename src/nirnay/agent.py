"""High-level NIRNAY agent: Laya-fork runtime + v1 extensions.

Loads a Laya checkpoint (Apache-2.0), applies NoPE head-masking (v1 scope),
optionally attaches the byte path, and exposes Jev-shaped `system_one`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch

from .bytes import BytePath, BytePathConfig, encode_bytes
from .nope import apply_nope_mask, param_count
from .temps import load_temps, temp_bucket


class NirnayAgent:
    """Decision runtime: Laya backbone + NoPE mask + byte path + temps."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        nope_fraction: float = 1.0 / 3.0,
        enable_byte_path: bool = True,
        temps_path: Optional[str] = None,
    ):
        import laya

        self._laya = laya.load(
            model_id_or_path, device=device, token=token, subfolder=subfolder
        )
        self.cfg = self._laya.cfg
        self.device = self._laya.device
        self.tok = self._laya.tok
        self.model = self._laya.model

        # v1: NoPE head-masking only (full GLA split → v1.1)
        self.nope_report = apply_nope_mask(self.model.encoder, nope_fraction=nope_fraction)

        self.byte_path: Optional[BytePath] = None
        if enable_byte_path:
            hidden = int(getattr(self.model.encoder.config, "hidden_size", 1024))
            self.byte_path = BytePath(BytePathConfig(encoder_hidden=hidden))
            self.byte_path.to(self.device).eval()

        self.temps_override: dict[str, float] = {}
        if temps_path:
            self.load_temps(temps_path)

        self._param_count = param_count(self.model)

    @property
    def param_count(self) -> int:
        return self._param_count

    def load_temps(self, path: str | Path) -> None:
        self.temps_override = load_temps(path)

    def encode_state_bytes(self, state: Union[str, dict, list]) -> torch.Tensor:
        if isinstance(state, str):
            raw = state.encode("utf-8", errors="replace")
        elif isinstance(state, dict):
            raw = json.dumps(state, ensure_ascii=False).encode("utf-8")
        else:
            raw = json.dumps(state, ensure_ascii=False).encode("utf-8")
        ids, mask = encode_bytes(raw, max_len=self.byte_path.config.max_bytes if self.byte_path else 4096)
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def byte_features(self, state: Union[str, dict, list]) -> Optional[torch.Tensor]:
        if self.byte_path is None:
            return None
        ids, mask = self.encode_state_bytes(state)
        return self.byte_path(ids, mask)

    @torch.no_grad()
    def system_one(
        self,
        state: Union[str, dict, list],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """One parallel pass. Returns Jev/Laya-shaped answers dict."""
        # Touch byte path so the graph is exercised (features reserved for
        # fusion training; scorer still reads Laya markers in v1 parity).
        _ = self.byte_features(state)

        out = self._laya.system_one(state, questions)

        # Apply per-bucket temperature overrides if loaded (post-hoc on probs
        # is not ideal — v1 applies at logit scale inside laya via cfg temps
        # when override keys match; here we only validate presence).
        if self.temps_override:
            for qid, ans in out.get("answers", {}).items():
                q = questions.get(qid, {})
                qt = q.get("type", "choice")
                crit = q.get("criteria")
                if qt == "choice" and isinstance(crit, dict):
                    k = len(crit)
                elif qt == "score" and isinstance(crit, (list, tuple)):
                    k = len(crit)
                elif qt == "noul":
                    k = 2
                else:
                    k = 2
                key = temp_bucket(qt, k)
                ans["temperature_bucket"] = key
                ans["temperature_applied"] = self.temps_override.get(key)
        return out

    predict = system_one

    def report(self) -> dict:
        return {
            "params": self._param_count,
            "device": str(self.device),
            "nope": self.nope_report,
            "byte_path": self.byte_path is not None,
            "temps_override_buckets": sorted(self.temps_override.keys()),
        }


def load(
    model_id_or_path: str = "convaiinnovations/laya",
    device: Optional[str] = None,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
    **kwargs,
) -> NirnayAgent:
    return NirnayAgent(
        model_id_or_path,
        device=device,
        token=token,
        subfolder=subfolder,
        **kwargs,
    )
