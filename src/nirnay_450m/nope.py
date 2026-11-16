"""NoPE head-masking for the v1 Laya fork."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


_NOPE_STATE = threading.local()
_ORIGINAL_ROTARY = None
_ROTARY_PATCHED = False


def split_head_range(num_heads: int, nope_fraction: float = 1.0 / 3.0) -> tuple[int, int]:
    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if not 0.0 <= nope_fraction <= 1.0:
        raise ValueError("nope_fraction must be in [0, 1]")
    if nope_fraction == 0.0:
        return 0, 0
    if num_heads == 1:
        return 1, 1
    nope_end = max(1, int(round(num_heads * nope_fraction)))
    return min(nope_end, num_heads), min(nope_end, num_heads)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _make_nope_apply():
    def apply_rotary_pos_emb_nope(q, k, cos, sin, unsqueeze_dim=1):
        active_end = getattr(_NOPE_STATE, "nope_end", None)
        if active_end is None or active_end <= 0:
            return _ORIGINAL_ROTARY(q, k, cos, sin, unsqueeze_dim)
        original_dtype = q.dtype
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        H = q.size(1)
        if cos.size(1) == 1 and H > 1:
            cos = cos.expand(cos.size(0), H, cos.size(2), cos.size(3)).clone()
            sin = sin.expand(sin.size(0), H, sin.size(2), sin.size(3)).clone()
            end = min(active_end, H)
            cos[:, :end] = 1.0
            sin[:, :end] = 0.0
        q_embed = (q.float() * cos) + (_rotate_half(q.float()) * sin)
        k_embed = (k.float() * cos) + (_rotate_half(k.float()) * sin)
        return q_embed.to(original_dtype), k_embed.to(original_dtype)

    return apply_rotary_pos_emb_nope


def _install_rotation_patch() -> None:
    global _ORIGINAL_ROTARY, _ROTARY_PATCHED
    if _ROTARY_PATCHED:
        return
    import transformers.models.modernbert.modeling_modernbert as mb

    _ORIGINAL_ROTARY = mb.apply_rotary_pos_emb
    mb.apply_rotary_pos_emb = _make_nope_apply()
    _ROTARY_PATCHED = True


@contextmanager
def nope_context(nope_end: int) -> Iterator[None]:
    previous = getattr(_NOPE_STATE, "nope_end", None)
    _NOPE_STATE.nope_end = int(nope_end)
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_NOPE_STATE, "nope_end")
            except AttributeError:
                pass
        else:
            _NOPE_STATE.nope_end = previous


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def apply_nope_mask(encoder: nn.Module, nope_fraction: float = 1.0 / 3.0) -> dict:
    before = param_count(encoder)
    num_heads = None
    if hasattr(encoder, "config") and hasattr(encoder.config, "num_attention_heads"):
        num_heads = int(encoder.config.num_attention_heads)
    if num_heads is None:
        for module in encoder.modules():
            candidate = getattr(module, "num_heads", None)
            if isinstance(candidate, int) and candidate > 0:
                num_heads = candidate
                break

    report = {
        "nope_fraction": nope_fraction,
        "num_heads": num_heads,
        "masked": False,
        "method": "unavailable",
        "params_unchanged": False,
    }
    if num_heads is None:
        report["params_unchanged"] = param_count(encoder) == before
        return report
    nope_end, _ = split_head_range(num_heads, nope_fraction)
    report["nope_end"] = nope_end
    old_handles = getattr(encoder, "_nirnay_nope_handles", [])
    for handle in old_handles:
        handle.remove()

    if nope_end > 0:
        try:
            _install_rotation_patch()

            def pre_hook(_module, _args):
                stack = getattr(_NOPE_STATE, "stack", [])
                previous = getattr(_NOPE_STATE, "nope_end", None)
                stack.append((previous, nope_end))
                _NOPE_STATE.stack = stack
                _NOPE_STATE.nope_end = nope_end

            def post_hook(_module, _args, _output):
                stack = getattr(_NOPE_STATE, "stack", [])
                if stack:
                    previous, _ = stack.pop()
                    if previous is None:
                        try:
                            delattr(_NOPE_STATE, "nope_end")
                        except AttributeError:
                            pass
                    else:
                        _NOPE_STATE.nope_end = previous
                if stack:
                    _NOPE_STATE.stack = stack
                else:
                    try:
                        delattr(_NOPE_STATE, "stack")
                    except AttributeError:
                        pass

            handles = [
                encoder.register_forward_pre_hook(pre_hook),
                encoder.register_forward_hook(post_hook),
            ]
            setattr(encoder, "_nirnay_nope_handles", handles)
            report["masked"] = True
            report["method"] = "patch_modernbert_apply_rotary_pos_emb"
        except Exception as error:
            report["method"] = f"failed:{type(error).__name__}"
            report["error"] = str(error)[:200]
    else:
        setattr(encoder, "_nirnay_nope_handles", [])
        report["method"] = "disabled"

    after = param_count(encoder)
    report["params_unchanged"] = after == before
    report["params"] = after
    return report
