"""Concept bottleneck + MoME slots (plan §1; Days 15–35).

VQ 32×128 product quantization, chunk 4, mixture-of-M slots, learned gate,
NCP aux loss. Param budget line for concept/MoME additions: 17M (plan §1).

Design (frozen mapping of the plan line "VQ 32×128 product, chunk 4"):
  - latent product space = 128 dims
  - split into 4 chunks × 32 dims
  - each chunk quantized to its own codebook of 32 centroids → codes in [0, 32)
  - z layer-normed (affine-free) before quantization: scale is not a degree
    of freedom — free-scale z ran away unboundedly (death 2026-09-26)
  - MoME: M slots as additive context offsets; learned gate softmax-mixes them
  - NCP loss = VQ codebook + commitment + gate usage entropy + code-usage
    balance (compositionality; anti-collapse, see ncp_loss docstring)

Straight-through estimator keeps gradients flowing to the encoder projection;
codebook centroids receive gradients via the codebook term (z detached).
Serving integration lands with trained weights (training stack Days 15–35).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAM_BUDGET = 17_000_000  # plan §1: concept/MoME share of ~30M additions


@dataclass
class ConceptConfig:
    encoder_hidden: int = 1024
    latent_dim: int = 128
    num_chunks: int = 4
    codes_per_chunk: int = 32
    adapter_hidden: int = 2048
    num_slots: int = 4
    ncp_entropy_weight: float = 0.01
    # Anti-collapse pressure on VQ code usage. The 2026-09-26 deaths: without
    # this term usage skews to one code per chunk (mode → 0.95) → z_q≈constant
    # → commitment homogenises encoder features → constant argmax → eval 1/77
    # (both with and without an ncp magnitude spike). Mechanism is the
    # Switch-Transformer load-balancing form: K·Σ f·P − 1 ≥ 0 per chunk, where
    # f = hard assignment fractions (detached: always sees the true skew) and
    # P = mean soft assignment (differentiable). Linear in P → gradients flow
    # even at collapse, unlike entropy-of-soft which saturates to one-hot and
    # goes silent exactly when pressure is most needed. temp τ shapes P; z is
    # layer-normed before quantize (unit scale), so τ=0.1 tracks hard codes.
    code_usage_weight: float = 0.01
    code_usage_temp: float = 0.1


class MoMESlots(nn.Module):
    """Mixture-of-M slots: each slot is an additive offset; gate mixes them."""

    def __init__(self, latent_dim: int, num_slots: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, latent_dim) * 0.02)
        hidden = max(latent_dim // 2, 16)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_slots),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, L, D] → (mixed [B, L, D], gate_weights [B, L, M])."""
        weights = F.softmax(self.gate(z), dim=-1)
        slot_view = z.unsqueeze(2) + self.slots.to(z.dtype)  # [B, L, M, D]
        mixed = torch.einsum("blm,blmd->bld", weights, slot_view)
        return mixed, weights


class ConceptBottleneck(nn.Module):
    """Encoder hidden → product-VQ concepts → MoME-mixed bottleneck → hidden."""

    def __init__(self, config: ConceptConfig | None = None):
        super().__init__()
        cfg = self.config = config or ConceptConfig()
        if cfg.latent_dim % cfg.num_chunks != 0:
            raise ValueError("latent_dim must be divisible by num_chunks")
        self.chunk_dim = cfg.latent_dim // cfg.num_chunks

        self.encode = nn.Sequential(
            nn.Linear(cfg.encoder_hidden, cfg.adapter_hidden),
            nn.GELU(),
            nn.Linear(cfg.adapter_hidden, cfg.latent_dim),
        )
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(cfg.codes_per_chunk, self.chunk_dim)
                for _ in range(cfg.num_chunks)
            ]
        )
        for emb in self.codebooks:
            nn.init.uniform_(
                emb.weight, -1.0 / cfg.codes_per_chunk, 1.0 / cfg.codes_per_chunk
            )

        self.mome = MoMESlots(cfg.latent_dim, cfg.num_slots)
        self.decode = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.adapter_hidden),
            nn.GELU(),
            nn.Linear(cfg.adapter_hidden, cfg.encoder_hidden),
        )
        # Zero-init the decode head so the bottleneck is a residual that is
        # exactly identity at construction — a random non-residual replacement
        # wiped laya's pretrained hidden states (diagnosis 2026-09-25).
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)
        # NO LayerNorm on delta: LN after a zero-init head amplifies the first
        # Adam steps' microscopic noise toward O(1) (var+eps saturation), so the
        # residual injection ramps up exactly while the codebook scrambles —
        # eval cliff at steps 6-9 (bisection 2026-09-25, same failure class as
        # the byte path). Zero-init + choice gradients bound delta growth.

    def quantize(
        self, z: torch.Tensor, with_soft: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: [B, L, D] → (z_q [B, L, D], codes [B, L, n_chunks]).

        with_soft also returns soft code assignments
        [B, L, n_chunks, codes_per_chunk] = softmax(−dist/τ); differentiable
        through dist (grads → z and centroids), used only for the code-usage
        balance term — the forward path still uses hard argmin codes.
        """
        B, L, _ = z.shape
        n = self.config.num_chunks
        chunks = z.view(B, L, n, self.chunk_dim)
        codes_list = []
        zq_list = []
        soft_list = []
        temp = self.config.code_usage_temp
        for i, emb in enumerate(self.codebooks):
            c = chunks[:, :, i, :]
            dist = (
                c.pow(2).sum(-1, keepdim=True)
                - 2.0 * c @ emb.weight.t()
                + emb.weight.pow(2).sum(-1)
            )
            code = dist.argmin(dim=-1)  # [B, L] in [0, codes_per_chunk)
            codes_list.append(code)
            zq_list.append(emb(code))
            if with_soft:
                soft_list.append(F.softmax(-dist / temp, dim=-1))
        codes = torch.stack(codes_list, dim=-1)  # [B, L, n]
        z_q = torch.cat(zq_list, dim=-1)  # [B, L, D]
        if with_soft:
            soft = torch.stack(soft_list, dim=2)  # [B, L, n, K]
            return z_q, codes, soft
        return z_q, codes

    def ncp_loss(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor,
        gate_weights: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        soft_codes: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Codebook + commitment + gate usage + code-usage balance (aux loss).

        Terms 1-3 (grads→centroids via codebook, →encoder via commitment,
        →gate via −H) are the original NCP. The balance term
        w·Σ_chunks clamp(K·f·P − 1, ≥0) penalises collapsed code usage: without
        it usage skews to one code per chunk, z_q≈constant, and the commitment
        term homogenises encoder features — the 2026-09-26 deaths (eval 1/77
        constant argmax, with or without an ncp magnitude spike). f = hard
        assignment fractions (detached, always sees true skew), P = mean soft
        assignment (differentiable, so gradients reach z and centroids); valid
        tokens only.
        """
        soft = soft_codes
        hard = codes
        if attention_mask is not None:
            valid = attention_mask.to(torch.bool)
            z = z[valid]
            z_q = z_q[valid]
            gate_weights = gate_weights[valid]
            if soft is not None:
                soft = soft[valid]
            if hard is not None:
                hard = hard[valid]
        if z.numel() == 0:
            return z.new_zeros(())
        codebook = F.mse_loss(z_q, z.detach())
        commitment = F.mse_loss(z, z_q.detach())
        mean_gate = gate_weights.reshape(-1, gate_weights.size(-1)).mean(dim=0)
        entropy = -(mean_gate * (mean_gate + 1e-9).log()).sum()
        usage = -self.config.ncp_entropy_weight * entropy
        code_usage = z.new_zeros(())
        if soft is not None and hard is not None and soft.numel() > 0:
            # flatten to [tokens, n_chunks, K] — dim0 must be tokens even
            # without attention_mask (otherwise mean(0) averages the batch
            # and the balance sums over positions instead of chunks)
            soft_t = soft.reshape(-1, soft.size(-2), soft.size(-1))
            hard_t = hard.reshape(-1, hard.size(-1))
            n_chunks = soft_t.size(1)
            k = soft_t.size(-1)
            prob_mean = soft_t.mean(dim=0)  # [n_chunks, K], grads→z+centroids
            aux = z.new_zeros(())
            for i in range(n_chunks):
                frac = torch.bincount(
                    hard_t[:, i], minlength=k
                ).float() / max(hard_t.size(0), 1)  # detached by construction
                aux = aux + (k * (frac * prob_mean[i]).sum() - 1.0).clamp_min(0.0)
            code_usage = self.config.code_usage_weight * aux
        return codebook + commitment + usage + code_usage

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """hidden: [B, L, encoder_hidden] → (out same shape, info dict).

        compute_aux=False skips the NCP loss (inference/ONNX-export path:
        the loss value never feeds back into logits, and its empty-check
        breaks symbolic export).
        """
        z = self.encode(hidden)
        if attention_mask is not None:
            z = z * attention_mask.to(z.dtype).unsqueeze(-1)
        # Bound z's scale before quantization (anti-runaway, 2026-09-26):
        # free-scale z let the choice gradient climb |z| while the codebook
        # chased it — unbounded mse(z, z_q) → ncp spike 323 → death. LN
        # quotients out scale (unit variance per token), so the task gradient
        # can only rotate z, ncp stays O(1), and assignments become purely
        # direction-based. Affine-free: no params, no state_dict keys.
        z = F.layer_norm(z, (self.config.latent_dim,))
        z_q, codes, soft = self.quantize(z, with_soft=True)
        z_st = z + (z_q - z).detach()
        mixed, gate_w = self.mome(z_st)
        delta = self.decode(mixed + z_st)
        out = hidden + delta
        if attention_mask is not None:
            out = out * attention_mask.to(out.dtype).unsqueeze(-1)
        if compute_aux:
            loss = self.ncp_loss(
                z, z_q, gate_w, attention_mask=attention_mask, soft_codes=soft,
                codes=codes,
            )
        else:
            loss = z.new_zeros(())
        info = {"codes": codes, "gate_weights": gate_w, "ncp_loss": loss}
        return out, info


    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
