"""Concept bottleneck + MoME slots (plan §1; Days 15–35).

VQ 32×128 product quantization, chunk 4, mixture-of-M slots, learned gate,
NCP aux loss. Param budget line for concept/MoME additions: 17M (plan §1).

Design (frozen mapping of the plan line "VQ 32×128 product, chunk 4"):
  - latent product space = 128 dims
  - split into 4 chunks × 32 dims
  - each chunk quantized to its own codebook of 32 centroids → codes in [0, 32)
  - MoME: M slots as additive context offsets; learned gate softmax-mixes them
  - NCP loss = VQ codebook + commitment + gate usage entropy (compositionality)

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
        self.norm = nn.LayerNorm(cfg.encoder_hidden)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, L, D] → (z_q [B, L, D], codes [B, L, n_chunks])."""
        B, L, _ = z.shape
        n = self.config.num_chunks
        chunks = z.view(B, L, n, self.chunk_dim)
        codes_list = []
        zq_list = []
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
        codes = torch.stack(codes_list, dim=-1)  # [B, L, n]
        z_q = torch.cat(zq_list, dim=-1)  # [B, L, D]
        return z_q, codes

    def ncp_loss(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Codebook (grads→centroids) + commitment (grads→encoder) + gate usage."""
        codebook = F.mse_loss(z_q, z.detach())
        commitment = F.mse_loss(z, z_q.detach())
        mean_gate = gate_weights.reshape(-1, gate_weights.size(-1)).mean(dim=0)
        entropy = -(mean_gate * (mean_gate + 1e-9).log()).sum()
        usage = -self.config.ncp_entropy_weight * entropy
        return codebook + commitment + usage

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """hidden: [B, L, encoder_hidden] → (out same shape, info dict)."""
        z = self.encode(hidden)
        z_q, codes = self.quantize(z)
        z_st = z + (z_q - z).detach()  # straight-through
        mixed, gate_w = self.mome(z_st)
        out = self.norm(self.decode(mixed + z_st))
        loss = self.ncp_loss(z, z_q, gate_w)
        info = {"codes": codes, "gate_weights": gate_w, "ncp_loss": loss}
        return out, info

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
