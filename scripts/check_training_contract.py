from __future__ import annotations

import torch

from nirnay.bytes import ByteFusion, BytePathConfig
from nirnay.coarse2fine import CoarseToFine
from nirnay.losses import assemble_plan_loss, plan_loss_from_batch
from nirnay.nope import split_head_range
from nirnay.rlcd import policy_gradient_loss
from nirnay.train import collate_examples


def main() -> int:
    torch.manual_seed(0)
    if split_head_range(16, 0.0) != (0, 0):
        raise AssertionError("zero NoPE fraction must not mask a head")
    fusion = ByteFusion(BytePathConfig(encoder_hidden=8, max_bytes=16))
    token_a = torch.zeros(2, 5, 8, requires_grad=True)
    token_b = torch.ones(2, 5, 8, requires_grad=True)
    byte_ids = torch.tensor([[1, 2, 3, 4], [250, 251, 252, 253]])
    byte_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    fused_a = fusion(token_a, byte_ids, byte_mask)
    if fused_a.shape != token_a.shape:
        raise AssertionError("byte fusion changed token embedding shape")
    # Identity-at-init contract (2026-09-25): zero-init encoder projection
    # keeps laya's pretrained embeddings intact at construction.
    if not torch.allclose(fused_a, token_a, atol=1e-6):
        raise AssertionError("byte fusion must be identity at init")
    # Trained-state contract: a nonzero encoder projection must alter embeddings.
    with torch.no_grad():
        fusion.path.to_encoder.weight.normal_(0.0, 0.05)
    fused_b = fusion(token_b, byte_ids, byte_mask)
    if fused_b.shape != token_b.shape or torch.allclose(fused_b, token_b):
        raise AssertionError("byte fusion cannot alter token embeddings")
    fused_b.pow(2).sum().backward()
    if token_b.grad is None or float(token_b.grad.abs().sum()) <= 0.0:
        raise AssertionError("byte fusion has no gradient")
    if (
        fusion.path.to_encoder.weight.grad is None
        or float(fusion.path.to_encoder.weight.grad.abs().sum()) <= 0.0
    ):
        raise AssertionError("byte fusion path not reachable by gradients")

    c2f = CoarseToFine(hidden_size=8, num_labels=77)
    query = torch.randn(3, 8, requires_grad=True)
    gold = torch.tensor([2, 40, 76])
    c2f_out = c2f(query, gold_idx=gold)
    c2f_loss = -c2f_out["full_probs"].gather(1, gold[:, None]).squeeze(1).log().mean()
    c2f_loss.backward()
    if c2f.label_embed.weight.grad is None or float(c2f.label_embed.weight.grad.abs().sum()) <= 0.0:
        raise AssertionError("coarse-to-fine has no label-bank gradient")
    if not torch.allclose(c2f_out["full_probs"].sum(-1), torch.ones(3), atol=1e-5):
        raise AssertionError("coarse-to-fine probabilities do not sum to one")

    items = [
        {
            "ids": [1, 2, 3],
            "markers": [1],
            "qtype": 0,
            "label": 0,
            "byte_ids": torch.tensor([4, 5, 6, 0]),
            "byte_mask": torch.tensor([1, 1, 1, 0]),
            "c2f_eligible": False,
            "c2f_label": -1,
        },
        {
            "ids": [1, 2],
            "markers": [1],
            "qtype": 2,
            "label": 1,
            "byte_ids": torch.tensor([7, 8, 0, 0]),
            "byte_mask": torch.tensor([1, 1, 0, 0]),
            "c2f_eligible": False,
            "c2f_label": -1,
        },
    ]
    batch = collate_examples(items, pad_id=0)
    if batch["byte_ids"].shape != (2, 4) or batch["c2f_mask"].tolist() != [False, False]:
        raise AssertionError("decision collation lost byte/c2f metadata")

    logits = torch.randn(2, 3, requires_grad=True)
    labels = torch.tensor([0, 1])
    marker_mask = torch.tensor([[True, False, False], [True, True, False]])
    relational = torch.tensor([0.3, 0.7], requires_grad=True)
    parts = plan_loss_from_batch(
        logits=logits,
        labels=labels,
        marker_mask=marker_mask,
        qtype=torch.tensor([0, 2]),
        ncp=torch.tensor(0.2),
        deep=torch.tensor(1.0),
        relational=relational,
    )
    if not torch.isfinite(parts.total) or float(parts.relational) <= 0.0:
        raise AssertionError("plan loss did not consume relational targets")
    parts.total.backward()
    if relational.grad is None or float(relational.grad.abs().sum()) <= 0.0:
        raise AssertionError("relational loss has no gradient")

    policy_logits = torch.randn(4, 5, requires_grad=True)
    actions = torch.randint(0, 5, (4,))
    advantages = torch.tensor([0.5, -0.5, 0.25, -0.25])
    policy = policy_gradient_loss(policy_logits, actions, advantages)
    policy.backward()
    if policy_logits.grad is None or float(policy_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("policy loss has no gradient")

    scalar = assemble_plan_loss(
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
    )
    if abs(float(scalar.total) - 0.2) > 1e-6:
        raise AssertionError("deep coefficient changed")
    print("TRAINING_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
