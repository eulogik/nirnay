from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from nirnay.losses import LossParts
from nirnay.train import PhaseASFT


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"value": self.weight * batch["x"].sum()}

    def loss_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        cal_lambda: float = 0.005,
    ) -> LossParts:
        value = out["value"].pow(2)
        zero = value.new_zeros(())
        return LossParts(
            choice_ce=value,
            rps=zero,
            noul_bce=zero,
            relational=zero,
            ncp=zero,
            deep=zero,
            cal_ce=zero,
            total=value,
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.detach().cpu().clone()}

    def load_checkpoint_payload(self, payload: dict) -> dict:
        self.weight.data.copy_(payload["model"]["weight"])
        return dict(payload.get("meta", {}))


def main() -> int:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "phase.pt"
        batch = {"x": torch.tensor([1.0])}
        first_model = TinyModel()
        first_trainer = PhaseASFT(first_model, lr=0.1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            first_history = first_trainer.fit(
                [batch],
                steps=2,
                log_every=1,
                checkpoint_path=path,
                checkpoint_every=1,
                checkpoint_meta={"phase": "test"},
            )
        if len(first_history) != 2 or not path.exists():
            raise AssertionError("periodic checkpoint was not written")
        if "sec/step=" not in output.getvalue() or "steps/s=" not in output.getvalue():
            raise AssertionError("timing labels are not explicit")
        if "rate=" in output.getvalue():
            raise AssertionError("mislabeled rate field remains")

        resumed_model = TinyModel()
        resumed_trainer = PhaseASFT(resumed_model, lr=0.1)
        meta = resumed_trainer.load_checkpoint(path)
        if meta.get("completed_steps") != 2:
            raise AssertionError(f"resume metadata={meta}")
        with contextlib.redirect_stdout(io.StringIO()):
            extra = resumed_trainer.fit(
                [batch],
                steps=3,
                start_step=2,
                checkpoint_path=path,
                checkpoint_every=1,
                checkpoint_meta={"phase": "test"},
            )
        if len(extra) != 1 or resumed_model.weight.item() == 1.0:
            raise AssertionError("resume did not continue the optimizer/model")
        final_meta = resumed_trainer.load_checkpoint(path)
        if final_meta.get("completed_steps") != 3:
            raise AssertionError(f"final checkpoint metadata={final_meta}")
    print("TRAIN_DURABILITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
