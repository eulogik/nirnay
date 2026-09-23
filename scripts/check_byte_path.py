"""G3: byte path produces encoder-shaped patches end-to-end."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from nirnay.bytes import BytePath, BytePathConfig, encode_bytes

    cfg = BytePathConfig(encoder_hidden=1024, max_bytes=512)
    path = BytePath(cfg).eval()
    sample = b"Duplicate charge on invoice #4411 - refund twice billed March."
    ids, mask = encode_bytes(sample, max_len=cfg.max_bytes)
    with torch.no_grad():
        patches = path(ids, mask)
    if patches.dim() != 3 or patches.size(-1) != cfg.encoder_hidden:
        print(f"BYTE_PATH_FAIL shape={tuple(patches.shape)}", file=sys.stderr)
        return 1
    if patches.size(1) < 1 or not torch.isfinite(patches).all():
        print("BYTE_PATH_FAIL non-finite or empty", file=sys.stderr)
        return 1

    # Wire into agent.byte_features as well when model available is optional here.
    from nirnay.bytes import BytePath as BP  # noqa: F401

    print(f"BYTE_PATH_OK patches={tuple(patches.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
