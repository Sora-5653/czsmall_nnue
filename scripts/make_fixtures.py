#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the binary test fixtures in tests/data/.

`cpp_matches_pytorch_exactly` pins the C++ forward pass against PyTorch, which
needs two artefacts that cannot live in a text file:

    tests/data/tiny_model.tetrawts        a small trained-shaped model
    tests/data/tiny_model_reference.bin   its reference forward pass

Both are fully determined by the seeds below, so they are reproducible rather
than precious. Run this once after checking out a tree that lacks them:

    python scripts/make_fixtures.py

Requires torch. Without it, the two parity tests skip and the other 261 pass.
"""

from __future__ import annotations

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "trainer"))

import torch  # noqa: E402

from tetraformer import TetraFormer, TetraFormerConfig  # noqa: E402
from export_weights import export  # noqa: E402

# These constants define the fixture. Changing any of them invalidates the
# committed reference, so the C++ test would need regenerating too.
SEED_MODEL = 0
SEED_INPUT = 1
WIDTH, LAYERS, HEADS, FFN = 32, 2, 4, 96
TOKENS, ACTIONS = 7, 5
TOKEN_FEATURES, ACTION_FEATURES = 24, 28


def main() -> int:
    out_dir = os.path.join(ROOT, "tests", "data")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(SEED_MODEL)
    model = TetraFormer(
        TetraFormerConfig(width=WIDTH, layers=LAYERS, heads=HEADS, ffn=FFN)
    ).eval()

    weights_path = os.path.join(out_dir, "tiny_model.tetrawts")
    size = export(model, weights_path)

    torch.manual_seed(SEED_INPUT)
    tokens = torch.randn(1, TOKENS, TOKEN_FEATURES)
    actions = torch.randn(1, ACTIONS, ACTION_FEATURES)
    with torch.no_grad():
        logits, wdl, _aux = model(
            tokens, torch.ones(1, TOKENS), actions, torch.ones(1, ACTIONS)
        )
    policy = torch.softmax(logits, dim=-1)[0]
    value = torch.softmax(wdl, dim=-1)[0]

    blob = bytearray()
    blob += struct.pack("<II", TOKENS, ACTIONS)
    for v in tokens[0].flatten().tolist():
        blob += struct.pack("<f", v)
    for v in actions[0].flatten().tolist():
        blob += struct.pack("<f", v)
    for v in policy.tolist():
        blob += struct.pack("<f", v)
    for v in value.tolist():
        blob += struct.pack("<f", v)

    ref_path = os.path.join(out_dir, "tiny_model_reference.bin")
    with open(ref_path, "wb") as fh:
        fh.write(blob)

    print(f"wrote {weights_path} ({size / 1024:.1f} KB)")
    print(f"wrote {ref_path} ({len(blob)} bytes)")
    print("policy:", [round(float(x), 6) for x in policy])
    print("wdl:   ", [round(float(x), 6) for x in value])
    print("\nnow run: make test")
    return 0


if __name__ == "__main__":
    sys.exit(main())