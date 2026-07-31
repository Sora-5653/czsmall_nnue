# SPDX-License-Identifier: MIT
"""Export a trained TetraFormer to the `.tetrawts` format the C++ engine loads.

This is the return leg of the round trip: the engine exports positions, PyTorch
trains on them, and this writes the weights back into a form
`tetra::TetraFormerEvaluator` can run inside the search.

Format (all little-endian)::

    "TETRAWTS"  version:u32
    token_features:u32 action_features:u32 width:u32 layers:u32
    heads:u32 ffn:u32 aux_targets:u32
    tensor_count:u32
    repeated: name_len:u32 name ndim:u32 dims:u32... data:f32...

Named tensors mean the loader verifies it got what it expected instead of
trusting an offset table, which is what makes a shape mismatch a clear error
rather than silent garbage.

Usage::

    python trainer/export_weights.py model.pt weights.tetrawts
"""

from __future__ import annotations

import argparse
import struct
import sys

import torch

from tetraformer import TetraFormer, TetraFormerConfig

MAGIC = b"TETRAWTS"
VERSION = 1


def _tensor_blob(name: str, t: torch.Tensor) -> bytes:
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    out = bytearray()
    encoded = name.encode("utf-8")
    out += struct.pack("<I", len(encoded))
    out += encoded
    out += struct.pack("<I", arr.ndim)
    for d in arr.shape:
        out += struct.pack("<I", int(d))
    out += arr.tobytes(order="C")
    return bytes(out)


def export(model: TetraFormer, path: str) -> int:
    cfg = model.cfg
    sd = model.state_dict()

    # The exact set the C++ loader validates. Keeping this list explicit means a
    # change to the architecture fails loudly here rather than at inference.
    names = [
        "token_in.weight",
        "token_in.bias",
        "norm.weight",
        "action_in.weight",
        "action_in.bias",
        "policy_attn.in_proj_weight",
        "policy_attn.in_proj_bias",
        "policy_attn.out_proj.weight",
        "policy_attn.out_proj.bias",
        "policy_norm.weight",
        "policy_out.weight",
        "policy_out.bias",
        "value_head.0.weight",
        "value_head.0.bias",
        "value_head.2.weight",
        "value_head.2.bias",
        "aux_head.0.weight",
        "aux_head.0.bias",
        "aux_head.2.weight",
        "aux_head.2.bias",
    ]
    for layer in range(cfg.layers):
        p = f"blocks.{layer}."
        names += [
            p + "n1.weight",
            p + "attn.in_proj_weight",
            p + "attn.in_proj_bias",
            p + "attn.out_proj.weight",
            p + "attn.out_proj.bias",
            p + "n2.weight",
            p + "ffn.gate.weight",
            p + "ffn.up.weight",
            p + "ffn.down.weight",
        ]

    missing = [n for n in names if n not in sd]
    if missing:
        raise KeyError(f"model is missing expected tensors: {missing}")

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack(
        "<8I",
        VERSION,
        cfg.token_features,
        cfg.action_features,
        cfg.width,
        cfg.layers,
        cfg.heads,
        cfg.ffn,
        cfg.aux_targets,
    )
    blob += struct.pack("<I", len(names))
    for n in names:
        blob += _tensor_blob(n, sd[n])

    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def load_checkpoint(path: str) -> TetraFormer:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, TetraFormer):
        return ckpt
    cfg = TetraFormerConfig(**ckpt["config"])
    model = TetraFormer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="a .pt saved by train.py")
    ap.add_argument("output", help="destination .tetrawts")
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint)
    size = export(model, args.output)
    cfg = model.cfg
    print(f"exported {args.output}")
    print(f"  parameters {model.parameter_count() / 1e6:.3f}M")
    print(f"  width {cfg.width}  layers {cfg.layers}  heads {cfg.heads}  ffn {cfg.ffn}")
    print(f"  size {size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())