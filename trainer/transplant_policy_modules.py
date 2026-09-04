#!/usr/bin/env python3
"""Create diagnostic checkpoints by transplanting policy submodules.

The base checkpoint supplies every parameter/config/optimizer-independent field.
Only explicitly requested policy-module prefixes are copied from the donor.
This is intended for controlled localization experiments, not model averaging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


MODULE_PREFIXES = {
    "action_in": ("action_in.",),
    "policy_attn": ("policy_attn.",),
    "policy_attn_in": ("policy_attn.in_proj_",),
    "policy_attn_out": ("policy_attn.out_proj.",),
    "policy_norm": ("policy_norm.",),
    "policy_out": ("policy_out.",),
}

ATTN_SLICE_MODULES = {
    "policy_attn_q": 0,
    "policy_attn_k": 1,
    "policy_attn_v": 2,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base")
    ap.add_argument("donor")
    ap.add_argument("output")
    ap.add_argument(
        "--modules", nargs="+",
        choices=tuple(MODULE_PREFIXES) + tuple(ATTN_SLICE_MODULES), required=True,
    )
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    if not isinstance(base, dict) or not isinstance(donor, dict):
        raise SystemExit("both checkpoints must be dictionary checkpoints")
    if "state_dict" not in base or "state_dict" not in donor:
        raise SystemExit("checkpoint is missing state_dict")
    structural_keys = (
        "token_features", "action_features", "width", "layers", "heads", "ffn",
        "aux_targets", "value_attention", "topout_attention", "dropout",
    )
    base_cfg = base.get("config", {})
    donor_cfg = donor.get("config", {})
    structural_diff = {
        key: (base_cfg.get(key), donor_cfg.get(key))
        for key in structural_keys
        if base_cfg.get(key) != donor_cfg.get(key)
    }
    if structural_diff:
        raise SystemExit(
            f"base/donor structural model configs differ; transplant is unsafe: {structural_diff}"
        )

    base_state = dict(base["state_dict"])
    donor_state = donor["state_dict"]
    prefixes = tuple(
        prefix
        for module_name in args.modules
        if module_name in MODULE_PREFIXES
        for prefix in MODULE_PREFIXES[module_name]
    )
    copied: list[str] = []
    for name, value in donor_state.items():
        if prefixes and name.startswith(prefixes):
            if name not in base_state or base_state[name].shape != value.shape:
                raise SystemExit(f"incompatible tensor: {name}")
            base_state[name] = value.clone()
            copied.append(name)

    width = int(base_cfg["width"])
    for module_name in args.modules:
        if module_name not in ATTN_SLICE_MODULES:
            continue
        part = ATTN_SLICE_MODULES[module_name]
        start = part * width
        end = (part + 1) * width
        for tensor_name in ("policy_attn.in_proj_weight", "policy_attn.in_proj_bias"):
            if tensor_name not in base_state or tensor_name not in donor_state:
                raise SystemExit(f"missing attention projection tensor: {tensor_name}")
            if base_state[tensor_name].shape != donor_state[tensor_name].shape:
                raise SystemExit(f"incompatible tensor: {tensor_name}")
            value = base_state[tensor_name].clone()
            value[start:end].copy_(donor_state[tensor_name][start:end])
            base_state[tensor_name] = value
            copied.append(f"{tensor_name}[{start}:{end}]")

    if not copied:
        raise SystemExit("no tensors matched requested modules")

    output = dict(base)
    output["state_dict"] = base_state
    output.pop("optimizer_state_dict", None)
    output.pop("sampling_generator_state", None)
    output["transplant"] = {
        "base": str(args.base),
        "donor": str(args.donor),
        "modules": list(args.modules),
        "tensors": copied,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved          {args.output}")
    print("modules        " + ", ".join(args.modules))
    print("copied         " + ", ".join(copied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
