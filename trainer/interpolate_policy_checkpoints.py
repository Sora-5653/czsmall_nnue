# SPDX-License-Identifier: MIT
"""Interpolate only the policy head between two compatible checkpoints.

This is intended for Pareto probing between a strength champion and a
policy-only specialist. Non-policy parameters must be bit-identical so value,
auxiliary heads, and the shared trunk cannot drift silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


POLICY_PREFIXES = (
    "action_in.",
    "policy_attn.",
    "policy_norm.",
    "policy_out.",
)


def is_policy_key(key: str) -> bool:
    return key.startswith(POLICY_PREFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="strength/reference checkpoint")
    ap.add_argument("specialist", help="policy-only specialist checkpoint")
    ap.add_argument("output", help="output checkpoint")
    ap.add_argument("--alpha", type=float, required=True,
                    help="0 = base policy, 1 = specialist policy")
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    specialist = torch.load(args.specialist, map_location="cpu", weights_only=False)
    a = base["state_dict"]
    b = specialist["state_dict"]
    if a.keys() != b.keys():
        raise SystemExit("checkpoint state_dict keys differ")

    changed_policy: list[str] = []
    for key in a:
        av = a[key]
        bv = b[key]
        if not torch.is_tensor(av) or not torch.is_tensor(bv):
            if av != bv:
                raise SystemExit(f"non-tensor state differs at {key}")
            continue
        if av.shape != bv.shape or av.dtype != bv.dtype:
            raise SystemExit(f"tensor contract differs at {key}")
        if is_policy_key(key):
            if torch.is_floating_point(av):
                a[key] = torch.lerp(av, bv, args.alpha)
            elif not torch.equal(av, bv):
                raise SystemExit(f"non-floating policy state differs at {key}")
            if not torch.equal(av, bv):
                changed_policy.append(key)
        elif not torch.equal(av, bv):
            raise SystemExit(
                f"non-policy parameter differs at {key}; interpolation refused"
            )

    base["state_dict"] = a
    base["interpolation"] = {
        "base": str(args.base),
        "specialist": str(args.specialist),
        "alpha": float(args.alpha),
        "scope": "policy_head_only",
        "changed_keys": changed_policy,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f"saved {out}")
    print(f"alpha {args.alpha:g}")
    print(f"changed policy tensors {len(changed_policy)}")
    for key in changed_policy:
        print(f"  {key}")


if __name__ == "__main__":
    main()
