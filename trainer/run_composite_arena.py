# SPDX-License-Identifier: MIT
"""Factor policy and value checkpoints independently in the GPU Arena.

This is a diagnostic runner: it can form a model whose policy logits come from one
checkpoint while WDL/aux outputs come from another.  It is useful for determining
whether an Arena gain is caused by policy, value, or policy-value interaction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from ablation_models import load_ablation_checkpoint
from gpu_arena import evaluate, report


class CompositePolicyValue(nn.Module):
    def __init__(self, policy_model: nn.Module, value_model: nn.Module):
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, tokens, token_mask, actions, action_mask):
        policy, _, _ = self.policy_model(tokens, token_mask, actions, action_mask)
        _, value, aux = self.value_model(tokens, token_mask, actions, action_mask)
        return policy, value, aux


def make_composite(policy_path: str, value_path: str, device: torch.device) -> nn.Module:
    if policy_path == value_path:
        return load_ablation_checkpoint(policy_path, device)
    policy = load_ablation_checkpoint(policy_path, device)
    value = load_ablation_checkpoint(value_path, device)
    model = CompositePolicyValue(policy, value)
    model.to(device)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-policy", required=True)
    ap.add_argument("--candidate-value", required=True)
    ap.add_argument("--champion-policy", required=True)
    ap.add_argument("--champion-value", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    candidate = make_composite(args.candidate_policy, args.candidate_value, device)
    champion = make_composite(args.champion_policy, args.champion_value, device)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    result, inference_seconds = evaluate(
        candidate, champion, device, engine,
        max(1, args.pairs), max(1, args.sims), max(1, args.pieces), max(1, args.batch),
        max(1, args.determinizations), False, args.precision, max(0, args.seed),
    )
    candidate_label = f"P={args.candidate_policy}; V={args.candidate_value}"
    champion_label = f"P={args.champion_policy}; V={args.champion_value}"
    report(
        result, candidate_label, champion_label, inference_seconds,
        max(1, args.pairs), max(1, args.sims), max(1, args.pieces),
        max(1, args.determinizations), False, args.precision, max(0, args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
