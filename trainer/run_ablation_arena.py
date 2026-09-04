# SPDX-License-Identifier: MIT
"""Arena helper that can load Transformer, CNN, and hybrid ablation checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ablation_models import load_ablation_checkpoint
from gpu_arena import evaluate, report


class ValueTemperatureWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, tokens, token_mask, actions, action_mask):
        policy, value, aux = self.model(tokens, token_mask, actions, action_mask)
        return policy, value / self.temperature, aux


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate")
    ap.add_argument("champion")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--candidate-sims", type=int, default=-1,
                    help="diagnostic override; 0 means policy-only")
    ap.add_argument("--champion-sims", type=int, default=-1,
                    help="diagnostic override; 0 means policy-only")
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--gumbel", action="store_true")
    ap.add_argument("--candidate-value-temperature", type=float, default=1.0)
    ap.add_argument("--champion-value-temperature", type=float, default=1.0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    candidate = load_ablation_checkpoint(args.candidate, device)
    champion = load_ablation_checkpoint(args.champion, device)
    if args.candidate_value_temperature <= 0 or args.champion_value_temperature <= 0:
        raise SystemExit("value temperature must be > 0")
    if args.candidate_value_temperature != 1.0:
        candidate = ValueTemperatureWrapper(candidate, args.candidate_value_temperature).to(device)
        candidate.eval()
    if args.champion_value_temperature != 1.0:
        champion = ValueTemperatureWrapper(champion, args.champion_value_temperature).to(device)
        champion.eval()

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    result, inference_seconds = evaluate(
        candidate,
        champion,
        device,
        engine,
        max(1, args.pairs),
        max(1, args.sims),
        max(1, args.pieces),
        max(1, args.batch),
        max(1, args.determinizations),
        args.gumbel,
        args.precision,
        max(0, args.seed),
        args.candidate_sims,
        args.champion_sims,
    )
    report(
        result,
        args.candidate,
        args.champion,
        inference_seconds,
        max(1, args.pairs),
        max(1, args.sims),
        max(1, args.pieces),
        max(1, args.determinizations),
        args.gumbel,
        args.precision,
        max(0, args.seed),
        args.candidate_sims,
        args.champion_sims,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
