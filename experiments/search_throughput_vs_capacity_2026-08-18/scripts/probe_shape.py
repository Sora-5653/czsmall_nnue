#!/usr/bin/env python3
"""Probe one evaluator tensor shape for a ROCm dispatch stall."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "trainer"))
from gpu_match import load_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batches", nargs="+", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=102)
    parser.add_argument("--actions", type=int, default=39)
    parser.add_argument("--valid-tokens", type=int, default=0)
    parser.add_argument("--valid-actions", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(str(Path(args.model).resolve()), device)
    batches = args.batches or [args.batch]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch in batches:
            token = torch.rand(batch, args.tokens, 24, device=device)
            token_mask = torch.ones(batch, args.tokens, device=device)
            action = torch.rand(batch, args.actions, 24, device=device)
            action_mask = torch.ones(batch, args.actions, device=device)
            if args.valid_tokens > 0:
                token_mask[:, args.valid_tokens:] = 0.0
            if args.valid_actions > 0:
                action_mask[:, args.valid_actions:] = 0.0
            for index in range(args.repetitions):
                started = time.perf_counter()
                model(token, token_mask, action, action_mask)
                torch.cuda.synchronize(device)
                print({"batch": batch, "iteration": index,
                       "elapsed_ms": (time.perf_counter() - started) * 1000.0},
                      flush=True)
            del token, token_mask, action, action_mask
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
