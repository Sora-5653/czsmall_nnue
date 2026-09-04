# SPDX-License-Identifier: MIT
"""Simple forward-only latency benchmark for ablation checkpoints."""

from __future__ import annotations

import argparse
import time

import torch

from ablation_models import load_ablation_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=102)
    ap.add_argument("--actions", type=int, default=106)
    ap.add_argument("--features", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device)
    batch = args.batch
    tokens = torch.rand(batch, args.tokens, args.features, device=device)
    token_mask = torch.ones(batch, args.tokens, device=device)
    actions = torch.rand(batch, args.actions, args.features, device=device)
    action_mask = torch.ones(batch, args.actions, device=device)

    for path in args.checkpoints:
        model = load_ablation_checkpoint(path, device)
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(args.warmup):
                model(tokens, token_mask, actions, action_mask)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.reps):
                model(tokens, token_mask, actions, action_mask)
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        ms_per_batch = 1000.0 * elapsed / args.reps
        positions_per_second = args.reps * batch / elapsed
        architecture = getattr(model, "architecture", "transformer")
        print(
            f"{architecture:24s} {ms_per_batch:8.3f} ms/batch  "
            f"{positions_per_second:9.1f} positions/s"
        )
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
