# SPDX-License-Identifier: MIT
"""Train a TetraFormer on a `.tetradat` set exported by the C++ engine.

This is deliberately a small, readable loop rather than a distributed trainer:
the point at this stage is to prove the handover works end to end and that the
loss actually decreases, not to reach strength. Spec section 13 describes the
full pipeline; scaling this up belongs on a GPU.

Usage::

    python trainer/train.py data.tetradat --steps 200 --model dev
"""

from __future__ import annotations

import argparse
import time

import torch

import tetra_dataset
from tetraformer import losses, tetraformer_dev, tetraformer_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model", choices=("dev", "s"), default="dev")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu, or cuda for a GPU. ROCm reports AMD cards through the cuda API, "
        "so an RX 9070 XT is also --device cuda.",
    )
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: no GPU visible to torch, falling back to CPU. See docs/SETUP.md.")
        device = "cpu"
    if device.startswith("cuda"):
        print(f"device        {torch.cuda.get_device_name(0)}")

    ds = tetra_dataset.load(args.dataset)
    ds.sanity_check()
    data = ds.torch(device if device != "cpu" else None)
    n = len(ds)
    print(f"dataset       {n} samples, {ds.tokens.shape[1]} tokens, {ds.actions.shape[1]} actions")

    model = tetraformer_s() if args.model == "s" else tetraformer_dev()
    model.to(device)
    print(f"model         {args.model} ({model.parameter_count() / 1e6:.2f}M parameters)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()

    # A held-out slice, so an improving loss cannot just be memorisation.
    split = max(1, int(n * 0.8))
    train_idx = torch.arange(0, split, device=device)
    val_idx = torch.arange(split, n, device=device)

    def batch_at(idx):
        return {k: v[idx] for k, v in data.items()}

    def evaluate():
        if len(val_idx) == 0:
            return None
        model.eval()
        with torch.no_grad():
            _, parts = losses(model, batch_at(val_idx))
        model.train()
        return parts

    first = evaluate()
    print(f"val (start)   total {first['total']:.4f}  policy {first['policy']:.4f}  "
          f"value {first['value']:.4f}")

    t0 = time.time()
    gen = torch.Generator().manual_seed(1)
    for step in range(1, args.steps + 1):
        pick = train_idx[
            torch.randint(0, len(train_idx), (min(args.batch, len(train_idx)),),
                          generator=gen).to(device)
        ]
        total, parts = losses(model, batch_at(pick))
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, args.steps // 10) == 0:
            print(f"step {step:5d}  total {parts['total']:.4f}  policy {parts['policy']:.4f}  "
                  f"value {parts['value']:.4f}  aux {parts['aux']:.4f}")

    secs = time.time() - t0
    last = evaluate()
    print(f"val (end)     total {last['total']:.4f}  policy {last['policy']:.4f}  "
          f"value {last['value']:.4f}")
    print(f"trained       {args.steps} steps in {secs:.1f}s ({args.steps / secs:.1f} steps/s)")

    improved = last["total"] < first["total"]
    print(f"held-out loss {'improved' if improved else 'DID NOT improve'}: "
          f"{first['total']:.4f} -> {last['total']:.4f}")

    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        model.to("cpu")
        torch.save({"config": model.cfg.__dict__, "state_dict": model.state_dict()}, args.save)
        print(f"saved         {args.save}")

    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())