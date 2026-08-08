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
import math
import time

import torch

import tetra_dataset
from tetraformer import TetraFormer, TetraFormerConfig, losses


DEFAULT_LOSS_WEIGHTS = {
    "policy": 1.0,
    "value": 1.0,
    "aux": 0.1,
}


def build_model(kind: str, header: tetra_dataset.Header) -> TetraFormer:
    if kind == "s":
        cfg = TetraFormerConfig(
            token_features=header.token_features,
            action_features=header.action_features,
            width=256,
            layers=8,
            heads=8,
            ffn=768,
            aux_targets=header.aux_targets,
        )
    else:
        cfg = TetraFormerConfig(
            token_features=header.token_features,
            action_features=header.action_features,
            width=64,
            layers=2,
            heads=4,
            ffn=192,
            aux_targets=header.aux_targets,
        )
    return TetraFormer(cfg)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def save_checkpoint(path: str, model: TetraFormer, optimizer: torch.optim.Optimizer,
                    step: int, sample_generator: torch.Generator,
                    loss_weights: dict[str, float]) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format_version": 3,
        "step": step,
        "config": dict(model.cfg.__dict__),
        "loss_weights": dict(loss_weights),
        "state_dict": to_cpu(model.state_dict()),
        "optimizer_state_dict": to_cpu(optimizer.state_dict()),
        "sampling_generator_state": sample_generator.get_state(),
    }
    torch.save(payload, path)
    print(f"checkpoint     step {step} -> {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", metavar="DATASET",
                    help="one or more replay generations; the last can be repeated")
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
    ap.add_argument("--resume", default="", help="resume model and optimizer state from a .pt checkpoint")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="resume model weights but initialize a fresh AdamW optimizer")
    ap.add_argument("--policy-head-only", action="store_true",
                    help="freeze the shared trunk and value/aux heads; train policy layers only")
    ap.add_argument("--start-step", type=int, default=-1,
                    help="override the stored global step for a legacy checkpoint")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save --save periodically every N global steps (0 disables)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="evaluate validation loss every N local steps (0 disables)")
    ap.add_argument("--best-save", default="",
                    help="save the best validation-loss checkpoint separately")
    ap.add_argument("--policy-weight", type=float, default=None,
                    help="policy loss weight (default: checkpoint value or 1.0)")
    ap.add_argument("--value-weight", type=float, default=None,
                    help="WDL value loss weight (default: checkpoint value or 1.0)")
    ap.add_argument("--aux-weight", type=float, default=None,
                    help="auxiliary loss weight (default: checkpoint value or 0.1)")
    ap.add_argument("--new-data-repeat", type=int, default=1,
                    help="repeat the last dataset this many times when replaying generations")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail instead of falling back to CPU when --device is cuda")
    ap.add_argument("--fail-on-no-improve", action="store_true",
                    help="return failure when held-out loss does not improve")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        if args.require_gpu:
            raise SystemExit("GPU requested but no CUDA/ROCm device is visible to torch")
        print("warning: no GPU visible to torch, falling back to CPU. See docs/SETUP.md.")
        device = "cpu"
    if device.startswith("cuda"):
        print(f"device        {torch.cuda.get_device_name(torch.device(device))}", flush=True)

    if args.new_data_repeat < 1:
        raise SystemExit("--new-data-repeat must be at least 1")
    dataset_paths = list(args.datasets)
    if len(dataset_paths) > 1 and args.new_data_repeat > 1:
        dataset_paths.extend([dataset_paths[-1]] * (args.new_data_repeat - 1))
    loaded_datasets = []
    for path in dataset_paths:
        loaded = tetra_dataset.load(path)
        loaded_datasets.append(loaded)
        print(f"replay input   {path}: {len(loaded)} samples")
    ds = tetra_dataset.Dataset.concatenate(loaded_datasets)
    ds.sanity_check()
    data = ds.torch(device if device != "cpu" else None)
    n = len(ds)
    print(f"dataset       {n} samples from {len(loaded_datasets)} replay inputs, "
          f"{ds.tokens.shape[1]} tokens, {ds.actions.shape[1]} actions")

    checkpoint = None
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, TetraFormer):
            model = checkpoint
        else:
            if "config" not in checkpoint or "state_dict" not in checkpoint:
                raise SystemExit(f"invalid checkpoint: {args.resume}")
            cfg = TetraFormerConfig(**checkpoint["config"])
            if (cfg.token_features != ds.header.token_features or
                    cfg.action_features != ds.header.action_features or
                    cfg.aux_targets != ds.header.aux_targets):
                raise SystemExit(
                    "checkpoint/dataset feature mismatch: "
                    f"checkpoint ({cfg.token_features}, {cfg.action_features}, {cfg.aux_targets}) "
                    f"vs dataset ({ds.header.token_features}, {ds.header.action_features}, "
                    f"{ds.header.aux_targets})"
                )
            model = TetraFormer(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            start_step = int(checkpoint.get("step", 0))
            if args.start_step >= 0:
                start_step = args.start_step
    else:
        model = build_model(args.model, ds.header)
    model.to(device)
    model_label = "checkpoint" if args.resume else args.model
    print(f"model         {model_label} ({model.parameter_count() / 1e6:.2f}M parameters)")
    if args.resume:
        print(f"resume        {args.resume} (starting at step {start_step})")

    stored_weights = checkpoint.get("loss_weights", {}) if isinstance(checkpoint, dict) else {}
    requested_weights = {
        "policy": args.policy_weight,
        "value": args.value_weight,
        "aux": args.aux_weight,
    }
    loss_weights = {}
    for name, default in DEFAULT_LOSS_WEIGHTS.items():
        weight = requested_weights[name]
        if weight is None:
            weight = stored_weights.get(name, default)
        if not math.isfinite(weight) or weight < 0.0:
            raise SystemExit(f"{name} loss weight must be finite and non-negative")
        loss_weights[name] = float(weight)
    if not any(loss_weights.values()):
        raise SystemExit("at least one loss weight must be positive")
    print("loss weights  " + " ".join(f"{name}={weight:g}" for name, weight in loss_weights.items()))

    if args.policy_head_only:
        policy_prefixes = ("action_in.", "policy_attn.", "policy_norm.", "policy_out.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(policy_prefixes)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not args.reset_optimizer:
            raise SystemExit("--policy-head-only requires --reset-optimizer")
        print(f"trainable     {sum(p.numel() for p in trainable):,} policy parameters")
    else:
        trainable = list(model.parameters())

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    if (isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict")
            and not args.reset_optimizer):
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(opt, device)
    model.train()

    # Use a deterministic shuffled holdout.  The exporter appends games in
    # seed order; a contiguous tail would turn validation into a game/seed
    # distribution test and make the WDL metric look worse simply because the
    # last games have a different outcome mix.  A future replay format can do
    # the stronger game-level split once provenance is available to Python.
    split = max(1, int(n * 0.8))
    split_generator = torch.Generator().manual_seed(0x5EED)
    permutation = torch.randperm(n, generator=split_generator)
    train_idx = permutation[:split].to(device)
    val_idx = permutation[split:].to(device)

    def batch_at(idx):
        return {k: v[idx] for k, v in data.items()}

    def evaluate():
        if len(val_idx) == 0:
            return None
        model.eval()
        with torch.no_grad():
            _, parts = losses(model, batch_at(val_idx), weights=loss_weights)
        model.train()
        return parts

    gen = torch.Generator().manual_seed(1)
    if isinstance(checkpoint, dict) and checkpoint.get("sampling_generator_state") is not None:
        gen.set_state(checkpoint["sampling_generator_state"])

    first = evaluate()
    print(f"val (start)   total {first['total']:.4f}  policy {first['policy']:.4f}  "
          f"value {first['value']:.4f}  v_acc {first['value_accuracy']:.3f}  "
          f"v_mse {first['value_scalar_mse']:.4f}")
    best_total = first["total"]
    best_step = start_step
    eval_interval = args.eval_every
    if eval_interval <= 0 and args.best_save and args.checkpoint_every > 0:
        eval_interval = args.checkpoint_every
    if args.best_save:
        save_checkpoint(args.best_save, model, opt, best_step, gen, loss_weights)

    t0 = time.time()
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        pick = train_idx[
            torch.randint(0, len(train_idx), (min(args.batch, len(train_idx)),),
                          generator=gen).to(device)
        ]
        total, parts = losses(model, batch_at(pick), weights=loss_weights)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if local_step % max(1, args.steps // 10) == 0:
            print(f"step {step:5d}  total {parts['total']:.4f}  policy {parts['policy']:.4f}  "
                  f"value {parts['value']:.4f}  v_acc {parts['value_accuracy']:.3f}  "
                  f"aux {parts['aux']:.4f}", flush=True)
        if args.checkpoint_every > 0 and args.save and step % args.checkpoint_every == 0:
            save_checkpoint(args.save, model, opt, step, gen, loss_weights)
        if eval_interval > 0 and local_step % eval_interval == 0:
            interim = evaluate()
            print(f"val (step {step}) total {interim['total']:.4f}  "
                  f"policy {interim['policy']:.4f}  value {interim['value']:.4f}  "
                  f"v_acc {interim['value_accuracy']:.3f}  "
                  f"v_mse {interim['value_scalar_mse']:.4f}", flush=True)
            if interim["total"] < best_total:
                best_total = interim["total"]
                best_step = step
                if args.best_save:
                    save_checkpoint(args.best_save, model, opt, best_step, gen, loss_weights)

    secs = time.time() - t0
    last = evaluate()
    print(f"val (end)     total {last['total']:.4f}  policy {last['policy']:.4f}  "
          f"value {last['value']:.4f}  v_acc {last['value_accuracy']:.3f}  "
          f"v_mse {last['value_scalar_mse']:.4f}")
    rate = args.steps / secs if secs > 0.0 else 0.0
    print(f"trained       {args.steps} steps in {secs:.1f}s ({rate:.1f} steps/s)")

    improved = last["total"] < first["total"]
    print(f"held-out loss {'improved' if improved else 'DID NOT improve'}: "
          f"{first['total']:.4f} -> {last['total']:.4f}")

    if args.save:
        save_checkpoint(args.save, model, opt, start_step + args.steps, gen, loss_weights)
    if args.best_save and last["total"] < best_total:
        save_checkpoint(args.best_save, model, opt, start_step + args.steps, gen, loss_weights)

    return 0 if improved or not args.fail_on_no_improve else 1


if __name__ == "__main__":
    raise SystemExit(main())
