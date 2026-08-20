#!/usr/bin/env python3
"""Distill a larger TetraFormer teacher into the XS local evaluator.

The loss follows the same high-level recipe as MochBot/fusion's public student
trainer: freeze the teacher, train only the student, use temperature-scaled
KL divergence for categorical outputs, and an MSE term for a continuous value.
For TetraFormer the categorical outputs are the variable-length policy and WDL
value distributions; the continuous target is the WDL expected value.

No human hard labels are mixed into the default distillation objective. The
student learns the teacher's softened judgement on the same exact replay states.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The existing trainer is intentionally executable both as `python trainer/...`
# and from Windows ROCm venvs. Its imports are script-local rather than package
# relative, so make that directory explicit here instead of rewriting the whole
# training stack just for the distiller.
_TRAINER_DIR = Path(__file__).resolve().parent
if str(_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINER_DIR))

import tetra_dataset  # noqa: E402
from train import build_model, load_checkpoint_model, split_indices_by_game, to_cpu  # noqa: E402


def resolve_device(requested: str, require_gpu: bool) -> str:
    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            best = max(
                range(torch.cuda.device_count()),
                key=lambda index: torch.cuda.get_device_properties(index).total_memory,
            )
            return f"cuda:{best}"
        if require_gpu:
            raise SystemExit("GPU requested but no CUDA/ROCm device is visible to torch")
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if require_gpu:
            raise SystemExit("GPU requested but no CUDA/ROCm device is visible to torch")
        return "cpu"
    return requested


def masked_temperature_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) over legal variable-length actions.

    TetraFormer masks padding with -inf. Replacing illegal entries before
    log_softmax avoids undefined (-inf)-(-inf) arithmetic while leaving the
    legal distribution exactly unchanged.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    legal = legal_mask > 0.5
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("every distillation row must contain at least one legal action")
    floor = torch.finfo(student_logits.dtype).min
    student_scaled = (student_logits / temperature).masked_fill(~legal, floor)
    teacher_scaled = (teacher_logits / temperature).masked_fill(~legal, floor)
    student_log = F.log_softmax(student_scaled, dim=-1)
    teacher_log = F.log_softmax(teacher_scaled, dim=-1)
    teacher_prob = teacher_log.exp().masked_fill(~legal, 0.0)
    per_row = (teacher_prob * (teacher_log - student_log).masked_fill(~legal, 0.0)).sum(dim=-1)
    return per_row.mean() * (temperature ** 2)


def temperature_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_log = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(
        student_log,
        teacher_log,
        log_target=True,
        reduction="batchmean",
    ) * (temperature ** 2)


def wdl_expected_value(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=-1)
    return probability[:, 0] - probability[:, 2]


def distillation_losses(
    student_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    action_mask: torch.Tensor,
    *,
    temperature: float,
    policy_weight: float,
    value_kl_weight: float,
    value_mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_policy, student_wdl, _student_aux = student_outputs
    teacher_policy, teacher_wdl, _teacher_aux = teacher_outputs
    policy_kl = masked_temperature_kl(
        student_policy, teacher_policy, action_mask, temperature
    )
    value_kl = temperature_kl(student_wdl, teacher_wdl, temperature)
    teacher_value = wdl_expected_value(teacher_wdl).detach()
    student_value = wdl_expected_value(student_wdl)
    value_mse = F.mse_loss(student_value, teacher_value)
    total = (
        policy_weight * policy_kl
        + value_kl_weight * value_kl
        + value_mse_weight * value_mse
    )
    return total, {
        "policy_kl": policy_kl,
        "value_kl": value_kl,
        "value_mse": value_mse,
    }


def save_student(
    path: str,
    student,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    teacher_path: str,
    temperature: float,
    weights: dict[str, float],
    generator: torch.Generator,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format_version": 3,
        "step": step,
        "config": dict(student.cfg.__dict__),
        "state_dict": to_cpu(student.state_dict()),
        "optimizer_state_dict": to_cpu(optimizer.state_dict()),
        "sampling_generator_state": generator.get_state(),
        "distillation": {
            "teacher": teacher_path,
            "temperature": temperature,
            "weights": dict(weights),
            "method": "frozen-teacher-temperature-kl-plus-value-mse",
        },
    }
    torch.save(payload, path)
    print(f"checkpoint     step {step} -> {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", metavar="DATASET")
    ap.add_argument("--teacher", required=True, help="trained teacher checkpoint")
    ap.add_argument("--student", choices=("xs",), default="xs")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-kl-weight", type=float, default=0.5)
    ap.add_argument("--value-mse-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--save", required=True)
    ap.add_argument("--best-save", default="")
    args = ap.parse_args()

    if args.steps < 0 or args.batch <= 0:
        ap.error("steps must be non-negative and batch must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        ap.error("temperature must be finite and positive")
    for name in ("lr", "weight_decay", "policy_weight", "value_kl_weight", "value_mse_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.policy_weight + args.value_kl_weight + args.value_mse_weight <= 0.0:
        ap.error("at least one distillation loss weight must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device, args.require_gpu)
    if device.startswith("cuda"):
        print(f"device        {torch.cuda.get_device_name(torch.device(device))}", flush=True)
    else:
        print(f"device        {device}", flush=True)

    loaded = []
    for path in args.datasets:
        dataset = tetra_dataset.load(path)
        loaded.append(dataset)
        print(f"replay input   {path}: {len(dataset)} samples", flush=True)
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    data = ds.torch()
    train_np, val_np = split_indices_by_game(ds)
    if len(val_np) == 0:
        val_np = np.array(train_np, copy=True)
        print("warning: no independent game-level validation group; reusing train states", flush=True)
    train_idx = torch.as_tensor(train_np, dtype=torch.long)
    val_idx = torch.as_tensor(val_np, dtype=torch.long)
    print(
        f"dataset       {len(ds)} samples; split train/val {len(train_idx)}/{len(val_idx)}; "
        f"max actions {ds.header.max_actions}",
        flush=True,
    )

    teacher = load_checkpoint_model(args.teacher, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    if (
        teacher.cfg.token_features != ds.header.token_features
        or teacher.cfg.action_features != ds.header.action_features
        or teacher.cfg.aux_targets != ds.header.aux_targets
    ):
        raise SystemExit("teacher/dataset feature contract mismatch")

    student = build_model(args.student, ds.header).to(device)
    student.train()
    print(
        f"teacher       {teacher.parameter_count() / 1e6:.3f}M frozen parameters\n"
        f"student       {student.parameter_count() / 1e6:.3f}M trainable parameters\n"
        f"distill       T={args.temperature:g} policy_kl={args.policy_weight:g} "
        f"value_kl={args.value_kl_weight:g} value_mse={args.value_mse_weight:g}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=min(1e-6, args.lr)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed ^ 0xD157111)
    weights = {
        "policy_kl": float(args.policy_weight),
        "value_kl": float(args.value_kl_weight),
        "value_mse": float(args.value_mse_weight),
    }

    def batch_at(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value[indices] for key, value in data.items()}
        if device != "cpu":
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        return batch

    def evaluate() -> dict[str, float]:
        student.eval()
        totals = {"total": 0.0, "policy_kl": 0.0, "value_kl": 0.0, "value_mse": 0.0,
                  "policy_top1_agreement": 0.0}
        count_total = 0
        with torch.no_grad():
            for offset in range(0, len(val_idx), max(1, args.batch)):
                indices = val_idx[offset:offset + max(1, args.batch)]
                batch = batch_at(indices)
                teacher_out = teacher(
                    batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
                )
                student_out = student(
                    batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
                )
                total, parts = distillation_losses(
                    student_out, teacher_out, batch["action_mask"],
                    temperature=args.temperature,
                    policy_weight=args.policy_weight,
                    value_kl_weight=args.value_kl_weight,
                    value_mse_weight=args.value_mse_weight,
                )
                n = len(indices)
                count_total += n
                totals["total"] += float(total.item()) * n
                for name in ("policy_kl", "value_kl", "value_mse"):
                    totals[name] += float(parts[name].item()) * n
                teacher_best = teacher_out[0].argmax(dim=-1)
                student_best = student_out[0].argmax(dim=-1)
                totals["policy_top1_agreement"] += float(
                    (teacher_best == student_best).float().sum().item()
                )
        student.train()
        return {name: value / max(1, count_total) for name, value in totals.items()}

    initial = evaluate()
    print(
        "val (start)   "
        f"total {initial['total']:.5f} policyKL {initial['policy_kl']:.5f} "
        f"valueKL {initial['value_kl']:.5f} valueMSE {initial['value_mse']:.5f} "
        f"top1 {initial['policy_top1_agreement']:.3f}",
        flush=True,
    )
    best_metric = initial["total"]
    best_step = 0
    start = time.perf_counter()

    for local_step in range(1, args.steps + 1):
        batch_n = min(args.batch, len(train_idx))
        picked = train_idx[
            torch.randint(0, len(train_idx), (batch_n,), generator=generator)
        ]
        batch = batch_at(picked)
        with torch.no_grad():
            teacher_out = teacher(
                batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
            )
        student_out = student(
            batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
        )
        total, parts = distillation_losses(
            student_out, teacher_out, batch["action_mask"],
            temperature=args.temperature,
            policy_weight=args.policy_weight,
            value_kl_weight=args.value_kl_weight,
            value_mse_weight=args.value_mse_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if local_step == 1 or local_step == args.steps or local_step % 100 == 0:
            with torch.no_grad():
                top1 = float((teacher_out[0].argmax(-1) == student_out[0].argmax(-1)).float().mean().item())
            print(
                f"step {local_step:6d}  total {float(total.item()):.5f} "
                f"policyKL {float(parts['policy_kl'].item()):.5f} "
                f"valueKL {float(parts['value_kl'].item()):.5f} "
                f"valueMSE {float(parts['value_mse'].item()):.5f} top1 {top1:.3f}",
                flush=True,
            )

        should_eval = args.eval_every > 0 and local_step % args.eval_every == 0
        if local_step == args.steps:
            should_eval = True
        if should_eval:
            metrics = evaluate()
            print(
                f"val step {local_step:6d} total {metrics['total']:.5f} "
                f"policyKL {metrics['policy_kl']:.5f} valueKL {metrics['value_kl']:.5f} "
                f"valueMSE {metrics['value_mse']:.5f} top1 {metrics['policy_top1_agreement']:.3f}",
                flush=True,
            )
            if metrics["total"] < best_metric:
                best_metric = metrics["total"]
                best_step = local_step
                if args.best_save:
                    save_student(
                        args.best_save, student, optimizer, local_step,
                        teacher_path=args.teacher, temperature=args.temperature,
                        weights=weights, generator=generator,
                    )

        if args.checkpoint_every > 0 and local_step % args.checkpoint_every == 0:
            save_student(
                args.save, student, optimizer, local_step,
                teacher_path=args.teacher, temperature=args.temperature,
                weights=weights, generator=generator,
            )

    elapsed = time.perf_counter() - start
    print(
        f"distilled     {args.steps} steps in {elapsed:.1f}s; "
        f"best val {best_metric:.5f} at step {best_step}",
        flush=True,
    )
    save_student(
        args.save, student, optimizer, args.steps,
        teacher_path=args.teacher, temperature=args.temperature,
        weights=weights, generator=generator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
