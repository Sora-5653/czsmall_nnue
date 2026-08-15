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

import numpy as np
import torch

import tetra_dataset
from tetraformer import TetraFormer, TetraFormerConfig, losses


DEFAULT_LOSS_WEIGHTS = {
    "policy": 1.0,
    "value": 1.0,
    "aux": 0.1,
}


def load_checkpoint_model(path: str, device: str) -> TetraFormer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, TetraFormer):
        model = checkpoint
    else:
        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise SystemExit(f"invalid checkpoint: {path}")
        model = TetraFormer(TetraFormerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


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


def split_indices_by_game(
    ds: tetra_dataset.Dataset,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by game seed so adjacent trajectory states never leak across sets.

    If ``indices`` is supplied, split only that source subset and return absolute
    dataset indices. This prevents a source with a disjoint/high seed range from
    being assigned entirely to validation when replay sources are mixed.
    """
    if indices is None:
        absolute = np.arange(len(ds), dtype=np.int64)
    else:
        absolute = np.asarray(indices, dtype=np.int64)
    seeds = np.asarray(ds.game_seed[absolute], dtype=np.uint64)
    unique = sorted({int(seed) for seed in seeds.tolist() if int(seed) != 0})
    if len(unique) >= 2:
        cut = max(1, min(len(unique) - 1, int(len(unique) * 0.8)))
        train_seeds = set(unique[:cut])
        train = np.asarray([int(seed) in train_seeds for seed in seeds], dtype=bool)
        return absolute[train], absolute[~train]

    # Legacy v1 files do not carry game provenance. Keep the old deterministic
    # fallback, but apply it within the selected source subset.
    generator = np.random.default_rng(0x5EED)
    permutation = generator.permutation(len(absolute))
    split = max(1, int(len(absolute) * 0.8))
    if split >= len(absolute) and len(absolute) > 1:
        split = len(absolute) - 1
    return absolute[permutation[:split]], absolute[permutation[split:]]


def gradient_diagnostics(model: TetraFormer, loss_tensors, weights: dict[str, float]) -> dict[str, float]:
    """Measure how each objective pulls on the shared representation."""
    shared = [
        parameter for name, parameter in model.named_parameters()
        if (name.startswith("token_in.") or name.startswith("blocks.") or
            name.startswith("norm.")) and parameter.requires_grad
    ]
    if not shared:
        return {"policy_grad": 0.0, "value_grad": 0.0, "aux_grad": 0.0,
                "vs_aux_grad": 0.0, "cancellation_aux_grad": 0.0,
                "timing_pair_grad": 0.0, "timing_rank_grad": 0.0,
                "policy_value_cos": 0.0, "policy_aux_cos": 0.0,
                "policy_vs_aux_cos": 0.0, "policy_cancellation_aux_cos": 0.0,
                "policy_timing_pair_cos": 0.0, "policy_timing_rank_cos": 0.0}

    gradients = []
    for loss, name in zip(
        loss_tensors,
        ("policy", "value", "aux", "vs_aux", "cancellation_aux", "timing_pair", "timing_rank")
    ):
        grads = torch.autograd.grad(
            loss * weights[name], shared, retain_graph=True, allow_unused=True
        )
        gradients.append([g if g is not None else torch.zeros_like(p)
                          for g, p in zip(grads, shared)])

    def norm(values):
        return torch.sqrt(sum(g.pow(2).sum() for g in values)).item()

    def cosine(left, right):
        dot = sum((a * b).sum() for a, b in zip(left, right))
        denominator = torch.sqrt(sum(a.pow(2).sum() for a in left)) * \
            torch.sqrt(sum(b.pow(2).sum() for b in right))
        if denominator.item() <= 1e-12:
            return 0.0
        return (dot / denominator).item()

    return {
        "policy_grad": norm(gradients[0]),
        "value_grad": norm(gradients[1]),
        "aux_grad": norm(gradients[2]),
        "vs_aux_grad": norm(gradients[3]),
        "cancellation_aux_grad": norm(gradients[4]),
        "timing_pair_grad": norm(gradients[5]),
        "timing_rank_grad": norm(gradients[6]),
        "policy_value_cos": cosine(gradients[0], gradients[1]),
        "policy_aux_cos": cosine(gradients[0], gradients[2]),
        "policy_vs_aux_cos": cosine(gradients[0], gradients[3]),
        "policy_cancellation_aux_cos": cosine(gradients[0], gradients[4]),
        "policy_timing_pair_cos": cosine(gradients[0], gradients[5]),
        "policy_timing_rank_cos": cosine(gradients[0], gradients[6]),
    }


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
    ap.add_argument("--seed", type=int, default=0,
                    help="training/model seed; keep fixed for paired ablations")
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
    ap.add_argument("--reset-sampling", action="store_true",
                    help="ignore checkpoint sampling RNG state and restart sampling from --seed")
    head_group = ap.add_mutually_exclusive_group()
    head_group.add_argument("--policy-head-only", action="store_true",
                            help="freeze the shared trunk and value/aux heads; train policy layers only")
    head_group.add_argument("--value-head-only", action="store_true",
                            help="freeze the shared trunk and policy/aux heads; train WDL value branch only")
    head_group.add_argument("--aux-head-only", action="store_true",
                            help="freeze the shared trunk and policy/value heads; train auxiliary branch only")
    ap.add_argument("--upgrade-value-attention", action="store_true",
                    help="when resuming a legacy checkpoint, add a learned attention value pooler")
    ap.add_argument("--upgrade-aux-schema", action="store_true",
                    help="widen a resumed auxiliary head to the dataset schema, preserving old outputs")
    ap.add_argument("--upgrade-topout-attention", action="store_true",
                    help="add a self/opponent attention readout for real-time top-out auxiliary logits")
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
    ap.add_argument("--chosen-action-weight", type=float, default=None,
                    help="cross-entropy weight for the action search actually executed (default: checkpoint value or 0)")
    ap.add_argument("--chosen-disagreement-weight", type=float, default=None,
                    help="cross-entropy on search-chosen actions only where they disagree with the current policy argmax")
    ap.add_argument("--policy-rank-weight", type=float, default=None,
                    help="confidence-weighted teacher top-action ranking loss (default: checkpoint value or 0)")
    ap.add_argument("--policy-target-temperature", type=float, default=None,
                    help="temperature applied to the teacher visit distribution for policy CE; below 1 sharpens it (default: checkpoint value or 1)")
    ap.add_argument("--policy-pair-rank-weight", type=float, default=None,
                    help="confidence-weighted teacher best-vs-second pairwise ranking loss (default: checkpoint value or 0)")
    ap.add_argument("--value-weight", type=float, default=None,
                    help="WDL value loss weight (default: checkpoint value or 1.0)")
    ap.add_argument("--aux-weight", type=float, default=None,
                    help="auxiliary loss weight (default: checkpoint value or 0.1)")
    ap.add_argument("--vs-aux-weight", type=float, default=None,
                    help="cumulative 1/2/4/8s VS/100 auxiliary loss weight (default: checkpoint value or 0)")
    ap.add_argument("--cancellation-aux-weight", type=float, default=None,
                    help="extra MSE weight for schema-v4 garbage-cancellation channels (default: checkpoint value or 0)")
    ap.add_argument("--timing-pair-weight", type=float, default=None,
                    help="FASTEST/WAIT_FOR_EVENT within-placement soft pair loss weight (default: checkpoint value or 0)")
    ap.add_argument("--timing-rank-weight", type=float, default=None,
                    help="FASTEST/WAIT_FOR_EVENT within-placement hard ranking loss weight (default: checkpoint value or 0)")
    timing_factor = ap.add_mutually_exclusive_group()
    timing_factor.add_argument("--factor-timing-policy", dest="factor_timing_policy",
                               action="store_true",
                               help="preserve base-placement prior mass across FASTEST/WAIT variants")
    timing_factor.add_argument("--no-factor-timing-policy", dest="factor_timing_policy",
                               action="store_false",
                               help="disable FASTEST/WAIT policy factorization")
    ap.set_defaults(factor_timing_policy=None)
    ap.add_argument("--timing-wait-bias", type=float, default=None,
                    help="WAIT logit bias inside factorized timing pairs (default: checkpoint/config value)")
    ap.add_argument("--topout-aux-weight", type=float, default=None,
                    help="BCE weight for real-time self/opponent top-out aux channels (default: checkpoint value or 0)")
    ap.add_argument("--new-data-repeat", type=int, default=1,
                    help="repeat the last dataset this many times when replaying generations")
    ap.add_argument("--secondary-source-count", type=int, default=0,
                    help="treat the final N dataset inputs as a distinct replay source (0 disables)")
    ap.add_argument("--secondary-source-fraction", type=float, default=-1.0,
                    help="when secondary-source sampling is enabled, draw this fraction of every training batch from it")
    ap.add_argument("--anchor-primary-policy", action="store_true",
                    help="replace primary-source training policy targets with the resumed model's initial policy distribution")
    ap.add_argument("--secondary-policy-teacher", default="",
                    help="replace secondary-source training policy targets with this checkpoint's policy distribution")
    ap.add_argument("--secondary-elite-app-threshold", type=float, default=-1.0,
                    help="sample only secondary-source player trajectories at/above this reconstructed APP; negative disables")
    ap.add_argument("--secondary-elite-min-samples", type=int, default=0,
                    help="minimum valid placement samples for a secondary elite trajectory")
    ap.add_argument("--elite-app-threshold", type=float, default=-1.0,
                    help="repeat player-trajectories whose reconstructed sent-APP meets this threshold; negative disables")
    ap.add_argument("--elite-app-repeat", type=int, default=1,
                    help="total sampling multiplicity for trajectories selected by --elite-app-threshold")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail instead of falling back to CPU when --device is cuda")
    ap.add_argument("--fail-on-no-improve", action="store_true",
                    help="return failure when held-out loss does not improve")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    if args.upgrade_value_attention and not args.resume:
        raise SystemExit("--upgrade-value-attention requires --resume")
    if args.upgrade_value_attention and not args.value_head_only:
        raise SystemExit("--upgrade-value-attention requires --value-head-only for a safe migration")
    if args.upgrade_aux_schema and not args.resume:
        raise SystemExit("--upgrade-aux-schema requires --resume")
    if args.upgrade_aux_schema and not args.reset_optimizer:
        raise SystemExit("--upgrade-aux-schema requires --reset-optimizer")
    if args.upgrade_aux_schema and args.upgrade_value_attention:
        raise SystemExit("perform aux-schema and value-attention migrations in separate runs")
    if args.upgrade_topout_attention and not args.resume:
        raise SystemExit("--upgrade-topout-attention requires --resume")
    if args.upgrade_topout_attention and not args.aux_head_only:
        raise SystemExit("--upgrade-topout-attention requires --aux-head-only for a safe migration")
    if args.upgrade_topout_attention and not args.reset_optimizer:
        raise SystemExit("--upgrade-topout-attention requires --reset-optimizer")
    if args.upgrade_topout_attention and (args.upgrade_aux_schema or args.upgrade_value_attention):
        raise SystemExit("perform topout-attention migration separately from other migrations")

    torch.manual_seed(args.seed)
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
    if args.secondary_source_count < 0:
        raise SystemExit("--secondary-source-count must be non-negative")
    if not math.isfinite(args.secondary_source_fraction):
        raise SystemExit("--secondary-source-fraction must be finite")
    source_aware_sampling = args.secondary_source_count > 0
    if source_aware_sampling:
        if args.secondary_source_count >= len(args.datasets):
            raise SystemExit("--secondary-source-count must leave at least one primary dataset")
        if not 0.0 < args.secondary_source_fraction < 1.0:
            raise SystemExit("--secondary-source-fraction must be in (0, 1) when enabled")
        if args.batch < 2:
            raise SystemExit("source-aware sampling requires --batch at least 2")
    elif args.secondary_source_fraction >= 0.0:
        raise SystemExit("--secondary-source-fraction requires --secondary-source-count")
    if args.anchor_primary_policy:
        if not source_aware_sampling:
            raise SystemExit("--anchor-primary-policy requires source-aware sampling")
        if not args.resume:
            raise SystemExit("--anchor-primary-policy requires --resume")
        if not args.policy_head_only:
            raise SystemExit("--anchor-primary-policy currently requires --policy-head-only")
    if args.secondary_policy_teacher:
        if not source_aware_sampling:
            raise SystemExit("--secondary-policy-teacher requires source-aware sampling")
        if not args.policy_head_only:
            raise SystemExit("--secondary-policy-teacher currently requires --policy-head-only")
    if not math.isfinite(args.secondary_elite_app_threshold):
        raise SystemExit("--secondary-elite-app-threshold must be finite")
    if args.secondary_elite_min_samples < 0:
        raise SystemExit("--secondary-elite-min-samples must be non-negative")
    if args.secondary_elite_app_threshold >= 0.0 and not source_aware_sampling:
        raise SystemExit("--secondary-elite-app-threshold requires source-aware sampling")
    if args.elite_app_repeat < 1:
        raise SystemExit("--elite-app-repeat must be at least 1")
    if not math.isfinite(args.elite_app_threshold):
        raise SystemExit("--elite-app-threshold must be finite")
    loaded_datasets = []
    for path in args.datasets:
        loaded = tetra_dataset.load(path)
        loaded_datasets.append(loaded)
        print(f"replay input   {path}: {len(loaded)} samples")
    ds = tetra_dataset.Dataset.concatenate(loaded_datasets)
    ds.sanity_check()
    secondary_start = None
    if source_aware_sampling:
        secondary_start = sum(len(item) for item in loaded_datasets[:-args.secondary_source_count])
    # Keep the replay tensors on CPU. Training and validation move only their
    # current bounded batch to the accelerator, so VRAM scales with batch size
    # rather than with the number of replay generations.
    data = ds.torch()
    n = len(ds)
    print(f"dataset       {n} samples from {len(loaded_datasets)} replay inputs, "
          f"{ds.tokens.shape[1]} tokens, {ds.actions.shape[1]} actions")
    stats = ds.target_statistics()
    print(f"aux schema    v{ds.header.aux_target_schema_version} / {ds.header.aux_targets} targets")
    print(f"aux valid     mean {stats['valid_rate'].mean():.3f}  "
          f"zero mean {stats['zero_rate'].mean():.3f}")

    checkpoint = None
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, TetraFormer):
            if args.upgrade_value_attention:
                raise SystemExit("value-attention migration requires a dictionary checkpoint")
            model = checkpoint
        else:
            if "config" not in checkpoint or "state_dict" not in checkpoint:
                raise SystemExit(f"invalid checkpoint: {args.resume}")
            cfg = TetraFormerConfig(**checkpoint["config"])
            aux_mismatch = cfg.aux_targets != ds.header.aux_targets
            if (cfg.token_features != ds.header.token_features or
                    cfg.action_features != ds.header.action_features or
                    (aux_mismatch and not args.upgrade_aux_schema)):
                raise SystemExit(
                    "checkpoint/dataset feature mismatch: "
                    f"checkpoint ({cfg.token_features}, {cfg.action_features}, {cfg.aux_targets}) "
                    f"vs dataset ({ds.header.token_features}, {ds.header.action_features}, "
                    f"{ds.header.aux_targets})"
                )
            if args.upgrade_aux_schema:
                if not aux_mismatch:
                    print("migration     auxiliary schema already matches dataset")
                    model = TetraFormer(cfg)
                    model.load_state_dict(checkpoint["state_dict"])
                elif ds.header.aux_targets < cfg.aux_targets:
                    raise SystemExit("--upgrade-aux-schema only supports widening the auxiliary head")
                else:
                    old_aux_targets = cfg.aux_targets
                    cfg.aux_targets = ds.header.aux_targets
                    model = TetraFormer(cfg)
                    migrated_state = dict(checkpoint["state_dict"])
                    old_aux_weight = migrated_state.pop("aux_head.2.weight", None)
                    old_aux_bias = migrated_state.pop("aux_head.2.bias", None)
                    incompatible = model.load_state_dict(migrated_state, strict=False)
                    allowed_missing = {"aux_head.2.weight", "aux_head.2.bias"}
                    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
                        raise SystemExit(
                            "unsafe aux-schema migration: missing="
                            f"{incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                        )
                    if old_aux_weight is None or old_aux_bias is None:
                        raise SystemExit("checkpoint is missing the auxiliary output layer")
                    with torch.no_grad():
                        model.aux_head[2].weight[:old_aux_targets].copy_(old_aux_weight)
                        model.aux_head[2].bias[:old_aux_targets].copy_(old_aux_bias)
                    print(
                        f"migration     widened auxiliary head {old_aux_targets} -> "
                        f"{ds.header.aux_targets}; preserved existing outputs"
                    )
            elif args.upgrade_value_attention and not cfg.value_attention:
                cfg.value_attention = True
                model = TetraFormer(cfg)
                incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
                allowed_missing = (
                    "value_query", "value_attn.", "value_norm."
                )
                bad_missing = [
                    key for key in incompatible.missing_keys
                    if not any(key == prefix or key.startswith(prefix)
                               for prefix in allowed_missing)
                ]
                if bad_missing or incompatible.unexpected_keys:
                    raise SystemExit(
                        "unsafe value-attention migration: missing="
                        f"{bad_missing}, unexpected={incompatible.unexpected_keys}"
                    )
                print("migration     added learned attention value pooler")
            elif args.upgrade_topout_attention and not cfg.topout_attention:
                cfg.topout_attention = True
                model = TetraFormer(cfg)
                incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
                allowed_missing = (
                    "topout_query", "topout_attn.", "topout_norm.", "topout_out."
                )
                bad_missing = [
                    key for key in incompatible.missing_keys
                    if not any(key == prefix or key.startswith(prefix)
                               for prefix in allowed_missing)
                ]
                if bad_missing or incompatible.unexpected_keys:
                    raise SystemExit(
                        "unsafe topout-attention migration: missing="
                        f"{bad_missing}, unexpected={incompatible.unexpected_keys}"
                    )
                print("migration     added self/opponent tactical topout attention readout")
            else:
                model = TetraFormer(cfg)
                model.load_state_dict(checkpoint["state_dict"])
            start_step = int(checkpoint.get("step", 0))
    else:
        model = build_model(args.model, ds.header)
    if args.resume and args.start_step >= 0:
        start_step = args.start_step
    if args.factor_timing_policy is not None:
        if not hasattr(model.cfg, "factor_timing_policy"):
            raise SystemExit("checkpoint architecture does not support timing factorization")
        model.cfg.factor_timing_policy = bool(args.factor_timing_policy)
    if args.timing_wait_bias is not None:
        if not hasattr(model.cfg, "timing_wait_logit_bias"):
            raise SystemExit("checkpoint architecture does not support timing WAIT bias")
        model.cfg.timing_wait_logit_bias = float(args.timing_wait_bias)
    if (abs(float(getattr(model.cfg, "timing_wait_logit_bias", 0.0))) > 1e-12 and
            not bool(getattr(model.cfg, "factor_timing_policy", False))):
        raise SystemExit("nonzero timing WAIT bias requires factor_timing_policy")
    model.to(device)
    model_label = "checkpoint" if args.resume else args.model
    print(f"model         {model_label} ({model.parameter_count() / 1e6:.2f}M parameters)")
    print(f"timing policy factor={bool(getattr(model.cfg, 'factor_timing_policy', False))} "
          f"wait_bias={float(getattr(model.cfg, 'timing_wait_logit_bias', 0.0)):g}")
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
    chosen_action_weight = args.chosen_action_weight
    if chosen_action_weight is None:
        chosen_action_weight = stored_weights.get("chosen_action", 0.0)
    if not math.isfinite(chosen_action_weight) or chosen_action_weight < 0.0:
        raise SystemExit("chosen action loss weight must be finite and non-negative")
    loss_weights["chosen_action"] = float(chosen_action_weight)
    chosen_disagreement_weight = args.chosen_disagreement_weight
    if chosen_disagreement_weight is None:
        chosen_disagreement_weight = stored_weights.get("chosen_disagreement", 0.0)
    if not math.isfinite(chosen_disagreement_weight) or chosen_disagreement_weight < 0.0:
        raise SystemExit("chosen disagreement loss weight must be finite and non-negative")
    loss_weights["chosen_disagreement"] = float(chosen_disagreement_weight)
    policy_rank_weight = args.policy_rank_weight
    if policy_rank_weight is None:
        policy_rank_weight = stored_weights.get("policy_rank", 0.0)
    if not math.isfinite(policy_rank_weight) or policy_rank_weight < 0.0:
        raise SystemExit("policy rank loss weight must be finite and non-negative")
    loss_weights["policy_rank"] = float(policy_rank_weight)
    policy_target_temperature = args.policy_target_temperature
    if policy_target_temperature is None:
        policy_target_temperature = stored_weights.get("policy_target_temperature", 1.0)
    if not math.isfinite(policy_target_temperature) or policy_target_temperature <= 0.0:
        raise SystemExit("policy target temperature must be finite and positive")
    loss_weights["policy_target_temperature"] = float(policy_target_temperature)
    policy_pair_rank_weight = args.policy_pair_rank_weight
    if policy_pair_rank_weight is None:
        policy_pair_rank_weight = stored_weights.get("policy_pair_rank", 0.0)
    if not math.isfinite(policy_pair_rank_weight) or policy_pair_rank_weight < 0.0:
        raise SystemExit("policy pair rank loss weight must be finite and non-negative")
    loss_weights["policy_pair_rank"] = float(policy_pair_rank_weight)
    vs_aux_weight = args.vs_aux_weight
    if vs_aux_weight is None:
        vs_aux_weight = stored_weights.get("vs_aux", 0.0)
    if not math.isfinite(vs_aux_weight) or vs_aux_weight < 0.0:
        raise SystemExit("VS aux loss weight must be finite and non-negative")
    loss_weights["vs_aux"] = float(vs_aux_weight)
    cancellation_aux_weight = args.cancellation_aux_weight
    if cancellation_aux_weight is None:
        cancellation_aux_weight = stored_weights.get("cancellation_aux", 0.0)
    if not math.isfinite(cancellation_aux_weight) or cancellation_aux_weight < 0.0:
        raise SystemExit("cancellation aux loss weight must be finite and non-negative")
    loss_weights["cancellation_aux"] = float(cancellation_aux_weight)
    timing_pair_weight = args.timing_pair_weight
    if timing_pair_weight is None:
        timing_pair_weight = stored_weights.get("timing_pair", 0.0)
    if not math.isfinite(timing_pair_weight) or timing_pair_weight < 0.0:
        raise SystemExit("timing pair loss weight must be finite and non-negative")
    loss_weights["timing_pair"] = float(timing_pair_weight)
    timing_rank_weight = args.timing_rank_weight
    if timing_rank_weight is None:
        timing_rank_weight = stored_weights.get("timing_rank", 0.0)
    if not math.isfinite(timing_rank_weight) or timing_rank_weight < 0.0:
        raise SystemExit("timing rank loss weight must be finite and non-negative")
    loss_weights["timing_rank"] = float(timing_rank_weight)
    topout_aux_weight = args.topout_aux_weight
    if topout_aux_weight is None:
        topout_aux_weight = stored_weights.get("topout_aux", 0.0)
    if not math.isfinite(topout_aux_weight) or topout_aux_weight < 0.0:
        raise SystemExit("topout aux loss weight must be finite and non-negative")
    loss_weights["topout_aux"] = float(topout_aux_weight)
    optimised_weights = (
        loss_weights["policy"], loss_weights["value"], loss_weights["aux"],
        loss_weights["chosen_action"], loss_weights["chosen_disagreement"],
        loss_weights["policy_rank"],
        loss_weights["policy_pair_rank"], loss_weights["vs_aux"],
        loss_weights["cancellation_aux"], loss_weights["timing_pair"],
        loss_weights["timing_rank"], loss_weights["topout_aux"],
    )
    if not any(optimised_weights):
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
    elif args.value_head_only:
        value_prefixes = ("value_head.", "value_attn.", "value_norm.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name == "value_query" or name.startswith(value_prefixes)
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not args.reset_optimizer:
            raise SystemExit("--value-head-only requires --reset-optimizer")
        print(f"trainable     {sum(p.numel() for p in trainable):,} value parameters")
    elif args.aux_head_only:
        tactical_prefixes = ("aux_head.", "topout_attn.", "topout_norm.", "topout_out.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name == "topout_query" or name.startswith(tactical_prefixes)
            )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not args.reset_optimizer:
            raise SystemExit("--aux-head-only requires --reset-optimizer")
        print(f"trainable     {sum(p.numel() for p in trainable):,} auxiliary parameters")
    else:
        trainable = list(model.parameters())

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    if (isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict")
            and not args.reset_optimizer):
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(opt, device)
    model.train()
    shared_gradient_model = model

    if source_aware_sampling:
        assert secondary_start is not None
        primary_range = np.arange(0, secondary_start, dtype=np.int64)
        secondary_range = np.arange(secondary_start, len(ds), dtype=np.int64)
        primary_train_np, primary_val_np = split_indices_by_game(ds, primary_range)
        secondary_train_np, secondary_val_np = split_indices_by_game(ds, secondary_range)
        train_np = np.concatenate((primary_train_np, secondary_train_np))
        val_np = np.concatenate((primary_val_np, secondary_val_np))
        print(
            f"source split   primary train/val {len(primary_train_np)}/{len(primary_val_np)}; "
            f"secondary train/val {len(secondary_train_np)}/{len(secondary_val_np)}"
        )
    else:
        train_np, val_np = split_indices_by_game(ds)
    raw_train_samples = len(train_np)
    base_train_np = np.array(train_np, copy=True)
    if len(loaded_datasets) > 1 and args.new_data_repeat > 1:
        newest_start = sum(len(item) for item in loaded_datasets[:-1])
        newest_train = train_np[train_np >= newest_start]
        if len(newest_train) > 0:
            train_np = np.concatenate(
                [train_np] + [newest_train] * (args.new_data_repeat - 1)
            )
            print(f"replay weight  newest train samples x{args.new_data_repeat} "
                  f"({len(newest_train)} unique)")
    if args.elite_app_threshold >= 0.0 and args.elite_app_repeat > 1:
        # The placement-0..1 attack target is the net attack actually sent by
        # this player's immediately following placement. Count targets are
        # stored as y = raw / (raw + 8), so invert that squash to reconstruct
        # TETR.IO-style sent APP per (game seed, player perspective). This is a
        # sampling curriculum only: rewards, labels, and loss definitions stay
        # unchanged, and non-elite trajectories remain in the replay window.
        if ds.header.aux_targets <= 20:
            raise SystemExit("elite APP replay requires interval auxiliary targets")
        attack_y = np.asarray(ds.aux_target[:, 20], dtype=np.float64)
        attack_valid = np.asarray(ds.aux_valid_mask[:, 20] > 0.5, dtype=bool)
        attack_y = np.clip(attack_y, 0.0, 1.0 - 1e-9)
        attack_raw = 8.0 * attack_y / (1.0 - attack_y)
        trajectory_app: dict[tuple[int, int], float] = {}
        keys = list(zip(np.asarray(ds.game_seed, dtype=np.uint64).tolist(),
                        np.asarray(ds.player_perspective, dtype=np.int32).tolist()))
        grouped: dict[tuple[int, int], list[int]] = {}
        for i, key in enumerate(keys):
            if int(key[0]) == 0 or not attack_valid[i]:
                continue
            grouped.setdefault((int(key[0]), int(key[1])), []).append(i)
        for key, indices in grouped.items():
            if indices:
                trajectory_app[key] = float(attack_raw[indices].sum() / len(indices))
        elite_keys = {key for key, app in trajectory_app.items()
                      if app >= args.elite_app_threshold}
        elite_np = np.asarray(
            [i for i in base_train_np.tolist()
             if (int(ds.game_seed[i]), int(ds.player_perspective[i])) in elite_keys],
            dtype=np.int64,
        )
        if len(elite_np) > 0:
            train_np = np.concatenate(
                [train_np] + [elite_np] * (args.elite_app_repeat - 1)
            )
        selected_apps = [trajectory_app[key] for key in elite_keys]
        print(
            f"elite replay   APP>={args.elite_app_threshold:g} x{args.elite_app_repeat}: "
            f"{len(elite_keys)} player-trajectories / {len(elite_np)} train samples; "
            f"mean selected APP {np.mean(selected_apps) if selected_apps else 0.0:.3f}"
        )

    train_idx = torch.as_tensor(train_np, dtype=torch.long)
    primary_train_idx = None
    secondary_train_idx = None
    if source_aware_sampling:
        assert secondary_start is not None
        primary_np = train_np[train_np < secondary_start]
        secondary_np = train_np[train_np >= secondary_start]
        if args.secondary_elite_app_threshold >= 0.0:
            if ds.header.aux_targets <= 20:
                raise SystemExit("secondary elite APP filtering requires interval auxiliary targets")
            attack_y = np.asarray(ds.aux_target[:, 20], dtype=np.float64)
            attack_valid = np.asarray(ds.aux_valid_mask[:, 20] > 0.5, dtype=bool)
            attack_y = np.clip(attack_y, 0.0, 1.0 - 1e-9)
            attack_raw = 8.0 * attack_y / (1.0 - attack_y)
            grouped_secondary: dict[tuple[int, int], list[int]] = {}
            for i in secondary_np.tolist():
                if not attack_valid[i]:
                    continue
                key = (int(ds.game_seed[i]), int(ds.player_perspective[i]))
                if key[0] == 0:
                    continue
                grouped_secondary.setdefault(key, []).append(i)
            secondary_elite_keys: set[tuple[int, int]] = set()
            secondary_elite_apps: list[float] = []
            for key, indices in grouped_secondary.items():
                if len(indices) < args.secondary_elite_min_samples:
                    continue
                app = float(attack_raw[indices].sum() / len(indices))
                if app >= args.secondary_elite_app_threshold:
                    secondary_elite_keys.add(key)
                    secondary_elite_apps.append(app)
            secondary_np = np.asarray(
                [i for i in secondary_np.tolist()
                 if (int(ds.game_seed[i]), int(ds.player_perspective[i])) in secondary_elite_keys],
                dtype=np.int64,
            )
            print(
                f"secondary elite APP>={args.secondary_elite_app_threshold:g}, "
                f"samples>={args.secondary_elite_min_samples}: "
                f"{len(secondary_elite_keys)} trajectories / {len(secondary_np)} train samples; "
                f"mean APP {np.mean(secondary_elite_apps) if secondary_elite_apps else 0.0:.3f}"
            )
        if len(primary_np) == 0 or len(secondary_np) == 0:
            raise SystemExit(
                "source-aware sampling produced an empty source after filtering; "
                "adjust dataset ordering, split, or elite criteria"
            )
        primary_train_idx = torch.as_tensor(primary_np, dtype=torch.long)
        secondary_train_idx = torch.as_tensor(secondary_np, dtype=torch.long)
        print(
            f"source mix     primary {len(primary_train_idx)} weighted samples / "
            f"secondary {len(secondary_train_idx)} from final {args.secondary_source_count} datasets; "
            f"batch secondary fraction {args.secondary_source_fraction:.3f}"
        )
    val_idx = torch.as_tensor(val_np, dtype=torch.long)
    if len(val_idx) == 0:
        print("warning: no game-level validation group is available; using training data")
        val_idx = torch.as_tensor(train_np[:raw_train_samples], dtype=torch.long)
    print(f"split         train {len(train_idx)} weighted / validation {len(val_idx)} samples")

    primary_policy_anchor = None
    secondary_policy_teacher_target = None

    def batch_at(idx, use_policy_overrides: bool = False):
        batch = {k: v[idx] for k, v in data.items()}
        if use_policy_overrides:
            assert secondary_start is not None
            overridden_target = None
            if primary_policy_anchor is not None:
                primary_mask = idx < secondary_start
                if bool(primary_mask.any()):
                    overridden_target = batch["policy_target"].clone()
                    overridden_target[primary_mask] = primary_policy_anchor[idx[primary_mask]]
            if secondary_policy_teacher_target is not None:
                secondary_mask = idx >= secondary_start
                if bool(secondary_mask.any()):
                    if overridden_target is None:
                        overridden_target = batch["policy_target"].clone()
                    secondary_indices = idx[secondary_mask] - secondary_start
                    overridden_target[secondary_mask] = secondary_policy_teacher_target[
                        secondary_indices
                    ]
            if overridden_target is not None:
                batch["policy_target"] = overridden_target
        if device != "cpu":
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return batch

    if args.anchor_primary_policy:
        assert secondary_start is not None
        action_count = data["policy_target"].shape[1]
        primary_policy_anchor = torch.zeros(
            (secondary_start, action_count), dtype=data["policy_target"].dtype
        )
        model.eval()
        anchor_batch = max(1, args.batch)
        with torch.no_grad():
            for offset in range(0, secondary_start, anchor_batch):
                end = min(secondary_start, offset + anchor_batch)
                idx = torch.arange(offset, end, dtype=torch.long)
                batch = batch_at(idx)
                logits, _, _ = model(
                    batch["tokens"], batch["token_mask"],
                    batch["actions"], batch["action_mask"]
                )
                primary_policy_anchor[offset:end] = torch.softmax(
                    logits.float(), dim=-1
                ).to("cpu", dtype=primary_policy_anchor.dtype)
        model.train()
        print(
            f"policy anchor  captured resumed policy on {secondary_start} primary-source samples"
        )

    if args.secondary_policy_teacher:
        assert secondary_start is not None
        teacher = load_checkpoint_model(args.secondary_policy_teacher, device)
        action_count = data["policy_target"].shape[1]
        secondary_count = len(ds) - secondary_start
        secondary_policy_teacher_target = torch.zeros(
            (secondary_count, action_count), dtype=data["policy_target"].dtype
        )
        teacher_batch = max(1, args.batch)
        with torch.no_grad():
            for offset in range(secondary_start, len(ds), teacher_batch):
                end = min(len(ds), offset + teacher_batch)
                idx = torch.arange(offset, end, dtype=torch.long)
                batch = batch_at(idx)
                logits, _, _ = teacher(
                    batch["tokens"], batch["token_mask"],
                    batch["actions"], batch["action_mask"]
                )
                secondary_policy_teacher_target[
                    offset - secondary_start:end - secondary_start
                ] = torch.softmax(logits.float(), dim=-1).to(
                    "cpu", dtype=secondary_policy_teacher_target.dtype
                )
        del teacher
        if device != "cpu":
            torch.cuda.empty_cache()
        model.train()
        print(
            f"policy teacher captured {args.secondary_policy_teacher} on "
            f"{secondary_count} secondary-source samples"
        )

    def evaluate():
        if len(val_idx) == 0:
            return None
        model.eval()
        sample_total = 0
        sample_sums = {
            "policy": 0.0,
            "chosen_action": 0.0,
            "chosen_disagreement": 0.0,
            "policy_rank": 0.0,
            "policy_pair_rank": 0.0,
            "timing_pair": 0.0,
            "timing_rank": 0.0,
            "value": 0.0,
            "value_accuracy": 0.0,
            "value_scalar_mse": 0.0,
            "topout_aux": 0.0,
        }
        aux_numerator = 0.0
        aux_valid_total = 0.0
        vs_aux_numerator = 0.0
        vs_aux_valid_total = 0.0
        cancellation_aux_numerator = 0.0
        cancellation_aux_valid_total = 0.0
        eval_batch = max(1, args.batch)
        with torch.no_grad():
            for offset in range(0, len(val_idx), eval_batch):
                idx = val_idx[offset:offset + eval_batch]
                batch = batch_at(idx)
                _, parts = losses(model, batch, weights=loss_weights)
                count = len(idx)
                sample_total += count
                for name in sample_sums:
                    sample_sums[name] += parts[name] * count
                valid = float(batch["aux_valid_mask"].sum().item())
                if valid > 0.0:
                    aux_numerator += parts["aux"] * valid
                    aux_valid_total += valid
                vs_valid = float(parts["vs_aux_valid"])
                if vs_valid > 0.0:
                    vs_aux_numerator += parts["vs_aux"] * vs_valid
                    vs_aux_valid_total += vs_valid
                cancellation_valid = float(parts["cancellation_aux_valid"])
                if cancellation_valid > 0.0:
                    cancellation_aux_numerator += parts["cancellation_aux"] * cancellation_valid
                    cancellation_aux_valid_total += cancellation_valid
        model.train()
        result = {name: total / max(1, sample_total)
                  for name, total in sample_sums.items()}
        result["aux"] = aux_numerator / aux_valid_total if aux_valid_total > 0.0 else 0.0
        result["aux_valid"] = aux_valid_total
        result["vs_aux"] = (
            vs_aux_numerator / vs_aux_valid_total if vs_aux_valid_total > 0.0 else 0.0
        )
        result["vs_aux_valid"] = vs_aux_valid_total
        result["cancellation_aux"] = (
            cancellation_aux_numerator / cancellation_aux_valid_total
            if cancellation_aux_valid_total > 0.0 else 0.0
        )
        result["cancellation_aux_valid"] = cancellation_aux_valid_total
        result["total"] = (
            loss_weights["policy"] * result["policy"]
            + loss_weights["chosen_action"] * result["chosen_action"]
            + loss_weights["chosen_disagreement"] * result["chosen_disagreement"]
            + loss_weights["policy_rank"] * result["policy_rank"]
            + loss_weights["policy_pair_rank"] * result["policy_pair_rank"]
            + loss_weights["timing_pair"] * result["timing_pair"]
            + loss_weights["timing_rank"] * result["timing_rank"]
            + loss_weights["value"] * result["value"]
            + loss_weights["aux"] * result["aux"]
            + loss_weights["vs_aux"] * result["vs_aux"]
            + loss_weights["cancellation_aux"] * result["cancellation_aux"]
            + loss_weights["topout_aux"] * result["topout_aux"]
        )
        return result

    gen = torch.Generator().manual_seed(args.seed + 1)
    if (isinstance(checkpoint, dict) and checkpoint.get("sampling_generator_state") is not None
            and not args.reset_sampling):
        gen.set_state(checkpoint["sampling_generator_state"])
    elif args.reset_sampling:
        print(f"sampling      reset from seed {args.seed}")

    first = evaluate()
    print(f"val (start)   total {first['total']:.4f}  policy {first['policy']:.4f}  "
          f"value {first['value']:.4f}  v_acc {first['value_accuracy']:.3f}  "
          f"v_mse {first['value_scalar_mse']:.4f}  vs {first['vs_aux']:.4f}  "
          f"cancel {first['cancellation_aux']:.4f}  "
          f"timing {first['timing_pair']:.4f}/{first['timing_rank']:.4f}")
    best_total = first["total"]
    best_step = start_step
    eval_interval = args.eval_every
    if eval_interval <= 0 and args.best_save and args.checkpoint_every > 0:
        eval_interval = args.checkpoint_every
    if args.best_save:
        save_checkpoint(args.best_save, model, opt, best_step, gen, loss_weights)

    t0 = time.time()
    log_interval = max(1, args.steps // 10)
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        batch_n = min(args.batch, len(train_idx))
        if secondary_train_idx is None:
            pick = train_idx[
                torch.randint(0, len(train_idx), (batch_n,), generator=gen)
            ]
        else:
            assert primary_train_idx is not None
            secondary_n = int(round(batch_n * args.secondary_source_fraction))
            secondary_n = max(1, min(batch_n - 1, secondary_n))
            primary_n = batch_n - secondary_n
            primary_pick = primary_train_idx[
                torch.randint(0, len(primary_train_idx), (primary_n,), generator=gen)
            ]
            secondary_pick = secondary_train_idx[
                torch.randint(0, len(secondary_train_idx), (secondary_n,), generator=gen)
            ]
            pick = torch.cat((primary_pick, secondary_pick))
            pick = pick[torch.randperm(len(pick), generator=gen)]
        total, parts = losses(
            model,
            batch_at(
                pick,
                use_policy_overrides=(
                    args.anchor_primary_policy or bool(args.secondary_policy_teacher)
                ),
            ),
            weights=loss_weights,
        )
        opt.zero_grad(set_to_none=True)
        # Gradient diagnostics require three additional autograd traversals.
        # They are observability, not part of optimisation, so only pay that
        # cost on steps whose diagnostics are actually printed.
        should_log = local_step % log_interval == 0
        grad_parts = None
        if should_log:
            grad_parts = gradient_diagnostics(
                shared_gradient_model, parts["_loss_tensors"], loss_weights
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if should_log:
            assert grad_parts is not None
            print(f"step {step:5d}  total {parts['total']:.4f}  policy {parts['policy']:.4f}  "
                  f"value {parts['value']:.4f}  v_acc {parts['value_accuracy']:.3f}  "
                  f"aux {parts['aux']:.4f}  vs {parts['vs_aux']:.4f}  "
                  f"cancel {parts['cancellation_aux']:.4f}  "
                  f"timing {parts['timing_pair']:.4f}/{parts['timing_rank']:.4f}  "
                  f"topout {parts['topout_aux']:.4f}  valid {parts['aux_valid']:.0f}  "
                  f"grad(policy/value/aux/vs/cancel/tpair/trank) {grad_parts['policy_grad']:.3g}/"
                  f"{grad_parts['value_grad']:.3g}/{grad_parts['aux_grad']:.3g}/"
                  f"{grad_parts['vs_aux_grad']:.3g}/{grad_parts['cancellation_aux_grad']:.3g}/"
                  f"{grad_parts['timing_pair_grad']:.3g}/{grad_parts['timing_rank_grad']:.3g}  "
                  f"cos(p,v/a/vs/cancel/tpair/trank) {grad_parts['policy_value_cos']:.3f}/"
                  f"{grad_parts['policy_aux_cos']:.3f}/"
                  f"{grad_parts['policy_vs_aux_cos']:.3f}/"
                  f"{grad_parts['policy_cancellation_aux_cos']:.3f}/"
                  f"{grad_parts['policy_timing_pair_cos']:.3f}/"
                  f"{grad_parts['policy_timing_rank_cos']:.3f}", flush=True)
        if args.checkpoint_every > 0 and args.save and step % args.checkpoint_every == 0:
            save_checkpoint(args.save, model, opt, step, gen, loss_weights)
        if eval_interval > 0 and local_step % eval_interval == 0:
            interim = evaluate()
            print(f"val (step {step}) total {interim['total']:.4f}  "
                  f"policy {interim['policy']:.4f}  value {interim['value']:.4f}  "
                  f"v_acc {interim['value_accuracy']:.3f}  "
                  f"v_mse {interim['value_scalar_mse']:.4f}  "
                  f"vs {interim['vs_aux']:.4f}  "
                  f"cancel {interim['cancellation_aux']:.4f}  "
                  f"timing {interim['timing_pair']:.4f}/{interim['timing_rank']:.4f}", flush=True)
            if interim["total"] < best_total:
                best_total = interim["total"]
                best_step = step
                if args.best_save:
                    save_checkpoint(args.best_save, model, opt, best_step, gen, loss_weights)

    secs = time.time() - t0
    last = evaluate()
    print(f"val (end)     total {last['total']:.4f}  policy {last['policy']:.4f}  "
          f"value {last['value']:.4f}  v_acc {last['value_accuracy']:.3f}  "
          f"v_mse {last['value_scalar_mse']:.4f}  vs {last['vs_aux']:.4f}  "
          f"cancel {last['cancellation_aux']:.4f}  "
          f"timing {last['timing_pair']:.4f}/{last['timing_rank']:.4f}")
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
