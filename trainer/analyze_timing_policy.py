# SPDX-License-Identifier: MIT
"""Measure learned action timing choices, especially deliberate non-cancelling.

The Tetra tokenizer has a fixed prefix under the current 10x24 / five-preview
contract.  The garbage-summary token is therefore token 43 (0-based), where
feature 2 is the pending-garbage flag. Action feature 21 stores delay_bin / 5.

This utility compares raw network argmax timing with the MCTS policy target on
one or more rectangular self-play datasets. It is diagnostic only; it does not
change checkpoints or datasets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

import tetra_dataset
from ablation_models import load_ablation_checkpoint


GARBAGE_SUMMARY_INDEX = 43
DELAY_NAMES = {0: "FASTEST", 1: "+1F", 2: "+2F", 3: "+4F", 4: "+8F", 5: "WAIT_FOR_EVENT"}


def add_counts(dst: dict[int, float], values: torch.Tensor, weights: torch.Tensor | None = None) -> None:
    if weights is None:
        for value in values.tolist():
            dst[int(value)] += 1.0
    else:
        for value, weight in zip(values.tolist(), weights.tolist()):
            dst[int(value)] += float(weight)


def pct_table(counts: dict[int, float]) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "(none)"
    return "  ".join(
        f"{DELAY_NAMES[i]}={100.0 * counts.get(i, 0.0) / total:.1f}%"
        for i in (0, 1, 2, 3, 4, 5)
        if counts.get(i, 0.0) > 0.0
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--factor-timing-policy", action="store_true",
                    help="inference-only override: preserve base-placement mass across FASTEST/WAIT variants")
    ap.add_argument("--timing-wait-bias", type=float, default=0.0,
                    help="diagnostic WAIT logit bias used with --factor-timing-policy")
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    if ds.tokens.shape[1] <= GARBAGE_SUMMARY_INDEX:
        raise SystemExit("dataset token width is too small for current timing diagnostic")

    device = torch.device(args.device)
    model = load_ablation_checkpoint(args.checkpoint, device)
    if abs(args.timing_wait_bias) > 1e-12 and not args.factor_timing_policy:
        raise SystemExit("--timing-wait-bias requires --factor-timing-policy")
    if args.factor_timing_policy:
        cfg = getattr(model, "cfg", None)
        if (cfg is None or not hasattr(cfg, "factor_timing_policy") or
                not hasattr(cfg, "timing_wait_logit_bias")):
            raise SystemExit("checkpoint architecture does not support --factor-timing-policy")
        cfg.factor_timing_policy = True
        cfg.timing_wait_logit_bias = float(args.timing_wait_bias)
    model.eval()
    data = ds.torch()

    groups = {
        "all": torch.ones(len(ds), dtype=torch.bool),
        "pending": torch.from_numpy(ds.tokens[:, GARBAGE_SUMMARY_INDEX, 2] > 0.5),
        "no_pending": torch.from_numpy(ds.tokens[:, GARBAGE_SUMMARY_INDEX, 2] <= 0.5),
    }

    model_counts = {name: defaultdict(float) for name in groups}
    model_mass = {name: defaultdict(float) for name in groups}
    model_clear_counts = {name: defaultdict(float) for name in groups}
    teacher_argmax_counts = {name: defaultdict(float) for name in groups}
    chosen_action_counts = {name: defaultdict(float) for name in groups}
    teacher_mass = {name: defaultdict(float) for name in groups}
    wait_model_values = {name: [] for name in groups}
    wait_teacher_values = {name: [] for name in groups}
    samples = {name: 0 for name in groups}
    clear_samples = {name: 0 for name in groups}
    pair_weight = 0.0
    pair_abs_error = 0.0
    pair_sq_error = 0.0
    pair_side_correct = 0.0
    pair_count = 0
    pair_teacher_wait_count = 0
    pair_model_wait_count = 0
    pair_teacher_wait_weight = 0.0
    pair_model_wait_weight = 0.0

    with torch.inference_mode():
        for offset in range(0, len(ds), max(1, args.batch)):
            end = min(len(ds), offset + max(1, args.batch))
            idx = slice(offset, end)
            batch = {key: value[idx].to(device, non_blocking=True) for key, value in data.items()}
            logits, _, _ = model(
                batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
            )
            legal = batch["action_mask"] > 0.5
            delay = torch.round(batch["actions"][..., 21] * 5.0).long().clamp(0, 5)
            clears = batch["actions"][..., 14] > 0.0
            masked_logits = logits.masked_fill(~legal, float("-inf"))
            model_prob = torch.softmax(masked_logits, dim=-1)
            model_best = masked_logits.argmax(dim=-1)
            teacher_best = batch["policy_target"].argmax(dim=-1)
            model_delay = delay.gather(1, model_best[:, None]).squeeze(1)
            teacher_delay = delay.gather(1, teacher_best[:, None]).squeeze(1)
            chosen = batch.get("chosen_action")
            if chosen is not None:
                chosen_safe = chosen.long().clamp(0, delay.shape[1] - 1)
                chosen_delay = delay.gather(1, chosen_safe[:, None]).squeeze(1)
            else:
                chosen_delay = teacher_delay
            model_clear = clears.gather(1, model_best[:, None]).squeeze(1)
            target = batch["policy_target"] * legal.float()
            model_wait = model_prob.masked_fill(delay != 5, 0.0).sum(dim=1)
            teacher_wait = target.masked_fill(delay != 5, 0.0).sum(dim=1)

            # WAIT_FOR_EVENT, when present, is emitted immediately after its
            # FASTEST base placement. Some placements need no WAIT because their
            # natural duration already crosses the event, so match adjacent
            # actions by all non-delay features rather than assuming even/odd
            # pairs across the entire action list.
            if batch["actions"].shape[1] >= 2:
                actions = batch["actions"]
                base_delta = torch.cat(
                    [actions[:, 1:, :21] - actions[:, :-1, :21],
                     actions[:, 1:, 22:] - actions[:, :-1, 22:]], dim=-1
                )
                same_base = base_delta.abs().amax(dim=-1) <= 1e-6
                fast_logits = masked_logits[:, :-1]
                wait_logits = masked_logits[:, 1:]
                fast_target = target[:, :-1]
                wait_target = target[:, 1:]
                pair_mass = fast_target + wait_target
                pending_batch = groups["pending"][offset:end].to(device)[:, None]
                valid_pair = (
                    pending_batch & legal[:, :-1] & legal[:, 1:]
                    & (delay[:, :-1] == 0) & (delay[:, 1:] == 5)
                    & same_base & (pair_mass > 0.0)
                )
                if valid_pair.any():
                    model_pair_wait = torch.sigmoid((wait_logits - fast_logits)[valid_pair])
                    teacher_pair_wait = (wait_target / pair_mass.clamp_min(1e-12))[valid_pair]
                    err = model_pair_wait - teacher_pair_wait
                    w = pair_mass[valid_pair]
                    pair_weight += float(w.sum().item())
                    pair_abs_error += float((err.abs() * w).sum().item())
                    pair_sq_error += float((err.square() * w).sum().item())
                    model_wait_side = model_pair_wait >= 0.5
                    teacher_wait_side = teacher_pair_wait >= 0.5
                    same_side = (model_wait_side == teacher_wait_side).float()
                    pair_side_correct += float((same_side * w).sum().item())
                    pair_teacher_wait_count += int(teacher_wait_side.sum().item())
                    pair_model_wait_count += int(model_wait_side.sum().item())
                    pair_teacher_wait_weight += float((teacher_wait_side.float() * w).sum().item())
                    pair_model_wait_weight += float((model_wait_side.float() * w).sum().item())
                    pair_count += int(valid_pair.sum().item())

            for name, group_full in groups.items():
                selected = group_full[offset:end].to(device)
                if not selected.any():
                    continue
                samples[name] += int(selected.sum().item())
                add_counts(model_counts[name], model_delay[selected])
                add_counts(teacher_argmax_counts[name], teacher_delay[selected])
                add_counts(chosen_action_counts[name], chosen_delay[selected])
                wait_model_values[name].extend(model_wait[selected].detach().cpu().tolist())
                wait_teacher_values[name].extend(teacher_wait[selected].detach().cpu().tolist())

                chosen_clear = selected & model_clear
                clear_samples[name] += int(chosen_clear.sum().item())
                if chosen_clear.any():
                    add_counts(model_clear_counts[name], model_delay[chosen_clear])

                # Network probability mass and MCTS visit mass by timing bin,
                # both restricted to legal actions. Argmax alone can be
                # misleading when FASTEST/WAIT variants form near-tied pairs.
                for bin_id in range(6):
                    prob_mass = model_prob.masked_fill(delay != bin_id, 0.0).sum(dim=1)
                    model_mass[name][bin_id] += float(prob_mass[selected].sum().item())
                    mass = target.masked_fill(delay != bin_id, 0.0).sum(dim=1)
                    teacher_mass[name][bin_id] += float(mass[selected].sum().item())

    print(f"checkpoint    {args.checkpoint}")
    print(f"samples       {len(ds)}")
    for name in ("all", "pending", "no_pending"):
        print(f"\n[{name}] n={samples[name]}")
        print("model argmax  " + pct_table(model_counts[name]))
        print("model mass    " + pct_table(model_mass[name]))
        print("teacher top1  " + pct_table(teacher_argmax_counts[name]))
        print("search chosen " + pct_table(chosen_action_counts[name]))
        print("teacher mass  " + pct_table(teacher_mass[name]))
        if name == "pending" and wait_model_values[name]:
            model_wait_np = np.asarray(wait_model_values[name], dtype=np.float64)
            teacher_wait_np = np.asarray(wait_teacher_values[name], dtype=np.float64)
            diff = model_wait_np - teacher_wait_np
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff * diff)))
            if np.std(model_wait_np) > 0.0 and np.std(teacher_wait_np) > 0.0:
                corr = float(np.corrcoef(model_wait_np, teacher_wait_np)[0, 1])
            else:
                corr = float("nan")
            decision = float(np.mean((model_wait_np >= 0.5) == (teacher_wait_np >= 0.5)))
            print(
                f"wait fit      MAE={mae:.4f} RMSE={rmse:.4f} "
                f"corr={corr:.4f} side@50={100.0 * decision:.1f}%"
            )
            if pair_weight > 0.0:
                print(
                    f"pair fit      n={pair_count} weighted-MAE={pair_abs_error / pair_weight:.4f} "
                    f"weighted-RMSE={np.sqrt(pair_sq_error / pair_weight):.4f} "
                    f"side@50={100.0 * pair_side_correct / pair_weight:.1f}%"
                )
                print(
                    f"pair WAIT     teacher={100.0 * pair_teacher_wait_count / max(1, pair_count):.1f}% "
                    f"model={100.0 * pair_model_wait_count / max(1, pair_count):.1f}%  "
                    f"mass-weighted teacher={100.0 * pair_teacher_wait_weight / pair_weight:.1f}% "
                    f"model={100.0 * pair_model_wait_weight / pair_weight:.1f}%"
                )
        print(f"model clear   n={clear_samples[name]}  " + pct_table(model_clear_counts[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
