#!/usr/bin/env python3
"""Compare two policy checkpoints across observable pressure/danger strata.

This diagnostic is intended for deciding whether a context-gated policy is
justified.  It uses only features already present in the model observation:
self board summary (token 34 for the current tokenizer contract) and garbage
summary (token 43).  The competitive search-chosen action is used as the local
reference action.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetra_dataset
from train import load_checkpoint_model


def unsquash(value: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(value.astype(np.float64), 0.0, 1.0 - 1e-8)
    return scale * value / (1.0 - value)


@dataclass
class PolicyResult:
    best: np.ndarray
    chosen_prob: np.ndarray


def evaluate(model, data: dict[str, torch.Tensor], chosen: np.ndarray,
             device: torch.device, batch_size: int) -> PolicyResult:
    n = len(chosen)
    best = np.full(n, -1, dtype=np.int32)
    chosen_prob = np.zeros(n, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            sl = slice(start, end)
            tokens = data["tokens"][sl].to(device, non_blocking=True)
            token_mask = data["token_mask"][sl].to(device, non_blocking=True)
            actions = data["actions"][sl].to(device, non_blocking=True)
            action_mask = data["action_mask"][sl].to(device, non_blocking=True)
            logits, _, _ = model(tokens, token_mask, actions, action_mask)
            probs = torch.softmax(logits.float(), dim=-1)
            model_best = logits.argmax(dim=-1)
            chosen_t = torch.as_tensor(chosen[start:end], device=device).long()
            safe = chosen_t.clamp(0, logits.shape[1] - 1)
            prob = probs.gather(1, safe[:, None]).squeeze(1)
            best[start:end] = model_best.cpu().numpy().astype(np.int32)
            chosen_prob[start:end] = prob.cpu().numpy()
    return PolicyResult(best=best, chosen_prob=chosen_prob)


def report_bucket(name: str, mask: np.ndarray, chosen: np.ndarray,
                  a: PolicyResult, b: PolicyResult) -> None:
    valid = mask & (chosen >= 0)
    n = int(valid.sum())
    if n == 0:
        print(f"{name:24s} n=0")
        return
    a_ok = a.best == chosen
    b_ok = b.best == chosen
    disagree = a.best != b.best
    dmask = valid & disagree
    dn = int(dmask.sum())
    a_acc = float(a_ok[valid].mean())
    b_acc = float(b_ok[valid].mean())
    a_prob = float(a.chosen_prob[valid].mean())
    b_prob = float(b.chosen_prob[valid].mean())
    if dn:
        a_wins = int((dmask & a_ok & ~b_ok).sum())
        b_wins = int((dmask & b_ok & ~a_ok).sum())
        both_miss = int((dmask & ~a_ok & ~b_ok).sum())
    else:
        a_wins = b_wins = both_miss = 0
    print(
        f"{name:24s} n={n:5d} A={a_acc:.3f} B={b_acc:.3f} "
        f"p(chosen)={a_prob:.3f}/{b_prob:.3f} disagree={dn/n:.3f} "
        f"Awin/Bwin/miss={a_wins}/{b_wins}/{both_miss}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--board-summary-index", type=int, default=34)
    ap.add_argument("--garbage-summary-index", type=int, default=43)
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    if ds.tokens.shape[1] <= max(args.board_summary_index, args.garbage_summary_index):
        raise SystemExit("dataset has too few tokens for requested summary indices")
    if not np.all(ds.token_mask[:, args.board_summary_index] > 0.5):
        raise SystemExit("board summary token is not present on every sample")
    if not np.all(ds.token_mask[:, args.garbage_summary_index] > 0.5):
        raise SystemExit("garbage summary token is not present on every sample")

    board = ds.tokens[:, args.board_summary_index]
    garbage = ds.tokens[:, args.garbage_summary_index]
    height = board[:, 0].astype(np.float64) * 20.0
    holes = unsquash(board[:, 1], 8.0)
    garbage_rows = unsquash(board[:, 4], 8.0)
    pending = unsquash(garbage[:, 0], 6.0)
    active = unsquash(garbage[:, 1], 6.0)
    has_pending = garbage[:, 2] > 0.5
    chosen = np.asarray(ds.chosen_action, dtype=np.int32)
    legal_count = ds.action_mask.sum(axis=1).astype(np.int32)
    chosen_valid = (chosen >= 0) & (chosen < ds.actions.shape[1])
    rows = np.arange(len(ds))
    chosen_valid &= ds.action_mask[rows, np.clip(chosen, 0, ds.actions.shape[1] - 1)] > 0.5
    chosen = np.where(chosen_valid, chosen, -1).astype(np.int32)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    print(f"dataset       {len(ds)} samples")
    print(f"device        {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(
        f"state ranges  height={height.min():.1f}..{height.max():.1f} "
        f"pending={pending.min():.1f}..{pending.max():.1f} "
        f"active={active.min():.1f}..{active.max():.1f} holes={holes.min():.1f}..{holes.max():.1f} "
        f"garbage_rows={garbage_rows.min():.1f}..{garbage_rows.max():.1f}"
    )

    data = ds.torch()
    model_a = load_checkpoint_model(args.model_a, str(device))
    result_a = evaluate(model_a, data, chosen, device, max(1, args.batch))
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_b = load_checkpoint_model(args.model_b, str(device))
    result_b = evaluate(model_b, data, chosen, device, max(1, args.batch))
    del model_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"A             {args.model_a}")
    print(f"B             {args.model_b}")
    print("\n-- broad strata --")
    all_mask = np.ones(len(ds), dtype=bool)
    report_bucket("all", all_mask, chosen, result_a, result_b)
    report_bucket("no pending", ~has_pending, chosen, result_a, result_b)
    report_bucket("pending > 0", has_pending, chosen, result_a, result_b)
    report_bucket("active > 0", active > 0.25, chosen, result_a, result_b)
    report_bucket("height < 8", height < 8.0, chosen, result_a, result_b)
    report_bucket("height 8-12", (height >= 8.0) & (height < 13.0), chosen, result_a, result_b)
    report_bucket("height >= 13", height >= 13.0, chosen, result_a, result_b)
    report_bucket("holes < 1", holes < 0.5, chosen, result_a, result_b)
    report_bucket("holes >= 3", holes >= 2.5, chosen, result_a, result_b)
    report_bucket("garbage rows = 0", garbage_rows < 0.5, chosen, result_a, result_b)
    report_bucket("garbage rows > 0", garbage_rows >= 0.5, chosen, result_a, result_b)

    print("\n-- gating-relevant joint strata --")
    safe = (~has_pending) & (height < 10.0) & (holes < 2.5)
    pressure_low = has_pending & (height < 10.0)
    pressure_high = has_pending & (height >= 10.0)
    high_no_pending = (~has_pending) & (height >= 10.0)
    report_bucket("safe/no-pressure", safe, chosen, result_a, result_b)
    report_bucket("pending + low stack", pressure_low, chosen, result_a, result_b)
    report_bucket("pending + high stack", pressure_high, chosen, result_a, result_b)
    report_bucket("high stack/no pending", high_no_pending, chosen, result_a, result_b)
    pristine = (~has_pending) & (garbage_rows < 0.5) & (holes < 0.5)
    pristine_low = pristine & (height < 10.0)
    clean_board = (~has_pending) & (garbage_rows < 0.5)
    report_bucket("pristine board", pristine, chosen, result_a, result_b)
    report_bucket("pristine + low stack", pristine_low, chosen, result_a, result_b)
    report_bucket("no pending/no garbage", clean_board, chosen, result_a, result_b)

    move_number = np.asarray(ds.move_number, dtype=np.int64)
    print("\n-- trajectory phase --")
    report_bucket("move < 50", move_number < 50, chosen, result_a, result_b)
    report_bucket("move 50-149", (move_number >= 50) & (move_number < 150), chosen, result_a, result_b)
    report_bucket("move >= 150", move_number >= 150, chosen, result_a, result_b)

    print("\n-- pending-line bands --")
    report_bucket("pending ~0", pending < 0.5, chosen, result_a, result_b)
    report_bucket("pending 1-4", (pending >= 0.5) & (pending < 4.5), chosen, result_a, result_b)
    report_bucket("pending 5-8", (pending >= 4.5) & (pending < 8.5), chosen, result_a, result_b)
    report_bucket("pending >=9", pending >= 8.5, chosen, result_a, result_b)

    print("\n-- legal-action complexity --")
    report_bucket("legal <= 20", legal_count <= 20, chosen, result_a, result_b)
    report_bucket("legal 21-40", (legal_count > 20) & (legal_count <= 40), chosen, result_a, result_b)
    report_bucket("legal > 40", legal_count > 40, chosen, result_a, result_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
