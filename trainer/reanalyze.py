# SPDX-License-Identifier: MIT
"""Refresh selected historical search targets with a current GPU teacher.

The C++ child reconstructs exact simulator roots and owns search.  This Python
process only serves model inference and writes content-addressed provenance.
Original datasets are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from gpu_match import REQUEST_MAGIC, answer_request, load_model, read_exact
from tetra_dataset import load as load_dataset


DONE_MAGIC = b"RANL"
DONE_FORMAT = "<4I2Q"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_child(
    model: torch.nn.Module, device: torch.device, precision: str,
    engine: Path, source: Path, output: Path, audit: Path,
    select_count: int, sims: int, batch: int, determinizations: int,
    use_gumbel: bool, no_attack_delivery: bool, timing_actions: bool,
    teacher_model_version: int,
) -> tuple[dict[str, int], float, str]:
    proc = subprocess.Popen(
        [
            str(engine), "gpu-reanalyse-protocol", str(source), str(output),
            str(audit), str(select_count), str(sims), str(batch),
            str(determinizations), "1" if use_gumbel else "0",
            "1" if no_attack_delivery else "0",
            "1" if timing_actions else "0", str(teacher_model_version),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    inference_seconds = 0.0
    summary: dict[str, int] = {}
    try:
        while True:
            magic = read_exact(proc.stdout, 4)
            if magic == REQUEST_MAGIC:
                inference_seconds += answer_request(model, device, proc, precision)
                continue
            if magic == DONE_MAGIC:
                values = struct.unpack(DONE_FORMAT, read_exact(
                    proc.stdout, struct.calcsize(DONE_FORMAT)))
                summary = {
                    "source_rows": values[0], "selected_rows": values[1],
                    "token_rows_verified": values[2],
                    "action_rows_verified": values[3],
                    "positions_evaluated": values[4], "batches_issued": values[5],
                }
                break
            raise RuntimeError(f"unexpected reanalyse protocol frame: {magic!r}")
    except Exception:
        proc.kill()
        proc.wait()
        raise
    proc.stdin.close()
    return_code = proc.wait()
    stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0:
        raise RuntimeError(f"reanalyse child failed with {return_code}: {stderr}")
    return summary, inference_seconds, stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--engine", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--select-count", type=int)
    select.add_argument("--select-fraction", type=float, default=0.05)
    parser.add_argument("--original-sims", type=int, required=True)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--no-gumbel", action="store_true")
    parser.add_argument("--no-attack-delivery", action="store_true")
    parser.add_argument("--timing-actions", action="store_true")
    parser.add_argument("--teacher-model-version", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    manifest = output.with_suffix(output.suffix + ".reanalyze.json")
    audit = output.with_suffix(output.suffix + ".audit.jsonl")
    for path in (output, manifest, audit):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing Reanalyse artifact: {path}")
    if not source.is_file() or not checkpoint.is_file():
        raise SystemExit("checkpoint and source dataset must exist")
    engine = Path(args.engine).resolve() if args.engine else root / (
        "build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli")
    if not engine.is_file():
        raise SystemExit(f"engine not found: {engine}; run make tools first")
    if args.original_sims < 1 or args.sims <= args.original_sims:
        raise SystemExit("--sims must be materially larger than positive --original-sims")
    if args.select_fraction is not None and not 0.0 < args.select_fraction <= 1.0:
        raise SystemExit("--select-fraction must be in (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")

    header_only = load_dataset(str(source))
    source_rows = len(header_only)
    del header_only
    select_count = args.select_count or max(1, round(source_rows * args.select_fraction))
    select_count = min(source_rows, select_count)
    if select_count >= source_rows:
        raise SystemExit("selection must leave some rows cheap-screen-only; uniform full re-search is disallowed")

    device = torch.device(args.device)
    model = load_model(str(checkpoint), device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tetradat", delete=False) as f:
            temp_output = Path(f.name)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".audit.jsonl", delete=False) as f:
            temp_audit = Path(f.name)
        temporary_paths.extend((temp_output, temp_audit))
        summary, inference_seconds, diagnostics = run_child(
            model, device, args.precision, engine, source, temp_output, temp_audit,
            select_count, args.sims, args.batch, args.determinizations,
            not args.no_gumbel, args.no_attack_delivery, args.timing_actions,
            args.teacher_model_version,
        )
        if summary["source_rows"] != source_rows or summary["selected_rows"] != select_count:
            raise RuntimeError(f"child summary mismatch: {summary}")
        if summary["token_rows_verified"] != source_rows or summary["action_rows_verified"] != source_rows:
            raise RuntimeError(f"reconstruction parity incomplete: {summary}")
        os.replace(temp_output, output)
        os.replace(temp_audit, audit)
        temporary_paths.clear()

        record = {
            "format": "tetraformer-reanalyse-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source),
            "source_sha256": sha256_file(source),
            "source_rows": source_rows,
            "original_model_version": int(load_dataset(str(source)).header.model_version),
            "original_search_budget": {"simulations": args.original_sims},
            "teacher_checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": sha256_file(checkpoint),
            "teacher_model_version": args.teacher_model_version,
            "engine": str(engine),
            "engine_commit": engine_commit(root),
            "rules": {
                "attack_delivery": not args.no_attack_delivery,
                "timing_actions": args.timing_actions,
            },
            "selection": {
                "method": "top historical-policy KL vs current raw policy",
                "requested_rows": select_count,
                "fraction": select_count / source_rows,
            },
            "reanalysis_search": {
                "simulations": args.sims, "batch": args.batch,
                "determinizations": args.determinizations,
                "algorithm": "puct" if args.no_gumbel else "gumbel",
                "root_noise_fraction": 0.0,
            },
            "label_contract": {
                "refreshed": ["policy_target", "search_value_in_audit"],
                "preserved": ["chosen_action", "terminal_value_target", "aux_targets"],
                "note": "rectangular v4 has no search-value training column; the refreshed search value is retained losslessly in the audit sidecar",
            },
            "output_dataset": str(output),
            "output_sha256": sha256_file(output),
            "audit_jsonl": str(audit),
            "audit_sha256": sha256_file(audit),
            "summary": summary,
            "gpu_inference_seconds": inference_seconds,
            "child_diagnostics": diagnostics,
        }
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    print(f"source       {source}")
    print(f"teacher      {checkpoint}")
    print(f"output       {output}")
    print(f"audit        {audit}")
    print(f"manifest     {manifest}")
    print(f"parity       {source_rows}/{source_rows} token rows, {source_rows}/{source_rows} action rows")
    print(f"selected     {select_count}/{source_rows} ({select_count / source_rows:.1%})")
    print(f"search       {args.original_sims} -> {args.sims} simulations, root noise 0")
    print(f"GPU infer    {inference_seconds:.2f}s / {summary['positions_evaluated']} positions / {summary['batches_issued']} batches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
