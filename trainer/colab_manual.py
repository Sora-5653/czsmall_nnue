# SPDX-License-Identifier: MIT
"""Semi-automatic Google Colab runner for self-play and ablation training.

This file is intentionally a small command-line wrapper around the project's
existing ``colab_generate.py`` and ``train.py`` entry points.  Run one command
at a time in Colab so that a failed stage can be inspected and rerun without
restarting the whole experiment::

    !python trainer/colab_manual.py setup
    !python trainer/colab_manual.py generate
    !python trainer/colab_manual.py inspect
    !python trainer/colab_manual.py train --condition aux --seed 0
    !python trainer/colab_manual.py train --condition noaux --seed 0

The source bundle and checkpoint are read from a Drive folder.  Generated
datasets, manifests and checkpoints are copied back to that same folder.  A
Windows-created ZIP may contain backslashes in member names; extraction below
normalizes those names to POSIX paths before writing them into the Colab VM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import zipfile
from typing import Iterable, Sequence


DEFAULT_DRIVE_FOLDER = "czsmall_nnue_colab_20260806"
DEFAULT_BUNDLE_NAME = "colab_bundle_sample_eff_manual.zip"
DEFAULT_CHECKPOINT_NAME = "baseline_gpu_gen_20260805_v2.best.pt"
DEFAULT_WORK_DIR = "/content/czsmall_nnue"
DEFAULT_MOUNTPOINT = "/content/drive"
DEFAULT_DATASET_NAME = "colab_shard_2026080600.tetradat"


@dataclass(frozen=True)
class Settings:
    drive_folder: str
    bundle_name: str
    checkpoint_name: str
    work_dir: Path
    mountpoint: Path
    dataset_name: str

    @property
    def drive_root(self) -> Path:
        return self.mountpoint / "MyDrive"

    @property
    def drive_dir(self) -> Path:
        return self.drive_root / self.drive_folder

    @property
    def bundle_path(self) -> Path:
        return self.drive_dir / self.bundle_name

    @property
    def drive_checkpoint_path(self) -> Path:
        return self.drive_dir / self.checkpoint_name

    @property
    def local_checkpoint_path(self) -> Path:
        return self.work_dir / "models" / "baseline.pt"

    @property
    def local_dataset_path(self) -> Path:
        return self.work_dir / "data" / "colab" / self.dataset_name

    @property
    def local_manifest_path(self) -> Path:
        return Path(str(self.local_dataset_path) + ".manifest.json")


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        drive_folder=args.drive_folder,
        bundle_name=args.bundle_name,
        checkpoint_name=args.checkpoint_name,
        work_dir=Path(args.work_dir).expanduser(),
        mountpoint=Path(args.mountpoint).expanduser(),
        dataset_name=args.dataset_name,
    )


def _print_command(command: Sequence[str]) -> None:
    print("$ " + shlex.join([str(part) for part in command]), flush=True)


def _run(command: Sequence[str], cwd: Path | None = None,
         env: dict[str, str] | None = None) -> None:
    _print_command(command)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    subprocess.run([str(part) for part in command], cwd=str(cwd) if cwd else None,
                   check=True, env=child_env)


CXX23_PROBE = r"""
#include <algorithm>
#include <array>
#include <ranges>
#include <utility>

constexpr int consteval_probe() {
    if consteval { return 1; }
    else { return 2; }
}

constexpr bool contains_probe() {
    constexpr std::array<int, 1> values{1};
    return std::ranges::contains(values, 1);
}

void unreachable_probe(bool condition) {
    if (condition) return;
    std::unreachable();
}

static_assert(consteval_probe() == 1);
static_assert(contains_probe());
"""


def _compiler_supports_cxx23(compiler: str) -> bool:
    try:
        result = subprocess.run(
            [compiler, "-std=c++23", "-x", "c++", "-fsyntax-only", "-"],
            input=CXX23_PROBE,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _compiler_candidates() -> list[str]:
    candidates: list[str] = []
    requested = os.environ.get("CXX", "").strip()
    if requested:
        candidates.append(requested)
    for candidate in (
        "g++-14", "g++-13", "g++-12", "g++",
        "clang++-18", "clang++-17", "clang++",
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _find_cxx23() -> str | None:
    for candidate in _compiler_candidates():
        if _compiler_supports_cxx23(candidate):
            return candidate
    return None


def _install_cxx23() -> str | None:
    if shutil.which("apt-get") is None:
        return None
    apt_env = {"DEBIAN_FRONTEND": "noninteractive"}
    try:
        _run(["apt-get", "update", "-qq"], env=apt_env)
    except subprocess.CalledProcessError:
        return None
    for package in ("g++-14", "g++-13", "g++-12"):
        try:
            _run(["apt-get", "install", "-y", "-qq", package], env=apt_env)
        except subprocess.CalledProcessError:
            continue
        compiler = _find_cxx23()
        if compiler:
            return compiler
    return None


def ensure_cxx23() -> str:
    compiler = _find_cxx23()
    if compiler:
        print(f"compiler     {compiler} (C++23 probe OK)")
        return compiler

    print("compiler     no installed compiler passed the C++23 feature probe")
    print("installing    a compatible g++ package via apt-get")
    compiler = _install_cxx23()
    if compiler:
        print(f"compiler     {compiler} (C++23 probe OK)")
        return compiler
    raise RuntimeError(
        "no compiler with the required C++23 library features was found. "
        "The probe requires if consteval, std::ranges::contains and "
        "std::unreachable; pass the apt-get output for diagnosis."
    )


def _require_colab_work_dir(path: Path) -> None:
    """Reject broad paths before a rerunnable setup replaces a directory."""
    resolved = path.resolve()
    if resolved == Path("/") or resolved == Path("/content"):
        raise RuntimeError(f"refusing to use a broad work directory: {resolved}")
    if not resolved.as_posix().startswith("/content/"):
        raise RuntimeError(
            f"work directory must be below /content in Colab, got: {resolved}"
        )


def mount_drive(settings: Settings) -> Path:
    """Mount Drive and return the experiment folder."""
    _require_colab_work_dir(settings.work_dir)
    settings.mountpoint.mkdir(parents=True, exist_ok=True)
    my_drive = settings.mountpoint / "MyDrive"
    if my_drive.is_dir():
        print(f"Drive already mounted at {settings.mountpoint}")
    else:
        try:
            from google.colab import drive  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "google.colab is unavailable. Run this command in Google Colab."
            ) from exc
        try:
            drive.mount(str(settings.mountpoint), force_remount=False)
        except AttributeError as exc:
            raise RuntimeError(
                "Drive is not mounted. Run drive.mount('/content/drive') "
                "in a notebook cell before invoking this script with !python."
            ) from exc
    if not settings.drive_dir.is_dir():
        raise FileNotFoundError(
            f"Drive folder not found: {settings.drive_dir}. "
            "Check --drive-folder and create/upload the bundle first."
        )
    return settings.drive_dir


def _normalise_zip_member(name: str) -> tuple[str, ...]:
    """Return safe POSIX path components for a ZIP member.

    PowerShell's ``Compress-Archive`` can produce member names containing
    backslashes.  Linux treats those as ordinary characters, so simply calling
    ``ZipFile.extractall`` creates a broken one-level tree.  Normalizing here
    keeps the bundle usable without requiring the user to rebuild it in Colab.
    """
    normalised = name.replace("\\", "/")
    pure = PurePosixPath(normalised)
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    if pure.is_absolute() or (parts and parts[0].endswith(":")) or ".." in parts:
        raise RuntimeError(f"unsafe ZIP member path: {name!r}")
    return parts


def extract_bundle(bundle_path: Path, destination: Path) -> None:
    """Extract a source bundle with path normalization and traversal checks."""
    bundle_path = bundle_path.resolve()
    destination = destination.resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"source bundle not found: {bundle_path}")

    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    seen_files: set[tuple[str, ...]] = set()
    staging_root = staging.resolve()

    try:
        with zipfile.ZipFile(bundle_path) as archive:
            for info in archive.infolist():
                parts = _normalise_zip_member(info.filename)
                if not parts:
                    continue
                target = staging.joinpath(*parts)
                target_resolved = target.resolve()
                try:
                    target_resolved.relative_to(staging_root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"ZIP member escapes extraction directory: {info.filename!r}"
                    ) from exc

                is_directory = info.is_dir() or info.filename.endswith(("/", "\\"))
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if parts in seen_files:
                    raise RuntimeError(
                        f"duplicate ZIP member after path normalization: {info.filename!r}"
                    )
                seen_files.add(parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)


def _check_bundle_layout(root: Path) -> None:
    required = (
        root / "Makefile",
        root / "trainer" / "colab_generate.py",
        root / "trainer" / "train.py",
        root / "include",
        root / "src",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "bundle extraction is incomplete; missing:\n  " + "\n  ".join(missing)
        )


def setup(settings: Settings) -> None:
    _require_colab_work_dir(settings.work_dir)
    drive_dir = mount_drive(settings)
    if not settings.bundle_path.is_file():
        raise FileNotFoundError(
            f"bundle not found: {settings.bundle_path}\n"
            "Upload the ZIP named by --bundle-name into the Drive folder."
        )
    if not settings.drive_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {settings.drive_checkpoint_path}\n"
            "Upload the checkpoint named by --checkpoint-name into the Drive folder."
        )

    print(f"bundle       {settings.bundle_path}")
    extract_bundle(settings.bundle_path, settings.work_dir)
    _check_bundle_layout(settings.work_dir)
    models = settings.work_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings.drive_checkpoint_path, settings.local_checkpoint_path)
    print(f"checkpoint   {settings.local_checkpoint_path} ({settings.local_checkpoint_path.stat().st_size} bytes)")

    compiler = ensure_cxx23()
    _run(["make", "tools"], cwd=settings.work_dir, env={"CXX": compiler})
    engine = settings.work_dir / "build" / "tetra_cli"
    if not engine.is_file():
        raise RuntimeError(f"engine was not built: {engine}")

    import torch

    print(f"root         {settings.work_dir}")
    print(f"torch        {torch.__version__}")
    print(f"cuda         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu          {torch.cuda.get_device_name(0)}")
    print(f"drive        {drive_dir}")
    print("setup        OK")


def _ensure_layout(settings: Settings) -> None:
    try:
        _check_bundle_layout(settings.work_dir)
    except RuntimeError:
        print("local bundle is not ready; running setup", flush=True)
        setup(settings)


def _copy_to_drive(settings: Settings, paths: Iterable[Path]) -> None:
    drive_dir = settings.drive_dir
    drive_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"expected output was not created: {source}")
        destination = drive_dir / source.name
        shutil.copy2(source, destination)
        print(f"uploaded     {destination} ({destination.stat().st_size} bytes)")


def _materialise_dataset(settings: Settings) -> None:
    if settings.local_dataset_path.is_file() and settings.local_manifest_path.is_file():
        return
    drive_dir = mount_drive(settings)
    drive_dataset = drive_dir / settings.dataset_name
    drive_manifest = drive_dir / (settings.dataset_name + ".manifest.json")
    if not drive_dataset.is_file() or not drive_manifest.is_file():
        raise FileNotFoundError(
            f"dataset or manifest not found in Drive: {drive_dataset} / {drive_manifest}"
        )
    settings.local_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_dataset, settings.local_dataset_path)
    shutil.copy2(drive_manifest, settings.local_manifest_path)
    print(f"downloaded   {settings.local_dataset_path}")
    print(f"downloaded   {settings.local_manifest_path}")


def validate(settings: Settings, require_complete: bool = True) -> None:
    _ensure_layout(settings)
    _materialise_dataset(settings)
    command = [
        sys.executable,
        str(settings.work_dir / "trainer" / "colab_generate.py"),
        "validate",
        str(settings.local_manifest_path),
        "--checkpoint",
        str(settings.local_checkpoint_path),
    ]
    if require_complete:
        command.append("--require-complete")
    _run(command, cwd=settings.work_dir)


def _output_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in ("", ".", ".."):
        raise ValueError(f"output name must be a single file name, got: {value!r}")
    return path.name


def generate(settings: Settings, args: argparse.Namespace) -> None:
    _ensure_layout(settings)
    output_name = _output_name(args.output_name)
    output = settings.work_dir / "data" / "colab" / output_name
    manifest = Path(str(output) + ".manifest.json")
    if not args.overwrite and (output.exists() or manifest.exists()):
        raise FileExistsError(
            f"output already exists: {output} or {manifest}; rerun with --overwrite"
        )

    command = [
        sys.executable,
        str(settings.work_dir / "trainer" / "colab_generate.py"),
        "generate",
        str(settings.local_checkpoint_path),
        str(output),
        "--repo-root",
        str(settings.work_dir),
        "--build-engine",
        "--device",
        args.device,
        "--games",
        str(args.games),
        "--pieces",
        str(args.pieces),
        "--sims",
        str(args.sims),
        "--inference-batch",
        str(args.inference_batch),
        "--base-seed",
        str(args.base_seed),
        "--shard-id",
        str(args.shard_id),
        "--shard-count",
        str(args.shard_count),
        "--model-version",
        str(args.model_version),
        "--determinizations",
        str(args.determinizations),
        "--precision",
        args.precision,
    ]
    if not args.gumbel:
        command.append("--no-gumbel")
    if args.overwrite:
        command.append("--overwrite")
    _run(command, cwd=settings.work_dir)

    validate(settings_for_dataset(settings, output_name),
             require_complete=args.shard_count == 1)
    _copy_to_drive(settings, (output, manifest))
    print(f"dataset      {output}")
    print(f"manifest     {manifest}")
    print("generate     OK")


def settings_for_dataset(settings: Settings, dataset_name: str) -> Settings:
    return Settings(
        drive_folder=settings.drive_folder,
        bundle_name=settings.bundle_name,
        checkpoint_name=settings.checkpoint_name,
        work_dir=settings.work_dir,
        mountpoint=settings.mountpoint,
        dataset_name=dataset_name,
    )


def inspect_dataset(settings: Settings) -> None:
    _ensure_layout(settings)
    _materialise_dataset(settings)
    _run([
        sys.executable,
        str(settings.work_dir / "trainer" / "tetra_dataset.py"),
        str(settings.local_dataset_path),
    ], cwd=settings.work_dir)


def train(settings: Settings, args: argparse.Namespace) -> None:
    _ensure_layout(settings)
    _materialise_dataset(settings)
    validate(settings, require_complete=True)

    if args.condition == "aux":
        aux_weight = 0.1 if args.aux_weight is None else args.aux_weight
    else:
        aux_weight = 0.0 if args.aux_weight is None else args.aux_weight
    if aux_weight < 0.0:
        raise ValueError("--aux-weight must be non-negative")

    result_dir = settings.work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.condition}_seed{args.seed}"
    final_path = result_dir / f"{prefix}.pt"
    best_path = result_dir / f"{prefix}.best.pt"
    if not args.overwrite and (final_path.exists() or best_path.exists()):
        raise FileExistsError(
            f"training output exists: {final_path} or {best_path}; rerun with --overwrite"
        )

    command = [
        sys.executable,
        str(settings.work_dir / "trainer" / "train.py"),
        str(settings.local_dataset_path),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--batch",
        str(args.batch),
        "--model",
        args.model,
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--aux-weight",
        str(aux_weight),
        "--eval-every",
        str(args.eval_every),
        "--save",
        str(final_path),
        "--best-save",
        str(best_path),
    ]
    if args.device.startswith("cuda"):
        command.append("--require-gpu")
    _run(command, cwd=settings.work_dir)
    _copy_to_drive(settings, (final_path, best_path))
    print(f"condition    {args.condition}")
    print(f"aux weight   {aux_weight:g}")
    print(f"seed         {args.seed}")
    print("train        OK")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--mountpoint", default=DEFAULT_MOUNTPOINT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="mount Drive, extract the bundle and build tetra_cli")
    add_common_arguments(setup_parser)

    generate_parser = subparsers.add_parser("generate", help="generate one GPU self-play shard and upload it")
    add_common_arguments(generate_parser)
    generate_parser.add_argument("--output-name", default=DEFAULT_DATASET_NAME)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--games", type=int, default=32)
    generate_parser.add_argument("--pieces", type=int, default=200)
    generate_parser.add_argument("--sims", type=int, default=32)
    generate_parser.add_argument("--inference-batch", type=int, default=16)
    generate_parser.add_argument("--base-seed", type=int, default=2026080600)
    generate_parser.add_argument("--shard-id", type=int, default=0)
    generate_parser.add_argument("--shard-count", type=int, default=1)
    generate_parser.add_argument("--model-version", type=int, default=5)
    generate_parser.add_argument("--determinizations", type=int, default=2)
    generate_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    generate_parser.add_argument("--no-gumbel", dest="gumbel", action="store_false")
    generate_parser.set_defaults(gumbel=True)
    generate_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate the local or Drive-copied shard")
    add_common_arguments(validate_parser)
    validate_parser.add_argument("--allow-incomplete", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="run the dataset sanity check")
    add_common_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="run one paired ablation condition")
    add_common_arguments(train_parser)
    train_parser.add_argument("--condition", choices=("aux", "noaux"), required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--steps", type=int, default=800)
    train_parser.add_argument("--batch", type=int, default=64)
    train_parser.add_argument("--model", choices=("dev", "s"), default="dev")
    train_parser.add_argument("--threads", type=int, default=2)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--eval-every", type=int, default=200)
    train_parser.add_argument("--aux-weight", type=float, default=None)
    train_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    try:
        if args.command == "setup":
            setup(settings)
        elif args.command == "generate":
            generate(settings, args)
        elif args.command == "validate":
            validate(settings, require_complete=not args.allow_incomplete)
        elif args.command == "inspect":
            inspect_dataset(settings)
        elif args.command == "train":
            train(settings, args)
        else:  # pragma: no cover - argparse makes this unreachable
            raise RuntimeError(f"unknown command: {args.command}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: command exited with status {exc.returncode}", file=sys.stderr)
        print("pass the complete Colab output back for diagnosis", file=sys.stderr)
        return int(exc.returncode or 1)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
