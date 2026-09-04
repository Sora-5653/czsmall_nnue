# Phase 0 preflight record

この記録は、Phase 0開始前の読み取り専用確認をまとめたものです。正確なdirty status、checkpoint hash、GPU gate、runnerの実行ログは生成manifest/recordを正とします。

## 指示書

- `C:\Users\eddyf\czsmall_nnue\AGENTS.md` をUTF-8全文で読了。
- `C:\Users\eddyf\czsmall_nnue\experiments\search_throughput_vs_capacity_2026-08-18\CAPACITY_QUALITY_HANDOFF.md` をUTF-8全文で読了。
- 現在のworking treeを使い、worktree分離はしていない。
- 書き込み先はこの `capacity_quality/` 配下だけ。

## Git

実測値は `manifests/checkpoints.json` の `evaluated_at_commit` と `worktree_status` に保存します。preflight時点では branch `codex/sample_eff_fix`、HEAD `04d49ffee95f1c937c07ab60e88a868b6eb61ef6` でした。既存の `.gitignore`, `Makefile`, `README.md`, `include/tetra/*`, `tools/tetra_cli.cpp`, `trainer/*` の変更と多数のuntracked成果物を保持し、reset/clean/stash/checkout/add/commitはしていません。

## GPU gate

使用runtimeは `.venv-rocm714\Scripts\python.exe`、PyTorch `2.12.0+rocm7.14.0`、HIP `7.14.60850` です。`torch.cuda.is_available()` は `True`、device countは2、device 1は `AMD Radeon RX 9070 XT`、`gfx1201` です。device 0は統合GPUのため、実験は `cuda:1` 固定です。Phase 0 wrapperは `--require-gpu` を必須にし、GPU不在時にCPUへフォールバックしません。

なお、AGENTS.mdの一般例はLinux/WSLでのROCm環境を示しますが、この既存実験の検証済みruntime metadataはWindows側のROCm venvと `build/tetra_cli.exe` です。今回も同じ検証済み組み合わせを使い、実測runtimeをmanifestへ明記します。

## engine / tests

- `build/tetra_cli.exe --help` で `gpu-arena-protocol` と引数列を確認。
- `build/tetra_tests.exe`: 292 tests, 1,055,146 assertions, 0 failed。
- `build/tetra_reanalyse_tests.exe`: 3 tests, 17 assertions, 0 failed。
- C++ source `include/tetra/arena.hpp` の `Arena::evaluate` は `pair_seed = base_seed + i * 0x9E3779B97F4A7C15`、各pairは normal/mirror × role-swap の4局であることを確認。

## runner

- `experiments/search_throughput_vs_capacity_2026-08-18/scripts/run_condition.py --help` をROCm venvで確認。
- `candidate_sims=0` / `champion_sims=0` は help、既存 `A0-B0-raw-policy` JSON、C++ `ArenaConfig` と一致。
- `candidate_node_budget=64` / `champion_node_budget=64` は help、既存 `D-equal-node64` JSON、C++ `ArenaConfig` と一致。
- runnerのstderr `drain`、variable-shape `fixed_token_count=0`, `fixed_action_count=0`、batch prewarmは既存完走pilotと同じ安定条件。
- runnerは `--require-gpu` を直接公開しないため、capacity_quality/scripts/run_phase0.pyでmandatory preflightを追加する。trainer、engine、modelsは変更しない。

## checkpoint / provenance

checkpointごとのsha256、architecture、parameter数、checkpoint step、seed、validation、training dataset provenance、unknown項目は `manifests/checkpoints.json` を参照。S400/M400/XS400は `models/size_search_ablation_20260816/seed42/train_results.json` のGen4 dataset metadataを参照するが、独立dataset manifest hashとtraining commitは埋め込まれていないためunknownとする。Aはcheckpointにvalidation/dataset/seed metadataがないためunknownを維持する。
