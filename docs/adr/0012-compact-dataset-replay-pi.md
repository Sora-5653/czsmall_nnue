# ADR 0012: Compact dataset format (Replay + π) and on-the-fly token regeneration

## Status
Accepted.

## Context
ADR 0010 established a single rectangular, padded layout (`TensorBatch` in `batch.hpp`) shared across C++ inference, on-disk dataset files (`.tetradat`), and the PyTorch trainer.

In `DatasetHeader::VERSION = 1`, `.tetradat` files stored the fully padded float32 tensors for every sample:
- `tokens`: `[max_tokens, TOKEN_FEATURES]` (80 * 24 * 4 = 7.68 KB per sample)
- `actions`: `[max_actions, ACTION_FEATURES]` (40 * 24 * 4 = 3.84 KB per sample)
- Together with masks and auxiliary targets, each sample consumed ~12.2 KB on disk.

For serious self-play runs (100,000+ samples), `.tetradat` files quickly exceeded 1 GB, turning disk capacity and file I/O into the primary bottleneck before GPU compute could be utilized.

## Decision
Introduce `DatasetHeader::VERSION_COMPACT = 2` for `.tetradat` serialization (`include/tetra/dataset.hpp`).

Instead of storing pre-generated tokens and action embeddings on disk, Version 2 stores **Replay + π**:
1. **Provenance:** `ruleset_hash`, `game_seed`, `move_number`, `chosen_action`, and garbage schedule parameters (`garbage_style`, `garbage_period`, `garbage_lines`).
2. **Search targets:** `search_policy` vector, `outcome`, `search_value`, and auxiliary targets (`time_to_terminal`, `future_attack_1s`, `future_garbage_received`, `topped_out_within_4`, `topped_out_within_8`).

### On-the-fly regeneration during load
When reading a `.tetradat` file:
- **In C++ (`deserialize_dataset`):** If `version == 2`, `deserialize_compact_dataset` groups records by `(ruleset_hash, game_seed)` and simulates each game in memory from move 0 using `Player`, `MoveGenerator`, and `Tokenizer`. At each recorded placement, it regenerates the exact tokenized observation and legal action embeddings, yielding a bit-for-bit identical `TensorBatch`.
- **In Python (`trainer/tetra_dataset.py`):** `load(path)` detects version 2 and invokes the engine tool `./build/tetra_cli decode-dataset <path>` via subprocess (or reads directly if version 1), streaming the unpadded binary tensors into numpy without intermediate disk writes.

## Consequences
- **Disk and I/O reduction:** Serialized size drops from ~12,200 bytes/sample to ~95 bytes/sample (over 100x reduction). A 100,000-sample dataset shrinks from ~1.2 GB to ~10 MB.
- **CPU regeneration overhead:** Simulating moves and tokenization in C++ runs at ~30,000 placements/second, adding less than 1 second of CPU time to load 30,000 samples.
- **Backward compatibility:** Both Version 1 and Version 2 are supported by `deserialize_dataset` and `trainer/tetra_dataset.py`. Existing tests and legacy files continue to work without modification.
