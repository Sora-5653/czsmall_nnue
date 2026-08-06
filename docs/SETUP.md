# Setup: training on an RX 9070 XT

What you need to install, and what is already done for you.

The engine itself has **no dependencies** — a C++17 compiler is enough, and the
whole test suite runs without Python or a GPU. Everything below is only needed
for *training*.

---

## 1. What you need to provide

### Hardware / OS

Your RX 9070 XT is **RDNA 4, LLVM target `gfx1201`**. This matters because
support is version-sensitive:

| | requirement |
|---|---|
| ROCm | **7.2 or newer** — the first release with official `gfx1201` support |
| OS | Linux (Ubuntu 22.04 / 24.04 recommended), or Windows via WSL2 |
| Python | 3.10–3.12 |
| Compiler | g++ ≥ 9 or clang++ ≥ 10 (C++17) |
| Disk | ~10 GB for ROCm + PyTorch |

ROCm 6.4.x also lists `gfx1201`, but 7.2 is the release where AMD declared
RDNA 4 properly supported; prefer it unless you hit a regression.

### Install ROCm

Follow AMD's installer for your distribution
(<https://rocm.docs.amd.com/projects/install-on-linux/>). Afterwards:

```sh
rocminfo | grep gfx        # must print gfx1201
```

Add yourself to the required groups and re-login if `rocminfo` fails:

```sh
sudo usermod -aG render,video $USER
```

### Install PyTorch for ROCm

**Not** the default PyPI wheel — that one is CPU/CUDA only:

```sh
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install -r trainer/requirements.txt
```

Verify the GPU is actually visible. ROCm PyTorch reports itself through the
`cuda` API, which is expected:

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True  AMD Radeon RX 9070 XT
```

If it prints `False`, the usual causes are: ROCm older than 7.2, a missing
`render`/`video` group membership, or a PyTorch wheel from the wrong index.

**Known RDNA 4 workaround.** Some tools still mis-detect `gfx1201`. If you hit
architecture-detection errors, set:

```sh
export PYTORCH_ROCM_ARCH=gfx1201
export HSA_OVERRIDE_GFX_VERSION=12.0.1   # only if detection still fails
```

---

## 2. What is already done

- The engine builds and its 272 tests pass with no dependencies at all.
- Self-play, search, the replay buffer and dataset export are complete.
- `trainer/` contains the network, the dataset reader and a training loop.
- The C++ inference path is **numerically verified against PyTorch** to ~1e-7
  (`cpp_matches_pytorch_exactly`), so what you train is what the engine plays.
- `scripts/bootstrap.sh` runs the whole loop end to end.

---

## 3. First run

If you reconstructed this tree from a text dump, the two binary test fixtures
in `tests/data/` will be missing. They are fully reproducible from seeds:

```sh
python scripts/make_fixtures.py   # needs torch; regenerates them byte-for-byte
```

Without them, `make test` skips the two PyTorch-parity tests and the other 270
still run.

```sh
git clone <this repo> && cd czsmall_nnue

# 1. Build and verify the engine (no Python needed).
make test

# 2. Generate a training set from self-play.
./build/tetra_cli export data/train.tetradat 50 200 32

# 3. Train. Add --device cuda to use the GPU (ROCm reports as "cuda").
python trainer/train.py data/train.tetradat --steps 2000 --model s \
    --device cuda --require-gpu --save models/gen1.pt

# 4. Export the weights for the C++ engine.
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts

# 5. Play with them.
./build/tetra_cli play models/gen1.tetrawts 200 64

# GPU-backed C++ search/inference with a Tetr.io-style APM/APP/PPS report.
python trainer/gpu_match.py models/gen1.pt --device cuda --games 4 \
    --pieces 200 --sims 32 --precision fp16 --workers 4

# GPU-backed paired Arena (the C++ simulator remains authoritative).
python trainer/gpu_arena.py models/candidate.pt models/champion.pt --device cuda \
    --pairs 20 --pieces 300 --sims 32 --precision fp16 --seed 42

# GPU-backed self-play data generation for the next training generation.
python trainer/gpu_selfplay.py models/gen1.pt data/gen2.tetradat \
    --device cuda --games 32 --pieces 300 --sims 64 --model-version 2

# One guarded generation: the candidate is promoted only if Arena passes.
python trainer/iterate.py --champion models/champion.pt \
    --replay data/gen1.tetradat --generation 2 \
    --champion-output models/champion --device cuda
```

Or simply:

```sh
./scripts/bootstrap.sh
```

---

## 4. The training loop

AlphaZero-style training is iterative. One generation is:

```sh
# Self-play with the current best weights -> new data
./build/tetra_cli export data/gen$N.tetradat 200 300 64

# Train on it; --resume also restores the optimizer and sampling RNG.
python trainer/train.py data/gen$N.tetradat --steps 5000 --model s \
    --device cuda --require-gpu --checkpoint-every 1000 \
    --value-weight 1.0 \
    --best-save models/gen$N.best.pt --save models/gen$N.pt

# Export and play
python trainer/export_weights.py models/gen$N.pt models/gen$N.tetrawts
python trainer/export_weights.py models/gen$N.best.pt models/gen$N.best.tetrawts
./build/tetra_cli play models/gen$N.tetrawts
```

`tetra_cli export` accepts `--weights` when the current model should drive the
next generation, for example:

```sh
./build/tetra_cli export data/gen$N.tetradat 200 300 64 \
    --weights models/gen$N.best.tetrawts
```

The exported data is still local simulator data; the project never connects to
TETR.IO.

For larger generations, `trainer/gpu_selfplay.py` keeps the same C++ rules and
search but serves network evaluations from PyTorch/ROCm. It writes a
rectangular v1 dataset from the C++ side: two-board self-play records both
players in chronological order, with each value outcome converted to the
recorded player's perspective. Because either player's observation includes
the opponent event stream, compact replay metadata would not be sufficient to
reconstruct these observations. The script also reports the number of
GPU-evaluated positions.

### Colab position-generation plan

Colab is reserved for generating additional self-play positions. The local
machine remains the authority for checkpoint promotion and Arena comparison.
The planned workflow is:

1. Every Colab instance checks out the same commit and loads the same
   checkpoint.
2. Instance `shard_id` runs `trainer/gpu_selfplay.py` with
   `--seed base_seed + shard_id * games_per_shard`; its games then consume the
   consecutive seeds in that shard only.
3. The instance returns its `.tetradat` file and a manifest containing the
   commit, checkpoint hash, ruleset/model/search settings, seed interval and
   sample count.
4. The local trainer validates the manifests and supplies the shard files as
   separate inputs, for example:

```sh
python trainer/train.py data/colab-shard-0.tetradat \
    data/colab-shard-1.tetradat data/local.tetradat \
    --resume models/champion.pt --device cuda --require-gpu \
    --value-weight 1.0 --steps 5000 --save models/candidate.pt
```

The notebook and manifest validator are intentionally a later stage; the
existing GPU self-play bridge is the execution core.

`trainer/train.py` accepts multiple dataset paths. The last path is the new
generation. Keep `--new-data-repeat` at 1 for an unbiased first trial; higher
values deliberately oversample the newest generation, for example:

```sh
python trainer/train.py data/gen1.tetradat data/gen2.tetradat \
    --resume models/champion.pt --new-data-repeat 1 \
    --device cuda --require-gpu --value-weight 1.0 \
    --steps 5000 --save models/gen2.pt
```

`trainer/iterate.py` runs this sequence automatically and uses the GPU Arena
by default. It copies the candidate to `--champion-output` only when the Arena
promotion threshold is met; pass `--cpu-arena` for the legacy CPU evaluator.

GPU play defaults to fp16 inference, one root determinization, and batched
PUCT to keep latency bounded. Use `--precision fp32`, `--determinizations 2`,
and `--gumbel` when reproducing the higher-exploration self-play configuration.

The WDL value head is trained together with the policy head. The default loss
weights are `policy=1.0`, `value=1.0`, and `aux=0.1`; they are stored in the
checkpoint and restored on `--resume`. Use `--value-weight 0` only for a
deliberate policy-only ablation. For a conservative policy experiment,
`--policy-head-only --reset-optimizer` freezes the shared trunk and value/aux
heads. Training output also reports value accuracy and scalar value MSE so a
run cannot look healthy from policy loss alone.

---

## 5. Sizing

| model | parameters | CPU forward | notes |
|---|---|---|---|
| `--model dev` | 0.13 M | ~1 ms | fast iteration, sanity checks |
| `--model s` | 7.2 M | ~8.6 ms | spec §9.5 TetraFormer-S |

On a 2-core CPU the dev model trains at ~25 steps/s. Your GPU should manage the
`s` model comfortably; start with a batch size of 256 and raise it until VRAM
(16 GB) complains.

`tetra_cli play` remains a dependency-free scalar CPU path. For GPU inference
inside the C++ search loop, use `trainer/gpu_match.py`: it launches the C++
rules/search child and answers its batched evaluator frames with PyTorch/ROCm.
The script prints PPS, APM and APP; APM is outgoing attack lines per minute and
APP is outgoing attack lines per placed piece.

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `torch.cuda.is_available()` is `False` | ROCm < 7.2, wrong wheel index, or missing `render`/`video` group |
| `HIP error: invalid device function` | architecture mismatch — set `PYTORCH_ROCM_ARCH=gfx1201` |
| `feature width mismatch` when loading weights | the engine's token layout changed after the model was trained; retrain or check out the matching commit |
| `make test` fails on `cpp_matches_pytorch_exactly` | the C++ forward pass and the fixture disagree — do not train until this is resolved |
| out of memory during training | lower `--batch`; the `s` model at batch 256 needs roughly 6-8 GB |
