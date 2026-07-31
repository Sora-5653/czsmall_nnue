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

- The engine builds and its 263 tests pass with no dependencies at all.
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

Without them, `make test` skips the two PyTorch-parity tests and the other 261
still run.

```sh
git clone <this repo> && cd czsmall_nnue

# 1. Build and verify the engine (no Python needed).
make test

# 2. Generate a training set from self-play.
./build/tetra_cli export data/train.tetradat 50 200 32

# 3. Train. Add --device cuda to use the GPU (ROCm reports as "cuda").
python trainer/train.py data/train.tetradat --steps 2000 --model s \
    --device cuda --save models/gen1.pt

# 4. Export the weights for the C++ engine.
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts

# 5. Play with them.
./build/tetra_cli play models/gen1.tetrawts 200 64
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

# Train on it
python trainer/train.py data/gen$N.tetradat --steps 5000 --model s \
    --device cuda --save models/gen$N.pt

# Export and play
python trainer/export_weights.py models/gen$N.pt models/gen$N.tetrawts
./build/tetra_cli play models/gen$N.tetrawts
```

**A limitation to be aware of.** `tetra_cli export` currently self-plays with
the built-in `HeuristicEvaluator`, not with your trained weights, so the loop
above is *supervised bootstrapping* rather than true self-play iteration.
Wiring `--weights` into `export` is a small change and is the first thing worth
doing on your machine; see `docs/ROADMAP.md`.

---

## 5. Sizing

| model | parameters | CPU forward | notes |
|---|---|---|---|
| `--model dev` | 0.13 M | ~1 ms | fast iteration, sanity checks |
| `--model s` | 7.2 M | ~8.6 ms | spec §9.5 TetraFormer-S |

On a 2-core CPU the dev model trains at ~25 steps/s. Your GPU should manage the
`s` model comfortably; start with a batch size of 256 and raise it until VRAM
(16 GB) complains.

The C++ inference path is scalar CPU code, so a spec-sized model is slow for
*self-play generation*. Options, in order of effort: keep generating with the
`dev` model, run generation on many CPU cores in parallel, or add an ONNX
Runtime backend behind the same `Evaluator` interface.

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `torch.cuda.is_available()` is `False` | ROCm < 7.2, wrong wheel index, or missing `render`/`video` group |
| `HIP error: invalid device function` | architecture mismatch — set `PYTORCH_ROCM_ARCH=gfx1201` |
| `feature width mismatch` when loading weights | the engine's token layout changed after the model was trained; retrain or check out the matching commit |
| `make test` fails on `cpp_matches_pytorch_exactly` | the C++ forward pass and the fixture disagree — do not train until this is resolved |
| out of memory during training | lower `--batch`; the `s` model at batch 256 needs roughly 6-8 GB |