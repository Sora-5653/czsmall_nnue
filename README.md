# TetraZero / TetraFormer

An exact, deterministic Tetris rule core, a Leela-style search, and a
Transformer policy/value network trained on its own self-play.

See [`docs/SPEC.md`](docs/SPEC.md) for the full design.

**This is a local simulator only.** It has no network code and cannot connect
to TETR.IO. Please read [`docs/POLICY.md`](docs/POLICY.md) first.

**Status:** M0 (rule core), M1 (movegen, tokenizer) and M2 (search, self-play,
training loop) are built and tested. What remains is documented in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quick start

The engine needs only a C++17 compiler — no third-party dependencies, no GPU,
no Python.

```sh
make test         # 263 tests
make tools        # the developer CLI
make test-asan    # the suite under AddressSanitizer + UBSan
```

On Windows the Makefile recipes assume a POSIX shell, so either build inside
WSL2 (recommended — see docs/SETUP.md), or invoke a native g++ (MinGW-w64 /
MSYS2) directly. The trainer finds both `build/tetra_cli` and
`build/tetra_cli.exe`:

```powershell
mkdir build -Force
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude tools\tetra_cli.cpp src\ruleset.cpp -o build\tetra_cli.exe
```

The whole training pipeline, end to end:

```sh
./scripts/bootstrap.sh    # build, test, self-play, train, export weights, play
```

Individual steps:

```sh
./build/tetra_cli ruleset league                    # a ruleset and its hash
./build/tetra_cli moves T 7                         # enumerate legal placements
./build/tetra_cli timing T                          # action costs and delay bins
./build/tetra_cli search 64 1 42                    # inspect one search
./build/tetra_cli record g.tetrarep 42 300          # record a game
./build/tetra_cli verify g.tetrarep                 # re-simulate and diff it
./build/tetra_cli export train.tetradat 10 100 16   # export a training set
./build/tetra_cli play models/gen1.tetrawts         # play with trained weights
./build/tetra_cli determinism 42                    # verify reproducibility
./build/tetra_cli bench 20000                       # movegen throughput

python trainer/train.py train.tetradat --steps 300 --save models/gen1.pt
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts
```

Training on an AMD GPU needs ROCm 7.2 or newer — the first release with
official `gfx1201` / RDNA 4 support. See [`docs/SETUP.md`](docs/SETUP.md).

## What is implemented

### M0 — rule core

| Area | Notes |
|---|---|
| Playfield | Bitboard rows, separate occupancy and garbage planes |
| Pieces | All 7 tetrominoes, 4 rotation states, precomputed collision masks |
| Rotation | Guideline SRS, **TETR.IO SRS+** (symmetric I kicks), TETR.IO 180 table |
| Spins | T-spin 3-corner, All-Mini, **All-Mini+**, 4-direction immobile check |
| Attack | `floor((base + b2b) × combo_mult)` + flat bonuses, matching the published TETR.IO table |
| Combo | Multiplier system, plus `floor(ln(1 + 1.25c))` for zero-base clears |
| B2B | Charging (flat +1 with Surge) and Chaining (osk's step table) |
| Surge | Charges from streak 4, releases in 3 segments with remainder carried |
| Garbage | Travel time, activation delay, FIFO cancellation, cap, messiness |
| Opener phase | Double cancellation for the first 14 placements |
| Rulesets | Versioned `RulesetConfig` with a stable hash; league / quickplay / guideline |
| Time | Integer ticks only — no floating-point clocks anywhere |
| Handling | DAS / ARR / SDF / ARE / lock delay priced into every action |
| Gravity | Enforced as a reachability constraint; exact rational, integer-only |
| Replays | Versioned binary format, checkpointed verification, ~19 B/placement |

### M1 — movegen and model inputs

- **Legal placements by Dijkstra** over real movement primitives, so kicks,
  spins, tucks and wall-climbs are *discovered*, not enumerated by hand.
- **Real action durations** (spec §8.4): each placement priced in ticks from the
  ruleset's handling settings, by shortest path, verified optimal against
  brute-force enumeration.
- **Delay bins** `FASTEST / +1F / +2F / +4F / +8F / WAIT_FOR_EVENT`, the last
  bounded by garbage activation, the opponent's next lock and the lock delay —
  never open-ended.
- Every action carries a **canonical input sequence**, and a test replays each
  one to confirm it lands on exactly the promised cells.
- **Observation masking** so hidden state cannot reach a policy.
- **Row/Column tokenizer** (spec §9.2): `H` row + `W` column + 1 summary token
  per board instead of ~400 cell tokens.
- **Variable-length action embeddings** (spec §10.1).

### M2 — search, self-play and training

- **Batched `Evaluator` interface.** Written *before* the search, because
  measurement showed the network dominates everything else: ~8.6 ms per position
  for a spec-sized TetraFormer-S against ~0.09 ms for move generation. Batching
  is therefore structural, not an optimisation
  ([ADR 0007](docs/adr/0007-evaluator-interface-first.md)).
- **PUCT and Gumbel sequential halving** with virtual loss and a transposition
  table. Gumbel is the default: PUCT is unreliable below roughly two simulations
  per legal action ([ADR 0008](docs/adr/0008-search-gumbel-calibration.md)).
- **Root determinization** (chance nodes, spec §11.3), which closed an
  information leak that let the search read pieces past the preview
  ([ADR 0009](docs/adr/0009-determinization-and-selfplay.md)).
- **Self-play worker**, replay buffer and `.tetradat` dataset export.
- **The TetraFormer itself** (`trainer/`): pre-norm RMSNorm blocks with SwiGLU,
  a variable-length policy head where each legal action cross-attends to the
  state tokens, a WDL value head and auxiliary regressions
  ([ADR 0010](docs/adr/0010-cpp-python-handover.md)).
- **C++ inference for trained weights** with no third-party runtime, verified
  against PyTorch to a maximum difference of **3e-08**, so the engine provably
  plays the network that was trained
  ([ADR 0011](docs/adr/0011-cpp-inference-without-onnx.md)).

Measured: under a garbage stream, Gumbel-32 reaches 250/250 placements against
228.7 for policy-only, and every configuration fits the spec §19.4 latency
budget with the heuristic evaluator. Held-out training loss falls 4.86 → 2.91 on
engine-generated data.

The model is **not yet strong**: it is bootstrapped from heuristic-guided
self-play on a CPU. True AlphaZero iteration needs `tetra_cli export` to
generate with trained weights rather than the heuristic — a small change, and
the first thing worth doing on a GPU machine.

## Verification

**263 tests, ~766k assertions**, clean under `-Werror`, AddressSanitizer and
UBSan. Highlights (spec §18):

- **Determinism** — the same seed replays bit-identically, including with
  garbage; RNG state can be snapshotted and restored.
- **Reflection equivalence** — rotation outcomes and the *entire* generated
  placement set are mirror-invariant, checked cell for cell on random boards.
- **Information leaks** — tokenizing a state with a different hidden garbage
  hole column must produce byte-identical tokens, and the search must not depend
  on pieces beyond the preview.
- **Attack conservation** — everything sent is either cancelled or lands.
- **PyTorch parity** — the C++ and Python forward passes must agree.

Two binary fixtures in `tests/data/` back the parity test. They are reproducible
rather than precious:

```sh
python scripts/make_fixtures.py   # regenerates them byte-for-byte
```

Without them those two tests skip and the other 261 still run.

### Rule findings worth knowing

These fell out of the property tests and are pinned by dedicated cases, because
they constrain how training may augment data (spec §14):

1. **SRS+ I-kicks are *exactly* mirror-symmetric**, entry for entry — the
   defining difference from guideline SRS.
2. **Classic SRS is not.** The I piece's kick *order* differs between the two
   sides, so left/right mirroring is **not** a sound augmentation for I
   placements under guideline SRS. Every other piece mirrors cleanly.
3. **Immobile spin detection needs the up-check.** TETR.IO defines "immobile" as
   unable to move left, right, **up** or down. Omitting the up-check makes any
   piece in a flat notch score a spin: in a 300-piece game that inflated the
   count from 16 to 198, poisoning both attack values and the training reward.
4. **Identical cells, different coordinates.** An I piece in rotation N at row
   `y` fills exactly the cells of rotation 2 at row `y+1`, so the two merge as
   one outcome. Merge logic must replace the *whole* execution, not just the
   input sequence, or an action ends up describing one representation while its
   inputs produce the other.
5. **TETR.IO's 180 kick table is deliberately asymmetric** (downward R↔L kicks
   but no upward ones, which is also why it cannot be written as SRS offset
   data). With 180 enabled a few positions are not mirror-equivalent;
   `movegen_mirror_asymmetry_comes_only_from_180` pins the scope exactly.

## Performance

Movegen is the search's inner loop, so it was profiled and optimised — 2.4×
faster than the first working version, with byte-identical output:

| | µs / call |
|---|---|
| Initial implementation | 279 |
| Parent-linked BFS paths | 163 |
| Flat generation-stamped visited table | 124 |
| Reused scratch buffers | **117** |
| *+ exact shortest-path action pricing (ADR 0004)* | *155* |

Measured on a 2-core machine with a messy 8-row board, averaged over 20k calls.
The last row is a deliberate trade: exact pricing costs ~38 µs and is what makes
timing decisions representable at all.

## Continuous integration

A ready-to-use workflow sits at [`docs/ci.yml`](docs/ci.yml) — it builds with
g++ and clang++, runs the sanitizers, and enforces `-Werror` and determinism. It
is not at `.github/workflows/` because the account that created this branch
could not add workflow files. To enable it:

```sh
mkdir -p .github/workflows && git mv docs/ci.yml .github/workflows/ci.yml
```

## Layout

```
include/tetra/
  types.hpp         pieces, rotations, clear descriptors, tick type
  bitboard.hpp      playfield with occupancy + garbage planes
  pieces.hpp        shapes and SRS / SRS+ / 180 kick tables
  ruleset.hpp       versioned RulesetConfig (spec 6)
  rng.hpp           xoshiro256** + 7-bag randomizer
  piece_state.hpp   collision, rotation with kicks, spin detection
  attack.hpp        attack table, combo, B2B, Surge
  garbage.hpp       pending queue, cancellation, activation
  events.hpp        bounded event log (spec 7.3)
  player.hpp        one player's simulation
  timing.hpp        handling model, action cost, gravity, delay bins (spec 8.4)
  movegen.hpp       Dijkstra legal placement generation (spec 8)
  observation.hpp   the observation mask (spec 5.1, 18.3)
  tokenizer.hpp     Row/Column tokens + action embeddings (spec 9, 10.1)
  replay.hpp        replay format, recorder, verifier (spec 17, 22)
  evaluator.hpp     batched policy/value interface + baselines (spec 10, 11.1)
  search.hpp        PUCT / Gumbel search with virtual loss and a TT (spec 11)
  replay_buffer.hpp training samples and the replay buffer (spec 13.5)
  selfplay.hpp      self-play worker with garbage curricula (spec 13.3)
  batch.hpp         fixed-shape padded tensors + masks (spec 9, 10.1)
  dataset.hpp       .tetradat export for the trainer (spec 13.5, 17)
  nnue.hpp          C++ inference for trained weights (spec 16, 17)
src/                ruleset presets and hashing
tests/              263 tests, dependency-free harness
tools/              tetra_cli developer tool
trainer/
  tetraformer.py    the spec 9-10 network
  tetra_dataset.py  .tetradat reader with validation
  train.py          a small, readable training loop
  export_weights.py PyTorch checkpoint -> .tetrawts for the engine
scripts/
  bootstrap.sh      the whole pipeline in one command
  make_fixtures.py  regenerate the binary test fixtures
docs/               spec, setup, policy, roadmap, 11 ADRs
```