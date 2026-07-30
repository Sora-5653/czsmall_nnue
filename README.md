# TetraZero / TetraFormer — M0 + M1

An exact, deterministic Tetris rule core and legal-move generator, built as the
foundation for a Leela-style self-play Transformer bot (see
[`docs/SPEC.md`](docs/SPEC.md) for the full design).

**This is a local simulator only.** It has no network code and cannot connect to
TETR.IO. Please read [`docs/POLICY.md`](docs/POLICY.md) before doing anything
with it.

Current status: **M0 (rule core) and M1 (movegen + tokenizer) are complete and
tested.** M2–M5 (search, self-play, training) are specified but not yet built.

## Quick start

Requires only a C++17 compiler. No third-party dependencies.

```sh
make test        # build and run the full test suite
make tools       # build the developer CLI
make test-asan   # run the suite under AddressSanitizer + UBSan

./build/tetra_cli ruleset league      # print a ruleset and its hash
./build/tetra_cli moves T 7           # enumerate legal T placements
./build/tetra_cli selfplay 42 300     # scripted greedy game
./build/tetra_cli determinism 42      # verify reproducibility
./build/tetra_cli bench 20000         # movegen throughput
```

## What is implemented

### M0 — rule core

| Area | Notes |
|---|---|
| Playfield | Bitboard rows, separate occupancy and garbage planes |
| Pieces | All 7 tetrominoes, 4 rotation states, precomputed collision masks |
| Rotation | Guideline SRS, **TETR.IO SRS+** (symmetric I kicks), TETR.IO 180 table |
| Spins | T-spin 3-corner, All-Mini, **All-Mini+** (immobile T), 4-direction immobile check, kick-aware mini/full |
| Attack | `floor((base + b2b) × combo_mult)` + flat bonuses, matching the published TETR.IO table |
| Combo | Multiplier system, plus `floor(ln(1 + 1.25c))` for zero-base clears |
| B2B | Charging (flat +1 with Surge) and Chaining (osk's step table) |
| Surge | Charges from streak 4, releases in 3 segments with remainder carried |
| Garbage | Travel time, activation delay, FIFO cancellation, cap, messiness, hole rules |
| Opener phase | Double cancellation for the first 14 placements |
| Rulesets | Fully versioned `RulesetConfig` with a stable `ruleset_hash`; league / quickplay / guideline presets |
| Time | Integer ticks only — no floating-point clocks anywhere |

### M1 — movegen, observation, tokenizer

- **Legal placement generation** by BFS over real movement primitives, so
  kicks, spins, tucks and wall-climbs are *discovered*, not enumerated by hand.
- Every action carries a **canonical input sequence**; a test replays each one
  and asserts it lands on exactly the promised cells.
- **Path-equivalence merging** per spec §8.3 (same board, clears, spin class,
  attack).
- **Observation masking**: hidden state cannot reach a policy.
- **Row/Column tokenizer** (spec §9.2): `H` row + `W` column + 1 summary token
  per board instead of ~400 cell tokens.
- **Variable-length action embeddings** (spec §10.1).

## Verification

The test suite is the point of this milestone: **118 tests, ~499k assertions**,
clean under `-Werror`, AddressSanitizer and UBSan.

Highlights (spec §18):

- **Determinism** — the same seed replays bit-identically, including with
  garbage; RNG state can be snapshotted and restored.
- **Reflection equivalence** — rotation outcomes and the *entire generated
  placement set* are mirror-invariant, checked cell for cell on random boards.
- **Information leaks** — tokenizing a state with a different hidden garbage
  hole column must produce byte-identical tokens.
- **Attack conservation** — everything sent is either cancelled or lands.
- **Bag constraints, non-negative garbage, replay reproduction.**

### Rule findings worth knowing

Three properties fell out of the property tests and are now pinned by
dedicated cases, because they affect how the training pipeline may augment
data (spec §14):

1. **SRS+ I-kicks are *exactly* mirror-symmetric** — entry for entry, which is
   the defining difference from guideline SRS.
2. **Classic SRS is not.** Under guideline SRS the I piece's kick *order*
   differs between the two sides, so left/right mirroring is **not** a sound
   augmentation for I placements. Every other piece mirrors cleanly.
3. **Immobile spin detection needs the up-check.** TETR.IO defines "immobile"
   as unable to move left, right, **up** or down. Omitting the up-check makes
   any piece sitting in a flat notch score a spin: in a 300-piece scripted
   game that inflated the spin count from 16 to 198, which would have poisoned
   both the attack values and the training reward. `is_immobile()` checks all
   four directions and `immobile_requires_all_four_directions_blocked` pins it.
4. **TETR.IO's 180 kick table is deliberately asymmetric** (it has downward
   R↔L kicks but no upward ones, which is also why it cannot be written as SRS
   offset data). With 180 enabled, a small number of positions are not
   mirror-equivalent. `movegen_mirror_asymmetry_comes_only_from_180` pins this
   scope exactly.

## Performance

Movegen is the MCTS inner loop, so it was profiled and optimised (2.4× faster
than the first working version, with byte-identical output):

| | µs / call |
|---|---|
| Initial implementation | 279 |
| Parent-linked BFS paths | 163 |
| Flat generation-stamped visited table | 124 |
| Reused scratch buffers | **117** |

Measured on a 2-core sandbox with a messy 8-row board, averaged over 20k calls.

## Continuous integration

A ready-to-use workflow is provided at [`docs/ci.yml`](docs/ci.yml). It builds
with both g++ and clang++, runs the suite under sanitizers, and enforces
`-Werror` and determinism. It is not installed at `.github/workflows/` because
the account that opened this branch cannot add workflow files; move it into
place to enable it:

```sh
mkdir -p .github/workflows && git mv docs/ci.yml .github/workflows/ci.yml
```

## Layout

```
include/tetra/
  types.hpp        pieces, rotations, clear descriptors, tick type
  bitboard.hpp     playfield with occupancy + garbage planes
  pieces.hpp       shapes and SRS / SRS+ / 180 kick tables
  ruleset.hpp      versioned RulesetConfig (spec 6)
  rng.hpp          xoshiro256** + 7-bag randomizer
  piece_state.hpp  collision, rotation with kicks, spin detection
  attack.hpp       attack table, combo, B2B, Surge
  garbage.hpp      pending queue, cancellation, activation
  events.hpp       bounded event log (spec 7.3)
  player.hpp       one player's simulation
  movegen.hpp      BFS legal placement generation (spec 8)
  observation.hpp  the observation mask (spec 5.1, 18.3)
  tokenizer.hpp    Row/Column tokens + action embeddings (spec 9, 10.1)
src/               ruleset presets and hashing
tests/             117 tests, dependency-free harness
tools/             tetra_cli developer tool
docs/              spec, policy, architecture decisions
```

## Not yet built

M2 (PUCT/Gumbel search), M3 (garbage-aware self-play), M4 (opponent board and
intent head), M5 (multimodal). The neural network itself is specified but not
implemented — this milestone deliberately prioritises rule correctness, since
per spec §18.1 "the most important requirement is rule consistency, not model
performance".

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what comes next and which
decisions are still open.
