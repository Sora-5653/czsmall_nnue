# ADR 0013: Borrow LC3 runtime structure without changing the learning objective

Status: accepted (2026-08-15)

## Context

The current TetraFormer training line has two constraints that matter more than copying another engine literally:

1. Gen14 remains the competitive reference/champion while stacking/firepower experiments continue separately.
2. Timing actions stay disabled until the clean-domain APP gate is reached.

The useful part of the LC3 architecture is therefore not its game-specific policy or evaluation function. It is the way search work is separated and scheduled: a shared streaming inference queue, an explicit search-policy layer, coherent edge-local statistics, and a Gather/Eval/Backprop pipeline with telemetry.

The existing engine already had most of the raw mechanisms (batched leaves, virtual loss, a transposition table, and a Python multi-game micro-batcher), but those mechanisms were coupled to their callers and were difficult to measure independently.

## Decision

Adopt four LC3-style runtime ideas while preserving search semantics and all training labels.

### 1. Shared streaming inference queue

Independent C++ searches feed one `StreamingInferenceQueue` on the Python GPU side. The queue gathers requests for a bounded time window and groups compatible model IDs into one GPU forward.

The queue records:

- wire request count;
- evaluated positions;
- GPU forward count;
- queue wait time;
- deadline-triggered flush count;
- maximum positions and wire requests in one forward;
- target batch fill ratio.

`gpu_match.py` and `gpu_selfplay_parallel.py` now use the same scheduler rather than carrying separate copies of the batching loop.

This changes scheduling only. Search decisions, replay targets, random seeds, and model outputs are unchanged.

### 2. Separate `SearchPolicy`

PUCT and Gumbel edge scoring are moved behind `SearchPolicy`. `Searcher` remains responsible for tree lifecycle, simulation state, determinization, evaluation scheduling and backup.

This boundary is intentional: future search calibration can be changed or ablated without rewriting the tree runtime.

### 3. Edge-local `P/Q/N/N_inflight`

A search edge is now represented by one `SearchEdge` object containing:

- `prior` (`P`);
- `value_sum`, with `q()` as the derived mean (`Q`);
- `visits` (`N`);
- `inflight` (`N_inflight`, the virtual-loss count);
- child node id.

Previously these lived in parallel arrays on each node. Keeping them together makes the search invariant explicit and prevents individual edge fields from drifting out of sync during later asynchronous work.

### 4. Explicit Gather / Eval / Backprop stages

Leaf collection, neural evaluation and backup are now distinct stages:

1. Gather walks the tree and reserves edges through `N_inflight`.
2. Eval groups gathered leaves by evaluator and performs batched inference.
3. Backprop commits the returned values and releases `N_inflight`.

The current implementation still executes those stages serially inside a search call. The separation is deliberately structural: it makes later overlap or a central scheduler possible without changing edge semantics first.

`SearchTelemetry` records gather attempts, gathered leaves, selection steps, terminal backups, depth cutoffs, evaluation flushes and maximum edge in-flight count.

## Validation

- Full C++ suite after the refactor: 289 tests, 1,055,122 assertions, 0 failures.
- Python modules compile successfully.
- Two-game GPU self-play smoke test on the RX 9070 XT:
  - 92 wire requests;
  - 230 evaluated positions;
  - 46 GPU forwards;
  - mean 5.0 positions/forward;
  - maximum 8 positions/forward;
  - queue fill ratio 0.625 for an 8-position target;
  - bounded-window deadline flushes are visible in telemetry.

## Non-goals / deferred LC3 ideas

The following are intentionally not copied yet:

- a new reward, value target or policy target;
- timing-search expansion (still gated by the APP milestone);
- a separate `NodeRepository` abstraction;
- a more aggressive transposition DAG ownership model;
- asynchronous legal-move expansion workers.

The existing node vector plus transposition table already provides the functional baseline for repository/DAG work. Those additional abstractions should be justified by the new telemetry before increasing concurrency and ownership complexity.

## Consequences

The engine now has a cleaner boundary between search policy, search state, inference scheduling and backup. This should make high-throughput self-play cheaper to scale and future ablations easier to isolate, while leaving the current Gen14/Gen24 learning programme and timing gate unchanged.
