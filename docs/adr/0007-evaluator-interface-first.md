# ADR 0007: The evaluator interface is defined before the search

## Status
Accepted.

## Context
The plan had been to build the MCTS skeleton first with a dummy evaluation, on
the assumption that legal-move generation was the bottleneck. Measurement
reversed that assumption.

On this sandbox (2 CPU cores), for ~100 state tokens and ~40 legal actions:

| | per position |
|---|---|
| `generate_for_piece` | **0.088 ms** |
| TetraFormer-S (5.3M params, spec §9.5) | **8.6 ms** |
| half-size variant (0.67M) | 1.18 ms |
| tiny dev proxy (0.09M) | 0.35 ms |

The network is roughly **100x** more expensive than everything around it. A
64-simulation search costs ~5.6 ms of movegen and ~550 ms of inference against
a 30 ms budget (spec §19.4). Optimising movegen to zero would not help.

## Decision
Define `Evaluator` first, and design the search against it, rather than
retrofitting batching later.

Two properties are baked into the interface:

1. **The unit of work is a batch.** `evaluate()` takes a vector of positions;
   `evaluate_one()` is a thin wrapper over it, so there is a single code path.
   Spec §11.1 requires batched leaf inference and §19.4 targets >= 80% batched
   leaf evaluation — a scalar-first interface cannot reach either without a
   rewrite, and the batching strategy (leaf collection, virtual loss,
   pending-node bookkeeping) is *structural* to MCTS, not an optimisation.

2. **Policies are variable length**, returned per legal action (spec §10.1),
   never as a fixed x/rotation grid. This is what lets one evaluator serve
   custom board widths and distinguish placements that differ only by spin.

Two implementations ship with the interface: `UniformEvaluator` (the null
hypothesis every search must beat, and the fixture most search tests run on,
since its output is trivially known) and `HeuristicEvaluator` (a deterministic
stand-in with real opinions, which doubles as the heuristic baseline opponent
spec §19.1 and §13.4 require).

`Observation` now carries its `RulesetConfig` by value so an evaluator is
self-contained and cannot be paired with the wrong rules.

## Consequences

- The search can be written, tested and profiled before any weights exist, and
  swapping in a trained model is an implementation change behind one interface.
- `HeuristicEvaluator` is a credible baseline, not a placeholder: greedy over
  its priors survives 500-piece games in 8 of 10 seeds and clears ~197 lines.
  A weak baseline would have made every later strength comparison meaningless.
- `ChunkedEvaluator` bounds batch size for fixed-shape backends (ONNX/TensorRT)
  without the search knowing about backend constraints.
- The evaluator tracks positions and batches so the spec §19.4 batch-efficiency
  metric is measurable from day one.

## Notes on the sandbox

PyTorch 2.13 does install here (via a venv; the system Python is PEP-668
managed) and was used for the measurements above. But with 2 CPU cores and no
GPU, full-size training and strength evaluation are not feasible in this
environment. The completion criterion for M2 is therefore *correctness and
verifiability* — that the search provably behaves as specified with a small
model — with real-scale strength deferred to a GPU environment.