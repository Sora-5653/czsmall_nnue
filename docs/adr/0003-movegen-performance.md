# ADR 0003: Movegen is optimised as the MCTS inner loop

## Status
Superseded by the pure-Cobra movegen integration. The measurements below are
the historical optimization record for the removed project-side generator.

## Context
Spec §19.4 sets inference targets of 5 ms for policy-only and 30–50 ms for
64–128 simulations. Every simulation needs a legal move list, so movegen
throughput bounds the whole search budget. The first correct implementation ran
at 279 µs per call, which would have consumed the entire budget in ~180 calls.

## Decision
Three targeted optimisations, each verified to leave the output byte-identical
(same 777,136 placements over the benchmark, same determinism hashes, all tests
green):

1. **Parent-linked BFS paths.** The BFS copied a `vector<Input>` per expansion.
   Nodes now store a parent index and a single input; the path is reconstructed
   only for placements that are actually emitted. 279 → 163 µs.
2. **Flat, generation-stamped visited table.** The 20-bit packed state indexes a
   direct-mapped array instead of an `unordered_map`. Clearing it per call would
   cost more than the search, so a generation counter invalidates stale entries
   in O(1). 163 → 124 µs.
3. **Reused scratch buffers.** The arena, frontier and landing vectors live in
   `thread_local` scratch, so a steady-state call performs no heap allocation.
   124 → 117 µs.

Collision detection was also rewritten to use precomputed per-row bitmasks.
This turned out not to be the bottleneck, but it is strictly better and is
differentially tested against the naive implementation over 5.9M cases.

## Consequences
- 2.4× faster overall, with headroom for the search budget.
- The `thread_local` scratch means a `MoveGenerator` is safe to share across
  threads but its buffers are per-thread; this is verified under ASan.
- The bitmask collision path has a subtlety: the bounding box may hang off the
  left edge (negative `x`) while no filled cell does, so the shift direction
  must be chosen accordingly. This was caught by the differential test and is
  commented at the call site.

## Current implementation

`MoveGenerator` now uses Cobra's fixed standard 10x40 board directly. Cobra's
`MoveList` enumerates legal targets and an all-target `PathFinder` traversal
supplies canonical input paths in one pass per piece and input model; the
removed legacy generator is not used as a fallback. On the same CLI benchmark,
the pure-Cobra implementation measured **99.0 µs/call**, versus **132.4
µs/call** for the pre-switch hybrid adapter (20,000 calls, standard 10x40
field).
