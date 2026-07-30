# ADR 0002: The movegen BFS state includes rotation provenance

## Status
Accepted.

## Context
The obvious BFS key for legal placement generation is `(x, y, rotation)`. That
is what the first implementation used.

It is wrong. Spin classification depends on *how* a piece arrived: only a piece
whose last successful action was a rotation can register a spin, and for the T
piece the kick index that was used distinguishes a mini from a full spin. Two
paths that reach the same coordinates are therefore not interchangeable.

Collapsing them made the emitted action set depend on BFS visit order. The
symptom that exposed this was a property test: left/right mirrored boards
produced *different numbers of legal placements* (18 of 175 sampled positions
disagreed), which should be impossible under the mirror-symmetric SRS+ table.

## Decision
Include `arrived_by_rotation` and the kick index in the packed BFS state, so
that a rotation-reached state and a slide-reached state at the same coordinates
are distinct search nodes.

## Consequences
- Mirror invariance is now exact: 0 mismatches, verified cell for cell.
- The state space grows, but the packed key stays at 20 bits, which is what
  makes the flat generation-stamped visited table (ADR 0003) possible.
- Spin-bearing placements are never lost to visit-order luck, which matters
  because those are precisely the high-attack actions the policy must see.
