# ADR 0004: Action duration, and why movegen became Dijkstra

## Status
Accepted.

## Context
Spec §8.4 defines a placement's cost in time and a set of delay bins, and spec
§12 insists that "cancel now / take the garbage / hold the attack back" must be
*learned* from a correct clock rather than implemented as a rule. Until this
change every action was priced as a constant (the CLI passed a hardcoded 20
ticks to `lock_piece`), so none of that was expressible.

## Decision

**1. Every action carries a real duration.** `HandlingModel` derives the cost of
each input from `RulesetConfig::movement` (DAS, ARR, SDF, ARE, lock delay), so a
custom room's handling flows straight into the search. `PlacementAction` gains
`base_duration` (cost of executing the inputs), `delay_ticks` (deliberate wait)
and `total_duration()`.

**2. The generator is Dijkstra, not BFS.** Spec §8.2 permits either. Once inputs
have different costs — a DAS slide is not a tap — plain BFS layering is wrong:
it settles nodes in *depth* order, so a node first reached by an expensive route
keeps that route's price, and improving a parent later leaves stale sums in its
children. The symptom was the cost model disagreeing with a full re-execution of
the emitted input sequence by up to 5 ticks.

Edge weights are small integers and settled cost is monotone, so the queue is a
bucket queue (Dial's algorithm) rather than a binary heap: O(1) push and pop,
no comparators, and reusable buckets keep generation allocation-free.

**3. Airborne nodes are landing candidates.** `L HD` and `SD L HD` reach the same
cells, but only the first is optimal. Since a soft drop changes `y` it is a
different search state, so the cheaper route existed but was never selected as a
landing. Any node reached by a move or a rotation is now also a landing
candidate via its implicit hard drop, and a trailing redundant `SoftDrop` is
elided from the emitted sequence.

## Consequences

- Costs are now genuinely optimal, verified against brute-force enumeration of
  every input sequence up to length 4 (`action_cost_is_the_true_shortest_path`).
  The cheapest T placement dropped from 2 ticks to 1, the dearest from 12 to 7.
- **Generation got slower: 117 → 155 µs/call.** This is the real price of exact
  pricing; most of it is the larger landing-candidate set. Deduplicating
  landings by final position *before* the expensive `evaluate_placement` call
  recovered a large part of it (213 → 155 µs).
- A latent self-consistency bug surfaced and was fixed: two different
  `(x, y, rot)` triples can place identical cells (I in rotation N at row `y`
  fills the same cells as rotation 2 at row `y+1`), so they merge as one
  outcome. Merging previously copied the cheaper input sequence but kept the
  other candidate's coordinates, leaving 162 of 5191 actions whose
  `canonical_input_sequence` did not reproduce their own stated position.
- `duration_matches_replaying_the_canonical_sequence` re-executes every emitted
  sequence and compares both position and price, so the incremental cost sum can
  never silently drift from the movement model.

## Alternatives considered

- **Keep BFS and re-execute each sequence to price it.** Simple and correct, but
  it double-walks every path (144 µs) *and* still reports non-optimal costs,
  since BFS would have chosen the route.
- **Restrict airborne candidates to horizontal moves.** Faster (149 µs) and
  intuitively sufficient, but wrong: a mid-air rotation can also undercut
  rotating after the drop. The optimality test rejected it.