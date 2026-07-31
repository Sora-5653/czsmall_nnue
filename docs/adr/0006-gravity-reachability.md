# ADR 0006: Gravity as a reachability constraint

## Status
Accepted.

## Context
`MovementCfg::gravity_num/den` was carried in the ruleset and hashed, but the
generator ignored it. At normal speed (1/60 G) that is harmless: a placement
takes a handful of ticks and gravity needs 60 to move the piece one cell. At
high gravity it is not — the generator would emit placements the player could
never physically reach, and at 20G the piece is on the floor the instant it
spawns.

## Decision
Now that actions are priced in ticks (ADR 0004), the constraint is nearly free
to state: at a node reached after `cost` ticks, gravity has pulled the piece
down `floor(cost * num / den)` cells, so a node sitting higher than the spawn
row minus that amount is unreachable and is not expanded.

Gravity is kept as an exact rational and the comparison is integer arithmetic,
so reachability stays bit-reproducible (spec 5.2, 18.1).

The check is skipped entirely when gravity is slower than
`gravity_check_threshold` (default 8 ticks/cell), so the common case pays
nothing.

## Consequences

- At 20G the placement count on an empty board drops from 162 to 58; at 1G and
  below nothing changes, verified by
  `normal_gravity_does_not_restrict_placements`.
- Restriction is monotonic in gravity, which is asserted directly: faster
  gravity can only remove options.
- Measured cost: none. 174.8 µs/call before the change, 173.2 µs after, on the
  same benchmark.

## Limitations

- **Lock delay and `reset_limit` are still not modelled.** A piece resting on
  the stack may be moved for `lock_delay` ticks, with a bounded number of
  resets, which at high gravity is the *only* remaining manoeuvring window.
  Modelling gravity without it is conservative in one direction (some genuinely
  reachable high-gravity placements are rejected) and is the next gap to close.
- The check assumes the piece falls from the spawn row. A piece that has already
  soft-dropped is treated correctly because the cost is accumulated along its
  path, but no credit is given for gravity that happened *during* a DAS slide.