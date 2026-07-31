# ADR 0008: PUCT and Gumbel, and why Gumbel is the default

## Status
Accepted.

## Context
Spec §11.1 specifies PUCT and §11.2 recommends Gumbel sequential halving "to
stabilise policy improvement at low simulation counts", which is the regime this
bot runs in (16-128 simulations against a 30-50 ms budget).

Both were implemented on top of the batched `Evaluator` (ADR 0007). Getting
Gumbel right took three attempts, each caught by the same behavioural test: a
position whose two bottom rows are complete except for one column, so exactly
one placement clears anything. A correct search must find it at every budget.

## Bugs found, in order

1. **The heuristic prior was degenerate.** Its softmax temperature produced a
   top prior of 0.9918, so the search had nothing to explore and every mode
   collapsed onto one action. The temperature is now scaled by the observed
   score spread, which keeps the prior informative on any board.

2. **Unvisited actions were scored as q = 0.** Values here are mostly negative,
   so an unexplored action always looked better than an explored one and
   sequential halving discarded exactly the actions it had learned about.
   Unvisited actions now inherit the parent value.

3. **Gumbel noise drowned the logits.** With a weak policy the logits of ~34
   placements span ~0.6 nats while Gumbel noise over that many draws spans ~5,
   making root candidate selection ~90% random. Measured effect: Gumbel played
   at 27 pieces per game against 200 for its own policy-only baseline.

## Decision

Add `SearchConfig::gumbel_noise_scale`, defaulting to **0.05**, and default
`use_gumbel` to true.

The scale was calibrated on the forced-clear position:

| noise scale | 16 sims | 64 sims | 256 sims |
|---|---|---|---|
| 1.0 | wrong | wrong | wrong |
| 0.2 | wrong | wrong | correct |
| 0.05 | correct | correct | correct |

A trained, confident policy can raise this back towards 1; root Dirichlet noise
remains the intended exploration knob for self-play.

## Measured outcome

Under a steady garbage stream (2 lines every 8 placements), 6 games of 250
placements:

| | pieces | survived | attack/piece |
|---|---|---|---|
| policy-only | 228.7 | 5/6 | 0.169 |
| gumbel 32 | 250.0 | 6/6 | 0.207 |
| puct 32 | 62.7 | 0/6 | 0.136 |
| puct 128 | 250.0 | 6/6 | 0.249 |

Timing on this 2-core CPU with the heuristic evaluator, all within the §19.4
budget:

| | ms | mean batch |
|---|---|---|
| gumbel 32 | 11.5 | 5.5 |
| gumbel 64 | 20.5 | 4.9 |
| puct 64 | 23.5 | 13.0 |
| puct 128 | 46.7 | 14.3 |

## A limitation worth knowing

**PUCT is unreliable below roughly two simulations per legal action.** A tetris
position offers 25-50 placements; with a flat prior and a thin budget PUCT
concentrates its visits on an arbitrary tie-break rather than surveying the
options, and can finish *worse* than following the prior (puct-32: 62.7 pieces
against 228.7 for policy-only). Gumbel surveys 8 distinct actions where PUCT
surveys 1 on the same budget. This is pinned by
`puct_needs_enough_simulations_to_beat_its_prior` so it is not later mistaken
for a regression, and it is the concrete reason Gumbel is the default.