# Roadmap

Status against the milestones in the specification (§21).

| Milestone | Status |
|---|---|
| **M0 — rule core** | **Done.** Board, pieces, SRS/SRS+/180 kicks, spins, clears, attack, garbage, ruleset versioning, event log. |
| **M1 — single-board policy inputs** | **Done** for the non-neural half: legal placement generation, observation masking, Row/Column tokenizer, action embeddings. The Transformer itself and supervised pretraining are not implemented. |
| M2 — Leela-style search | Not started. PUCT / Gumbel sequential halving, transposition table, chance nodes, replay buffer, candidate gating. |
| M3 — garbage-aware self-play | Partially enabled: the rule core already models travel time, activation, cancellation and action duration. The self-play loop and curriculum are not written. |
| M4 — opponent board | `Observation` already has opponent fields and the tokenizer emits opponent tokens. Opponent Intent Head and 1v1 event-driven search are not written. |
| M5 — multimodal | Not started. |

## Immediate next steps

1. **Action duration and delay bins (spec §8.4).** `PlacementAction::delay_bin`
   exists but every action is currently emitted as `FASTEST`. Wiring real
   per-action durations through the BFS (DAS charge, ARR, lock delay) is the
   prerequisite for M3's timing decisions, and it is the last piece of M1 that
   is specified but stubbed.
2. **Replay serialisation.** `tests/test_simulator.cpp` proves a placement
   script reproduces a board, but there is no on-disk format yet. Spec §17
   wants Protobuf; a versioned binary chunk carrying `ruleset_hash` would be
   enough to start.
3. **The network.** Even a small MLP over the existing tokens would let M2's
   search be exercised end to end before the full TetraFormer is built.

## Open questions

- **Kick table provenance.** The SRS+ and 180 tables are transcribed from
  community documentation and verified structurally (exact mirror symmetry for
  SRS+, the documented asymmetry for classic SRS and for 180). They have *not*
  been diffed against real TETR.IO replays. Spec §18.1 wants replay-diff
  testing; that needs legally obtained replay files.
- **Attack rounding at high B2B × high combo.** The community notes that osk's
  published chart diverges from the game by 1–2 lines when a large B2B is
  combined with a large combo, because the engine keeps non-integer
  intermediates. This implementation follows the documented formula. If exact
  parity matters, this is the first place to look.
- **Garbage messiness constants.** Modelled as a configurable per-line
  probability with a between-attack multiplier, matching the described
  behaviour; the real constants are not public.
- **Surge in reversed / QUICK PLAY modes.** Only the base cases are modelled.
