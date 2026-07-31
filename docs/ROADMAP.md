# Roadmap

Status against the milestones in the specification (§21).

| Milestone | Status |
|---|---|
| **M0 — rule core** | **Done.** Board, pieces, SRS/SRS+/180 kicks, spins, clears, attack, garbage, ruleset versioning, event log. |
| **M1 — single-board policy inputs** | **Done** for the non-neural half: legal placement generation, per-action durations and delay bins, observation masking, Row/Column tokenizer, action embeddings. The Transformer itself and supervised pretraining are not implemented. |
| M2 — Leela-style search | **Done.** Batched `Evaluator`, PUCT and Gumbel sequential halving with virtual loss, transposition table, root determinization (chance nodes), training samples, replay buffer, self-play worker, candidate gating (`Arena` / spec §20), and compact Replay+π dataset format (ADR 0012). |
| M3 — garbage-aware self-play | Partially enabled: the rule core already models travel time, activation, cancellation and action duration. The self-play loop and curriculum are not written. |
| M4 — opponent board | `Observation` already has opponent fields and the tokenizer emits opponent tokens. Opponent Intent Head and 1v1 event-driven search are not written. |
| M5 — multimodal | Not started. |

## Immediate next steps

1. ~~**Action duration and delay bins (spec §8.4).**~~ **Done.** Actions are
   priced from the ruleset's handling settings via shortest path, and
   `MoveGenerator::expand_delay_bins()` produces the FASTEST / +NF /
   WAIT_FOR_EVENT variants with WAIT_FOR_EVENT bounded by garbage activation,
   the opponent's next lock and the lock delay. See ADR 0004.
2. ~~**Replay serialisation.**~~ **Done.** A versioned binary chunk carrying
   the `ruleset_hash`, seed and placements, with interleaved checkpoints so
   `verify_replay()` reports the first diverging placement. `tetra_cli record`
   and `tetra_cli verify` drive it. See ADR 0005.
3. ~~**The search.**~~ **Done.** PUCT and Gumbel on top of `Evaluator`, batched
   from the start, with virtual loss and a transposition table. Gumbel is the
   default; see ADR 0008 for the calibration and for PUCT's low-budget failure
   mode.
4. ~~**Chance nodes and the replay buffer.**~~ **Done.** Root determinization
   closes an information leak that let the search read past the preview, and
   the self-play worker now fills a replay buffer with training samples. See
   ADR 0009.
5. ~~**The network.**~~ **Built and training.** `trainer/tetraformer.py`
   implements the spec §9-10 architecture and trains on engine-exported data
   (held-out loss 4.86 -> 2.91). What remains is *training it seriously*, which
   needs a GPU: a spec-sized forward pass is ~8.6 ms/position on this 2-core
   CPU against a 30 ms / 64-simulation budget (ADR 0007). The C++ inference
   path is now closed: `TetraFormerEvaluator` loads `.tetrawts` weights and is
   verified against PyTorch to ~1e-7 (ADR 0011).
6. ~~**Self-play with trained weights.**~~ **Done.** `tetra_cli export`, `selfplay-gen`, `search`, and `play` now accept an optional `--weights <path>` flag to load `.tetrawts` and evaluate positions with `TetraFormerEvaluator`.
7. ~~**Candidate gating and the Arena** (spec §20).~~ **Done.** `Arena` (`include/tetra/arena.hpp`) evaluates Candidate against Champion over paired games with mirrored boards and identical piece sequences (`J <-> L`, `S <-> Z`), computing Wilson score 95% confidence intervals and checking spec §20 promotion thresholds. `tetra_cli arena` drives it.
8. ~~**Compact dataset serialization (ADR 0012).**~~ **Done.** `DatasetHeader::VERSION_COMPACT` (=2) stores Replay + π (ruleset, seed, move number, chosen action, policy/value/aux targets) and reconstructs tokenized observations and action embeddings on the fly during dataset loading (`deserialize_compact_dataset` in C++ and `tetra_cli decode-dataset` in Python), reducing `.tetradat` disk size and I/O by over 100x.
9. **Lock delay and `reset_limit`.** Gravity is now enforced as a reachability
   constraint (ADR 0006), but the lock-delay window is not: a piece resting on
   the stack may be manoeuvred for `lock_delay` ticks with a bounded number of
   resets, which at high gravity is the only manoeuvring window there is. The
   current model is conservative — it rejects some genuinely reachable
   high-gravity placements — and closing this is the last gap in the movement
   model.

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