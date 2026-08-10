# Current architecture

This document describes the architecture that exists on `main` and the boundary
between production code and experiments. It is not a replacement for
`SPEC.md`; later design changes are recorded in ADRs.

## 1. Simulation and rules: C++ is authoritative

The C++ engine owns all state transitions that can affect game truth:

- ruleset configuration and hashing;
- board state, pieces, spins, attack, garbage, B2B/combo/Surge;
- Cobra-backed legal placement generation;
- action duration, gravity reachability, and delay bins;
- observation masking and token/action generation;
- PUCT/Gumbel search, determinization, and two-player event ordering;
- replay verification, dataset serialization, and Arena game mechanics.

The learner is not allowed to reconstruct a second, approximate game. This
keeps training labels and inference semantics tied to one simulator.

## 2. Observation and action contract

The engine converts a masked player observation into a variable-length state
sequence and legal-action sequence. `TensorBatch` pads those ragged sequences
and carries masks so the same contract can be consumed by C++ inference,
PyTorch, and dataset readers.

The observation contains public information only. Hidden queue state and hidden
garbage information must not leak into tokens or search. Dataset/schema metadata
exists so equal tensor widths cannot be mistaken for semantic compatibility.

See ADR 0009, ADR 0010, ADR 0012, and `SAMPLE_EFFICIENCY_PLAN.md`.

## 3. Neural evaluators

### TetraFormer reference

`trainer/tetraformer.py` is the current reference architecture. It uses a
Transformer state encoder, variable-length action scoring, a WDL value head,
and auxiliary prediction heads. It remains the comparison control even when an
experimental architecture wins an Arena.

### CNN and hybrid experiments

`trainer/ablation_models.py` contains local-board CNN and CNN+Transformer
variants that preserve the same public forward contract. Their purpose is to
measure whether an explicit 2D inductive bias improves game strength without
changing the simulator or training data contract.

The 2026-08-08 experiment established an important separation:

- the Transformer fit the Transformer-generated policy teacher slightly better;
- the CNN learned WDL/value targets substantially better on the corrected split;
- the CNN's most reproducible advantage appeared under search rather than in
  raw policy-only play;
- small bolt-on CNN hybrid branches did not reproduce the full CNN's behaviour
  and showed late value overfitting.

Therefore the next hybrid experiment should test a concrete representation
hypothesis—such as a full-capacity local encoder shared by policy and value—
rather than adding more weakly coupled heads. See ADR 0013 and
`CNN_ABLATION_20260808.md`.

## 4. Search

The evaluator returns policy/value estimates; C++ search turns those estimates
into an improved root policy. Gumbel sequential halving is the default low-budget
search because ordinary PUCT was poorly calibrated when the simulation budget
was small relative to the branching factor.

Hidden future pieces are handled by root determinization. Self-play targets are
therefore generated from information a real player is allowed to observe.

## 5. Self-play and datasets

The generation loop is:

1. load a checkpoint and exact repository/ruleset/schema provenance;
2. generate self-play with a fixed, recorded search configuration and seed
   range;
3. store immutable shards plus manifests;
4. train on a controlled replay mixture;
5. evaluate Candidate against Champion with paired Arena seeds;
6. promote only if the configured gate passes.

Colab is an additional worker, not a second authority for rules, labels, or seed
allocation. Shards from different schema/ruleset/checkpoint contracts are not
silently merged. See ADR 0015 and `COLAB_MANUAL.md`.

## 6. Objectives

The objective hierarchy is intentionally separated:

- **policy:** imitate the search-improved action distribution;
- **value:** predict WDL from a fixed player perspective;
- **auxiliary heads:** extract dense supervision such as future attack, garbage,
  survival/top-out horizons, and action-conditioned consequences;
- **reward / terminal target:** remains the actual game result.

Combat statistics such as VS Score, APM, APP, PPS, and cancellation are
measurements. They may become auxiliary prediction targets after ablation, but
they do not replace WDL or Arena promotion. See ADR 0014.

## 7. Timing as a learned capability

Delay actions already exist in the action space, but their existence does not
mean the network has learned timing. Early Arena evidence showed policies almost
always choosing the fastest action. Explicit garbage timing / cancellation
avoidance is therefore treated as a staged capability after basic board tactics
are competent. See ADR 0015.

## 8. Deferred research: teaching agents, language mediation, MoE/SAE

The long-term AI→human learning agenda separates two roles that need not share
one model: an agent optimized for playing strength and an agent optimized for
analysis/teaching. The latter may interrogate or distill the former and is
judged by whether it exposes useful, checkable strategic knowledge rather than
by Arena strength alone.

Natural language is intended to become a bidirectional mediation layer: human
questions should be convertible into reproducible engine/model probes, while
states, interventions, learned features, and discovered regularities should be
convertible back into explanations. The exact simulator and provenance remain
the source of game truth.

Sparse MoE and Sparse Autoencoders are candidate techniques inside that agenda,
not architectural requirements. MoE is deferred until a strong dense baseline
exists; if tested, it should start as a small observable-routing ablation over a
proven trunk. SAE work likewise comes after a strong checkpoint and must validate
candidate features with counterexamples/interventions before passing them to a
teaching layer. See ADR 0016.
