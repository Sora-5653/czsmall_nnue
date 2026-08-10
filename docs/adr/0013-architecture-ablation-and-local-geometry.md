# ADR 0013: Judge architecture changes by game strength; keep local-geometry experiments incremental

## Status
Accepted.

## Context

The original network path is TetraFormer: a Transformer consumes the engine's
row/column/global tokens and scores a variable legal-action set. By August 2026
the self-play corpus was large enough to ask whether an explicit 2D board
inductive bias would learn local geometry more efficiently.

The 2026-08-08 ablation compared three parameter-matched families on the same
47,693-sample corpus and controlled optimizer, update count, minibatch schedule,
and seed:

- the existing TetraFormer-S control;
- a residual CNN board encoder plus global-token pooling;
- a CNN+Transformer hybrid.

The experiment exposed a useful conflict between metrics. The Transformer
consistently achieved slightly better held-out policy cross-entropy against a
Transformer-search-generated teacher. On the corrected hashed split, however,
the CNN learned WDL/value much more successfully. The CNN's strongest repeatable
advantage appeared when the network was used *inside search*: its 16-simulation
Arena results were materially stronger than the matched Transformer across
independent seed blocks, while raw policy-only play was closer to parity.

Follow-up hybrids also mattered. A compact independent CNN value branch could
look good on static held-out value metrics and then overfit badly; small
CNN-token/fusion variants did not recover the full CNN baseline's search
advantage. Simple inference-time policy/value swaps likewise failed to identify
one head as the sole cause.

The central lesson is that this project has at least three different questions:

1. how closely a network imitates its search teacher;
2. how well it predicts WDL/value on held-out samples;
3. how useful its policy/value representation is when embedded back inside
   search.

Those questions are related, but the ablation showed they are not equivalent.

## Decision

### 1. Keep TetraFormer as the reference/control

The existing Transformer is not replaced merely because a new architecture
wins one short Arena, and it is not discarded because a CNN has a stronger local
inductive bias. It remains the stable comparison model while alternatives are
measured.

### 2. Treat the CNN result as real evidence, not as a production replacement

The full CNN baseline remains a first-class experimental candidate because its
search advantage survived corrected splitting and multiple Arena seed blocks.
That evidence justifies further work on local board encoders.

It does **not** justify silently changing the production/champion architecture.
Promotion still goes through the normal Candidate/Champion gate.

### 3. Do not use held-out policy loss as the architecture selector

Held-out policy cross-entropy is an imitation diagnostic. In this corpus the
teacher itself is Transformer-search-derived, so small differences in imitation
loss are not architecture-neutral evidence of game strength.

Architecture evaluation must include paired Arena play under fixed search
budgets and multiple seeds. Raw-policy Arena remains useful as a diagnostic, but
search Arena is the relevant test for a network intended to live inside search.

### 4. Future hybrids must test a concrete representation hypothesis

The next CNN+Transformer experiment should not repeat the pattern of attaching a
small, weakly coupled CNN head and hoping that local geometry transfers.

A justified next hybrid should test one of the hypotheses left open by the
ablation, for example:

- a full-capacity local CNN encoder comparable to the successful CNN baseline;
- local features shared by both policy and value rather than an isolated value
  branch;
- CNN-derived spatial features fed into a Transformer that is reserved for
  longer-range/global/opponent interactions.

The exact implementation is still an experiment. The accepted design rule is
that the hybrid must isolate a stated hypothesis and preserve the common
input/output contract so the comparison remains controlled.

### 5. Keep production Champion protected during ablations

Architecture experiments produce experimental checkpoints. They do not mutate
the production Champion until the configured Arena promotion criterion is met.

## Consequences

- The project may retain multiple model families longer than a single-metric
  workflow would, increasing experiment code and checkpoint complexity.
- The comparison is more expensive because search Arenas are required, but this
  directly measures the deployment regime of the evaluator.
- Policy loss, value metrics, and game strength can be reported without forcing
  them into one scalar story.
- Future sparse-MoE work must be based on a trunk that has already demonstrated
  useful dense-model behaviour; MoE is not a shortcut around unresolved
  CNN-vs-Transformer representation questions. See ADR 0016.

## Evidence

See `../CNN_ABLATION_20260808.md` for the fixed-data experiment, split audit,
independent-seed runs, policy/value factorial diagnostics, and failed hybrid
variants.
