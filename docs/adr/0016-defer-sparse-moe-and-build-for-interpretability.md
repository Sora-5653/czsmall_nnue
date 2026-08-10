# ADR 0016: Keep the AI→human learning agenda implementation-agnostic; defer sparse MoE

## Status
Accepted direction; implementation deferred.

## Context

The long-term objective is broader than making the versus engine stronger. Once
the learning loop produces genuinely capable agents, the project should also ask
what strategic knowledge those agents acquire and how that knowledge can be made
useful to people.

This goal has two related but distinct research tracks.

First, agents optimized for different roles may learn differently. A model
optimized only to win as strongly as possible need not organize knowledge in the
same way as a model optimized to expose, explain, or teach useful concepts. The
project therefore distinguishes an **overwhelmingly strong playing agent** from
an **overwhelmingly useful teaching/analysis agent**. The latter may learn from,
distill, interrogate, or otherwise use the former, while humans primarily learn
through the teaching layer rather than being expected to decode the strongest
agent directly.

Second, natural language should eventually act as a bidirectional interface to
problems in the game world. A language-capable mediator should be able to turn a
human strategic question into formal/game-level probes and, in the other
direction, turn engine states, interventions, learned features, and discovered
regularities into explanations that can be inspected and challenged by a human.
Tetris is the first concrete domain, but this is intentionally phrased as a
problem-interface principle rather than a Tetris-specific UI feature.

Sparse mixture-of-experts (MoE) and Sparse Autoencoders (SAEs) are promising
techniques inside this agenda. MoE routing could expose coarse regime selection;
SAEs could identify recurring internal features. Neither technique is the goal
itself, and the final implementation is deliberately left open.

There is also a near-term engineering constraint. Tetris versus play contains
qualitatively different regimes—opening construction, ordinary midgame,
finishing/pressure conversion, upper-field survival, garbage interaction, and
timing—so sparse specialization is plausible. But introducing many experts too
early fragments training data, adds a router that must itself be learned, and
can starve rare regimes before the dense baseline is understood.

## Decision

### 1. Treat playing strength and teaching quality as separate objectives

The Champion/Arena loop continues to optimize and measure playing strength. A
future teaching/analysis agent is evaluated separately: it should recover useful
strategic distinctions, answer counterfactual questions, expose uncertainty,
and produce explanations or curricula that can be checked against the engine.

A teaching agent may consume traces, activations, search statistics, learned
features, or demonstrations from a stronger playing agent. It is not assumed to
be architecturally identical to that agent, and teaching quality is not inferred
from Elo/Arena strength alone.

### 2. Make natural language a bidirectional mediation layer, not the source of game truth

The eventual language layer should support both directions:

- **human → engine/research problem:** translate a strategic question or concept
  into concrete positions, slices, searches, interventions, comparisons, or
  hypotheses;
- **engine/model → human:** translate discovered regularities, feature
  activations, routing patterns, counterfactual results, and representative
  positions into explanations and lessons.

C++ simulation and recorded provenance remain authoritative for game facts.
Language-model output is an interface and hypothesis generator, not a substitute
for exact rules, search measurements, or reproducible experiments.

### 3. Keep the interpretability mechanism implementation-agnostic

Do not lock the human-learning pipeline to MoE, SAE, or any one interpretability
method. Candidate techniques can be replaced if better evidence or methods
appear.

The stable requirement is the pipeline:

1. obtain a strong, well-characterized agent;
2. collect provenance-tagged internal and behavioural evidence;
3. extract candidate strategic concepts;
4. validate them with counterexamples, interventions, ablations, or controlled
   engine experiments;
5. mediate validated concepts into human-readable explanations or training
   material;
6. test whether the teaching layer actually helps humans reason or play better.

### 4. Do not make sparse MoE the next scaling step

The immediate priority is to strengthen the dense self-play loop, resolve the
CNN/Transformer representation question, and obtain stable Arena competence.

A sparse MoE experiment is deferred until:

- the dense baseline is demonstrably strong;
- self-play generation and Candidate/Champion gating are stable;
- enough diverse data exists to populate specialists;
- the extra routing/compute complexity can be compared fairly with a dense
  model.

MoE is not a substitute for fixing weak targets, insufficient data, or an
unresolved shared trunk.

### 5. If MoE is tested, start small and make routing observable

The first MoE ablation should preserve a proven shared representation and add a
small number of sparsely activated expert blocks. Broad phase/situation signals
may influence routing, but the project should not immediately create one expert
for every named tactic or Cartesian combination of tactics.

Log enough routing information to diagnose and later interpret the system:

- selected expert(s) and top-k routing probabilities;
- router entropy and load balance;
- phase/situation metadata from the game record;
- per-expert usage over labelled slices such as openings, finishers, high-stack
  states, and garbage pressure;
- performance under expert/router ablations where practical.

Human labels are hypotheses about specialization, not semantic guarantees.

### 6. Use SAEs as a candidate feature-extraction method after strength exists

After a sufficiently strong checkpoint exists, SAE-based work may:

1. collect activations from the shared trunk and, if present, expert blocks over
   a diverse corpus;
2. train Sparse Autoencoders on selected activation streams;
3. retrieve positions/actions that strongly activate learned features;
4. correlate features with engine-derived concepts and tactical outcomes;
5. challenge interpretations with counterexamples and interventions;
6. pass validated concepts to the language/teaching layer.

A feature correlated with a human concept is not considered a causal explanation
merely because cherry-picked examples look convincing.

### 7. Preserve uncertainty in explanations

Router telemetry, SAE features, probes, and language explanations provide
different grades of evidence. The teaching pipeline must retain that distinction
rather than present every plausible interpretation as settled strategy.

This is particularly important because the final consumer is a human learner:
confidence in an explanation should track the strength of the validation behind
it.

## Consequences

- Near-term model development remains simpler and more sample-efficient than a
  premature MoE expansion.
- The architecture keeps room for specialization without hard-coding a brittle
  taxonomy of Tetris strategies.
- Interpretability work is not reduced to "inspect the router" or "train an
  SAE"; those are candidate instruments in a larger AI→human learning system.
- A future teaching agent becomes a first-class research object with evaluation
  criteria distinct from raw playing strength.
- Natural-language mediation is constrained by exact engine evidence, reducing
  the risk that fluent explanations become detached from what the agent or game
  actually does.

## Related documents

- ADR 0013 — dense CNN/Transformer architecture experiments.
- ADR 0014 — auxiliary targets and evaluation hierarchy.
- ADR 0015 — staged self-play/timing curriculum.
- `../ARCHITECTURE.md` — current architecture and deferred research placement.
