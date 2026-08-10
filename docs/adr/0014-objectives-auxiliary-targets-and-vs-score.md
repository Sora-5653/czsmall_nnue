# ADR 0014: Keep WDL as the objective anchor; use dense auxiliary targets and VS Score as diagnostics

## Status
Accepted.

## Context

ADR 0009 deliberately separated terminal outcome from attack/garbage statistics:
self-play reward is the game result, while combat quantities are auxiliary data.
That remains important because a hand-shaped reward can make locally attractive
behaviour dominate the actual objective of winning.

At the same time, early training exposed two limitations of a sparse WDL-only
view:

- a complete game supplies relatively few independent terminal outcomes compared
  with the number of useful intermediate positions it contains;
- policy, value, and auxiliary heads can interfere through the shared trunk, so
  adding more supervision is not automatically beneficial.

The sample-efficiency work therefore expanded trajectory-derived targets across
multiple horizons and added explicit gradient diagnostics. Separately, Arena
reporting used APM, APP, and PPS, but these statistics describe volume and speed
more directly than combat effectiveness. The project decided to add TETR.IO's
VS Score as an additional comparison metric because it is intended to summarize
versus combat more directly than APM or APP alone.

The metric must still remain a proxy: no combat statistic is allowed to replace
actual paired wins/losses as the promotion criterion.

## Decision

### 1. Terminal game result remains the reward/value anchor

The outcome target is win/draw/loss from a fixed player perspective. Attack,
stack shape, cancellation, survival time, VS Score, and similar statistics do
not replace it and are not added directly as hand-shaped terminal reward.

A piece-limit truncation is not converted into a synthetic win. Unknown future
horizons in truncated trajectories are masked rather than trained as zero.

### 2. Use dense trajectory-derived supervision to improve representation

The training objective may take the form

\[
L = L_{\pi} + \lambda_v L_v + \sum_i \lambda_i L_{\mathrm{aux},i}.
\]

Current or planned auxiliary families include:

- future attack over several time or placement horizons;
- future garbage received;
- self/opponent top-out or survival horizons;
- time-to-terminal / discrete time-to-event targets;
- action-conditioned immediate consequences that can be computed exactly by
  the engine;
- VS Score or other combat summaries as explicitly ablated prediction targets.

Dense targets increase supervision per trajectory, but they do not create new
independent games. Sample-efficiency claims therefore remain tied to fixed game
counts or fixed generation compute.

### 3. Make VS Score a standard Arena/reporting metric

Once the metric implementation is available and pinned to a documented ruleset
interpretation, Arena and match reports should include VS Score alongside APM,
APP, and PPS.

The evaluation hierarchy is:

1. paired Arena win rate and its confidence interval at the specified search
   budget;
2. VS Score as a combat-oriented explanatory metric;
3. APM, APP, PPS, survival, cancellation, and timing statistics;
4. held-out policy/value/auxiliary metrics.

VS Score is intentionally below win rate. It can explain *why* a model is
stronger or distinguish styles that APM/APP blur, but optimizing the proxy is not
the project's terminal goal.

### 4. Treat VS Score as an auxiliary target only through ablation

Reporting VS Score does not automatically mean training on it. A VS auxiliary
head is an experimental target and must be tested against an otherwise identical
baseline.

The exact formula/version used to create a target must be recorded with the
schema. If the project cannot reproduce the intended metric exactly, it should
report the limitation rather than train against an undocumented approximation.

### 5. Monitor gradient interaction, not only scalar loss

Auxiliary weights are not selected from raw loss magnitudes alone. Training
should expose, at a useful cadence, shared-trunk gradient norms and policy/value
or policy/auxiliary gradient cosine similarities.

A small auxiliary loss can still dominate or oppose policy gradients. Conversely,
a numerically large loss can be harmless after weighting. Valid masks must
remove both loss and gradient contribution from unknown targets.

### 6. Require gameplay evidence before declaring an auxiliary target useful

An auxiliary prediction can improve its own held-out loss without improving the
representation used by policy/search. Therefore an auxiliary target is not
considered strategically successful unless controlled experiments show policy
and/or Arena benefit without an unacceptable compute cost.

Raw-policy and searched-policy evaluations should both be retained when they help
separate representation quality from search interaction.

## Consequences

- The trainer and dataset schema become more explicit and somewhat wider.
- Arena reports gain a more combat-specific diagnostic without weakening the
  Champion promotion rule.
- Auxiliary objectives can be aggressively explored while remaining removable;
  no target becomes part of the project's definition of "winning" merely
  because it correlates with strength.
- APM/APP remain useful. In particular, APP can help identify whether basic
  attack construction is present before a timing curriculum, but it is not a
  substitute for VS Score or win rate. See ADR 0015.

## Related documents

- ADR 0009 — terminal reward and determinized self-play.
- `../SAMPLE_EFFICIENCY_PLAN.md` — target schema, masks, split rules, and
  gradient diagnostics.
- `../TRAINING_AND_EVALUATION.md` — experiment and Arena reporting protocol.
