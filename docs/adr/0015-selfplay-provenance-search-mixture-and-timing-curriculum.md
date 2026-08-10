# ADR 0015: Keep self-play provenance-locked; stage timing and cancellation play after basic tactics

## Status
Accepted.

## Context

The training loop now spans local GPU generation, Colab workers, replay mixing,
resumable training, and paired Candidate/Champion Arena evaluation. That makes
sample volume much easier to scale, but also creates two risks:

1. data from different commits, checkpoints, rulesets, schemas, seeds, or search
   budgets can be mixed so that an apparent learning gain is actually a data
   provenance change;
2. because delay bins already exist in the legal action space, it is easy to
   mistake "the engine can represent timing" for "the policy has learned
   timing".

The second risk was observed directly. Early trained policies overwhelmingly
selected the fastest placement. `WAIT_FOR_EVENT` and other delay choices existed,
but the generated data did not establish that garbage timing or cancellation
avoidance (相殺外し) had been explored enough to learn.

At the same time, the project needs more self-play than a uniformly deep search
can cheaply provide. A mixture of broad shallow-search data and a smaller amount
of deeper-search data can improve coverage without making every position
expensive.

## Decision

### 1. Treat generated shards as immutable, provenance-tagged artifacts

A serious self-play shard must be attributable to a concrete generation
configuration. Record or validate at least:

- repository commit;
- checkpoint identity/hash;
- ruleset hash;
- dataset/tokenizer/action/aux schema versions;
- model version;
- search algorithm and simulation budget;
- determinization and root-noise settings;
- seed interval / shard identity;
- sample count and termination metadata.

Do not byte-concatenate shards or silently merge incompatible schemas. Colab,
Drive, or Google Apps Script may transport artifacts, but they are not the
authority for seed allocation, labels, or dataset merge semantics.

### 2. Protect Champion from data-generation experiments

Experiments generate Candidates. Champion is replaced only through the normal
paired Arena gate.

A successful training run, lower validation loss, higher APM/APP/VS Score, or a
single short Arena is not by itself permission to overwrite Champion. This keeps
the comparison target stable while the self-play distribution is changing.

### 3. Use a search-strength mixture rather than one uniform budget

Once the self-play loop is closed and stable, prefer a mixture with:

- **mostly shallow-search games/positions** for broad and inexpensive coverage;
- **a smaller deep-search component** for higher-quality policy/value targets;
- multiple search strengths or deliberately imperfect/recovery positions so the
  learner is not confined to the narrow state distribution of its current best
  policy.

Any position-start, stratified, or recovery curriculum is stored as a distinct
provenance class so its contribution can be ablated.

The comparison between mixtures must hold total game count or generation compute
fixed when making sample-efficiency claims.

### 4. Treat timing as a staged capability

Timing actions remain representable throughout training, but an explicit timing
curriculum is deferred until basic board tactics are visibly competent.

Readiness is judged from multiple signals:

- qualitative play shows stable stacking, Quads, and T-spins rather than random
  attack production;
- Arena strength is no longer dominated by elementary board failures;
- APP near the flat-stack Quad baseline of roughly 0.5 can be used as a rough
  diagnostic that attack construction exists.

The APP number is **not** a hard gate and is not a reward target. It is only one
readiness signal alongside play inspection, VS Score, and Arena results.

### 5. When timing is activated, measure actual use of the action dimension

Do not infer timing skill from the presence of delay bins in move generation.
Track at least:

- frequency of `FASTEST` versus delayed actions;
- `WAIT_FOR_EVENT` usage;
- attack sent/received/cancelled around timing decisions;
- cancellation avoidance / off-cancel events where definable;
- paired win rate and VS Score under the same search budget.

A timing experiment should demonstrate that delayed actions are selected in
situations where they improve game outcomes, not merely that their frequency
increased.

### 6. Do not hand-shape a positive reward for waiting

The model should learn *when* delay is useful from search, game outcome, and
trajectory-derived targets. A generic reward for waiting or for avoiding
cancellation would hard-code a tactical preference that can be wrong in other
states.

If exploration is insufficient, change search exploration, data sampling, or
curriculum coverage rather than redefining the terminal objective.

## Consequences

- Self-play storage and manifests become slightly more verbose, but experiments
  are reproducible and failed runs can be diagnosed instead of guessed at.
- Deep-search data is spent where it has the most teaching value instead of on
  every generated position.
- Timing/cancellation play can be developed without contaminating the earlier
  question of whether the model can stack and attack at all.
- The project may temporarily underrepresent sophisticated timing while the
  board-policy baseline is weak. This is deliberate: it reduces the chance that
  a sparse, difficult action dimension consumes training capacity before the
  simpler tactical substrate exists.

## Related documents

- ADR 0009 — determinization and the original self-play reward contract.
- ADR 0014 — objective/metric hierarchy and VS Score.
- `../COLAB_MANUAL.md` — shard generation workflow.
- `../TRAINING_AND_EVALUATION.md` — fixed-budget ablation and Arena protocol.
