# Documentation map

This directory separates the project's original specification, accepted design
history, current implementation status, experiments, and operational guides.
Keeping those roles distinct is intentional: the project has moved quickly, and
retrospectively rewriting the original specification would erase useful design
history.

## Authority and update policy

Use the documents in this order when answering different questions:

1. **`SPEC.md` — original design baseline.** It is the initial specification and
   is kept as a historical contract. Do not rewrite it to make later decisions
   look as if they were present from the start.
2. **`adr/` — design decisions after the specification.** ADRs record why a
   design was accepted, rejected, deferred, or superseded. When an ADR changes
   direction after `SPEC.md`, the ADR is the design-history record for that
   change.
3. **Code and tests — implemented behaviour.** An accepted ADR may describe
   future work. Only code and passing tests establish that a decision is
   implemented.
4. **`ROADMAP.md` — current execution state.** This tracks what exists on `main`,
   what is being evaluated, and what is deliberately deferred.
5. **Experiment reports — evidence, not policy.** Files such as
   `CNN_ABLATION_20260808.md` preserve measurements and failed hypotheses. ADRs
   cite them when an experiment changes project direction.
6. **Operational guides — how to run the system.** `SETUP.md`,
   `COLAB_MANUAL.md`, and the repository-root `AGENTS.md` document build,
   training, GPU, and shard workflows.
7. **`POLICY.md` — scope and safety boundary.** The project is a local simulator;
   it does not connect to TETR.IO.

When a new architectural decision is made, prefer adding an ADR and updating the
roadmap rather than editing old ADRs into a new story. Existing ADRs may receive
small follow-up sections that point to the newer decision.

## Current system at a glance

The project is split deliberately across two ownership domains:

- **C++ owns game truth:** rules, Cobra-backed legal placement generation,
  timing, hidden-information masking, search, self-play transitions, dataset
  serialization, replay verification, and Arena game mechanics.
- **Python/PyTorch owns learning:** neural evaluation, GPU batching, training,
  checkpointing, architecture ablations, and ROCm/CUDA inference workers.
- **The boundary is tested:** the same token/action contract is consumed by
  training and inference, and C++ inference is checked against PyTorch.

The production reference network is still TetraFormer. CNN and CNN+Transformer
models are first-class experimental candidates, not silent replacements. The
2026-08-08 ablation found that policy cross-entropy, value learning, and search
strength can disagree materially; architecture promotion therefore depends on
Arena evidence rather than one held-out loss number. See ADR 0013.

## Training and evaluation principles

The current training direction is deliberately conservative about objectives:

- game outcome remains the reward/value target; heuristic combat statistics do
  not replace win/draw/loss;
- dense auxiliary targets are used to improve representation and sample
  efficiency only when they do not damage policy learning;
- paired Arena win rate and confidence intervals remain the promotion criterion;
- VS Score is the next required combat-oriented diagnostic alongside APM, APP,
  PPS, survival, cancellation, and raw-policy/search-policy comparisons once
  its implementation/ruleset interpretation is pinned;
- fixed-data, fixed-budget, multi-seed ablations are required before attributing
  gains to an architecture or auxiliary head.

See `TRAINING_AND_EVALUATION.md`, ADR 0014, and ADR 0015 for the complete
protocol.

## Design-history index

See [`adr/README.md`](adr/README.md) for the ADR index. The post-spec decisions
added in August 2026 are:

- **ADR 0013:** architecture ablations are judged by search strength as well as
  imitation loss; TetraFormer remains the reference while stronger local-board
  hybrids are tested incrementally.
- **ADR 0014:** WDL remains the objective anchor; dense auxiliary targets and VS
  Score are diagnostics/representation targets, not reward shaping.
- **ADR 0015:** self-play data is provenance-locked and timing/cancellation play
  is introduced as a staged capability rather than assumed learned because
  delay actions exist.
- **ADR 0016:** the long-term AI→human learning agenda separates playing strength
  from teaching quality and uses natural language as a bidirectional mediation
  layer. MoE/SAE are candidate techniques, deliberately deferred rather than
  hard requirements.

## Where to look

| Need | Document |
|---|---|
| Original end-state design | `SPEC.md` |
| Why a design changed | `adr/README.md` and the numbered ADRs |
| Current architecture | `ARCHITECTURE.md` |
| Training / Arena protocol | `TRAINING_AND_EVALUATION.md` |
| Current priorities | `ROADMAP.md` |
| Sample-efficiency implementation detail | `SAMPLE_EFFICIENCY_PLAN.md` |
| CNN/CNN+Transformer evidence | `CNN_ABLATION_20260808.md` |
| GPU / local setup | `SETUP.md`, root `AGENTS.md` |
| Colab shard generation | `COLAB_MANUAL.md` |
| Project scope | `POLICY.md` |
