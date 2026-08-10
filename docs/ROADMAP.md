# Roadmap

This is the current execution view of the project. `SPEC.md` remains the original
specification; accepted changes in direction are recorded in `adr/` and linked
here.

## Milestone status

| Area | Status |
|---|---|
| **M0 — rule core** | **Done.** Board, pieces, SRS/SRS+/180 kicks, spins, clears, attack, garbage, ruleset versioning, event log. |
| **M1 — policy inputs / move generation** | **Done for the current contract.** Cobra legal placements, action timing/delay bins, masked observations, row/column/global tokens, bag and opponent-counter tokens, schema identifiers, and variable legal-action embeddings are implemented. |
| **M2 — search / self-play / training** | **Done as an end-to-end loop.** Batched evaluators, PUCT/Gumbel search, determinization, replay/datasets, PyTorch training, C++ weight inference, GPU self-play, GPU Arena, resumable checkpoints, replay mixing, and guarded Candidate→Champion iteration exist. |
| **M3 — garbage-aware self-play** | **Enabled.** Two boards advance by timestamp and route attacks through the same event machinery used by search/Arena. No-attack curricula remain available explicitly. |
| **M4 — opponent-aware model** | **Core observation path enabled.** Opponent board/counters are tokenized and two-player search is implemented. Dedicated opponent-intent modelling and league training remain future work. |
| **M5 — multimodal** | **Not started.** No image/video input path is part of the current priority stack. |

## Current model status

TetraFormer remains the reference/control architecture. CNN and CNN+Transformer
families are experimental candidates using the same simulator and tensor
contract.

The 2026-08-08 ablation changed how architecture work is evaluated:

- the Transformer fit the Transformer-derived policy teacher slightly better;
- the full CNN learned WDL/value substantially better on the corrected split;
- the CNN's clearest repeatable advantage appeared inside search;
- small bolt-on CNN hybrids did not reproduce that advantage and could overfit
  value badly.

Therefore the next architecture experiment is **not** "replace Transformer with
CNN" and is not another arbitrary head attachment. It should test a concrete
shared-representation hypothesis, such as a full-capacity local CNN encoder
shared by policy/value and combined with Transformer global context. See
[ADR 0013](adr/0013-architecture-ablation-and-local-geometry.md) and
[CNN_ABLATION_20260808.md](CNN_ABLATION_20260808.md).

## Active priority 1 — sample efficiency and objective diagnostics

The repository now has:

- tokenizer/observation/action/auxiliary schema identifiers;
- explicit `terminated` / `truncated` handling in the dataset contract;
- bag and opponent-counter tokens;
- 36 auxiliary targets: four legacy targets plus interval targets over real-time
  and placement horizons for attack, garbage received, self top-out, and
  opponent top-out;
- valid masks for unknown future horizons;
- trainer-side auxiliary target statistics;
- shared-trunk gradient norms and policy/value / policy/auxiliary cosine
  diagnostics.

Next steps:

1. validate the multi-horizon targets on larger mixed-generation data and keep
   split leakage checks explicit;
2. run fixed-data/fixed-budget ablations of policy-only, WDL, and auxiliary
   configurations rather than assuming the added heads help;
3. add **VS Score** to match/Arena reporting alongside APM/APP/PPS;
4. only after the reporting metric is pinned, test VS Score as an optional
   auxiliary prediction target—never as a replacement for WDL reward;
5. add action-conditioned consequence targets only where the engine can produce
   exact labels without duplicating an input feature as its own target.

See [ADR 0014](adr/0014-objectives-auxiliary-targets-and-vs-score.md),
[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md), and
[SAMPLE_EFFICIENCY_PLAN.md](SAMPLE_EFFICIENCY_PLAN.md).

## Active priority 2 — close the self-play loop under controlled provenance

The local/Colab generation path and manifest validator exist. The remaining work
is to turn the loop into a repeatable source of *comparable* generations rather
than merely a way to produce more samples.

1. Keep every shard attributable to commit, checkpoint, ruleset, schema, search
   settings, and a non-overlapping seed interval.
2. Compare local-only and mixed local/Colab generations under the same Arena
   protocol before promotion.
3. Prefer a search-strength mixture once generation is stable: mostly shallow
   search for coverage plus a smaller deeper-search component for stronger
   targets.
4. Record position-start/recovery curricula as separate provenance classes if
   introduced; do not silently mix them into ordinary self-play.
5. Keep Champion immutable outside the configured Arena gate.

Resumable Drive transport remains useful operational work, but Drive/GAS must
remain transport/orchestration rather than the authority for seeds or labels.
See [ADR 0015](adr/0015-selfplay-provenance-search-mixture-and-timing-curriculum.md).

## Active priority 3 — basic tactics before explicit timing curriculum

Delay bins and `WAIT_FOR_EVENT` already exist, but early trained policies did
not demonstrate meaningful use of them. The presence of timing actions is not
evidence that cancellation timing (相殺外し) has been learned.

The staged plan is:

1. first obtain visibly competent stacking, Quads, T-spins, and ordinary attack
   construction;
2. use Arena, VS Score, qualitative play, and APP together to judge readiness;
   APP around the flat-stack Quad baseline (~0.5) is only a rough diagnostic;
3. then strengthen exploration/data coverage for delayed actions and garbage
   timing;
4. measure delayed-action frequency, `WAIT_FOR_EVENT` use, cancellation
   interaction, VS Score, and paired wins before claiming the capability exists.

No positive reward for "waiting" is introduced. Search and actual game outcomes
must determine when delay is useful. See ADR 0015.

## Later — strong playing agents, teaching agents, and language mediation

The long-term research target is an **AI→human learning pipeline**, not a
particular interpretability architecture. It separates a maximally strong
playing agent from a teaching/analysis agent whose job is to expose useful,
checkable strategic knowledge. The teaching layer may interrogate or distill a
stronger player rather than being architecturally identical to it.

Natural language should mediate in both directions: human strategic questions
become reproducible game/model probes, and engine/model discoveries become
human-readable explanations or curricula. Exact simulation, interventions, and
provenance remain the evidence underneath the language layer.

Sparse MoE and Sparse Autoencoders are candidate tools in this program. MoE is
not the next scaling step and is deferred until a strong dense baseline exists;
if tested, routing must be observable and broad regime labels remain hypotheses,
not guarantees. SAE feature extraction is likewise downstream of strength and
must survive counterexamples/interventions before a feature is used for
teaching.

See [ADR 0016](adr/0016-defer-sparse-moe-and-build-for-interpretability.md).

## Engine correctness work that remains

### Lock delay and `reset_limit`

Gravity is enforced as a reachability constraint (ADR 0006), but the full
lock-delay manoeuvring window with bounded resets is still a known gap. At high
gravity the current model can conservatively reject placements that should be
reachable.

### TETR.IO parity questions

- **Kick-table provenance:** SRS+ and 180 tables are structurally tested but not
  yet replay-diffed against legally obtained real TETR.IO replays.
- **High B2B × high combo rounding:** the implementation follows the documented
  formula; exact engine intermediates may differ in extreme combinations.
- **Garbage messiness constants:** behaviour is configurable, but real constants
  are not public.
- **Surge variants:** reversed / QUICK PLAY modes are not fully modelled.

These are correctness/parity questions, not reasons to loosen the deterministic
ruleset/hash/schema contracts.

## Documentation rule

Do not rewrite `SPEC.md` to match this roadmap. When direction changes, add an
ADR, update this file and the relevant operational guide, and leave the original
specification as the historical baseline. See [docs/README.md](README.md) for
the document hierarchy.
