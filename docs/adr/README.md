# Architecture Decision Records

ADRs preserve the reasons behind architectural choices. `SPEC.md` remains the original design baseline; later decisions are recorded here instead of being back-written into the specification.

## Status vocabulary

- **Accepted** — the design rule is active. Code and tests determine what is actually implemented.
- **Accepted direction; implementation deferred** — the project has chosen the direction, but deliberately does not want it in the current production path yet.
- **Proposed** — under discussion and not yet a project constraint.
- **Superseded** — retained for history, with a link to the replacing ADR.

Existing ADRs should normally remain historical. If a later decision changes their scope, add a short follow-up link and create a new ADR rather than rewriting the old rationale.

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-cpp-instead-of-rust.md) | Use C++ for the engine instead of Rust | Accepted |
| [0002](0002-movegen-state-includes-rotation-provenance.md) | Preserve rotation provenance in move-generation state | Accepted |
| [0003](0003-movegen-performance.md) | Treat move-generation performance as a search constraint | Accepted |
| [0004](0004-action-duration-and-dijkstra.md) | Price actions by shortest-path timing and bounded delay bins | Accepted |
| [0005](0005-replay-format.md) | Use a deterministic versioned replay format | Accepted |
| [0006](0006-gravity-reachability.md) | Enforce gravity through reachability rather than post-hoc filtering | Accepted |
| [0007](0007-evaluator-interface-first.md) | Make batched evaluation structural before implementing search | Accepted |
| [0008](0008-search-gumbel-calibration.md) | Prefer Gumbel sequential halving at low search budgets | Accepted |
| [0009](0009-determinization-and-selfplay.md) | Determinize hidden futures and keep terminal reward equal to game result | Accepted |
| [0010](0010-cpp-python-handover.md) | Use one padded/masked C++↔Python tensor contract | Accepted |
| [0011](0011-cpp-inference-without-onnx.md) | Keep trained-weight inference in C++ without an ONNX runtime dependency | Accepted |
| [0012](0012-compact-dataset-replay-pi.md) | Store compact Replay+π data and regenerate tensors when possible | Accepted |
| [0013](0013-architecture-ablation-and-local-geometry.md) | Judge CNN/Transformer changes by Arena/search evidence, not policy loss alone | Accepted |
| [0014](0014-objectives-auxiliary-targets-and-vs-score.md) | Keep WDL as objective anchor; use dense aux targets and VS Score diagnostically | Accepted |
| [0015](0015-selfplay-provenance-search-mixture-and-timing-curriculum.md) | Lock self-play provenance and stage timing/cancellation after basic tactics | Accepted |
| [0016](0016-defer-sparse-moe-and-build-for-interpretability.md) | Separate strong-playing and teaching agents; keep language mediation implementation-agnostic and defer MoE/SAE | Accepted direction; implementation deferred |

## Decision chronology after the original specification

1. **Search/data correctness first (0007–0012).** The evaluator boundary, low-budget search, hidden-information determinization, C++/Python contract, inference parity, and compact datasets were stabilized before scaling learning.
2. **Architecture evidence became non-scalar (0013).** The 2026-08-08 CNN ablation showed that teacher imitation, WDL learning, and search strength can disagree. Arena behaviour therefore became mandatory evidence for model selection.
3. **Sample efficiency moved to dense supervision without reward shaping (0014).** Multi-horizon/action-conditioned targets are allowed, while WDL remains the terminal objective. VS Score joins APM/APP/PPS as a combat diagnostic once implemented.
4. **Self-play became a controlled curriculum (0015).** Data provenance, search budget mixtures, Champion protection, and staged timing/cancellation learning are explicit design constraints rather than informal training habits.
5. **Human learning is the goal; MoE/SAE are candidate means (0016).** The project separates maximally strong play from teaching/analysis quality and plans a bidirectional natural-language mediation layer between human questions and reproducible engine/model evidence. Sparse MoE and Sparse Autoencoders remain deferred candidate techniques rather than fixed requirements.

## Adding an ADR

A new ADR should contain `Status`, `Context`, `Decision`, `Consequences`, and links to measurements or implementation plans when relevant. Avoid turning experiment observations into Accepted decisions unless the project has actually chosen the resulting constraint.
