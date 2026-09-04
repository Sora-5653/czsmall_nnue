# Code map

## Model and evaluator path

- `trainer/gpu_match.py::load_model` loads dictionary checkpoints into
  `TetraFormer(TetraFormerConfig(...))`, moves them to the requested PyTorch
  device, and keeps inference in evaluation mode.
- `trainer/gpu_arena.py` owns the existing binary bridge and serves model 0
  (candidate) and model 1 (champion). The experiment runner reuses its frame
  readers, request decoder, and `answer_request` implementation instead of
  introducing another tensor serialization path.
- The fixed A checkpoint is Gen14 (`width=256`, `layers=8`, `ffn=768`, 24 input
  features); B is the existing XS checkpoint (`width=64`, `layers=2`, `ffn=192`,
  the same 24 input features). Policy/value semantics and action encoding are
  therefore shared; B has fewer auxiliary output channels, which the bridge
  already handles without changing policy/value tensors.

## Search and simulator path

- `include/tetra/search.hpp::Searcher::search` is the common PUCT/Gumbel entry;
  `run_puct`, `run_gumbel`, `descend`, and `visit_root_action` expand the tree.
- `include/tetra/search.hpp::Evaluator` is the evaluator abstraction. The GPU
  child uses `RemoteGpuEvaluator`, which packs observations with the canonical
  tokenizer and waits for Python's response over the existing binary protocol.
- `include/tetra/arena.hpp::Arena::evaluate` performs the four-game factorial
  block for each pair: normal/mirrored geometry crossed with role swapping.
  `play_game` owns the event-driven two-player transition, outgoing attack
  delivery, move generation, hold, timing state, and top-out accounting.
- `RulesetConfig::tetra_league()` is the fixed ruleset. `MoveGenerator::generate`
  is the authoritative final-placement action generator.
- `Player::receive_attack`, `Player::lock_piece`, `GarbageQueue`, and the
  simulator's event timestamps own garbage travel, cancellation, activation,
  hold, and line-clear state transitions.

## Experiment-only controls added

- `SearchConfig::time_budget_ms` is disabled by default and stops starting new
  leaf work after a per-decision wall-clock deadline; an in-flight evaluator
  batch is allowed to finish and is counted as overshoot.
- `SearchConfig::node_budget` is disabled by default and provides the equal-node
  diagnostic cap.
- `ArenaConfig` applies side-specific time/node overrides without changing the
  production defaults.
- `GarbageStyle::None` is E1 (no incoming interaction), `GarbageStyle::Scripted`
  is E2 (fixed incoming schedule, no outgoing opponent-dependent attacks), and
  `GarbageStyle::Steady` is E3/full versus (the existing dynamic delivery path).
- Search telemetry records decision latency, evaluator elapsed time, leaves,
  nodes, depth, batch/flush counts, raw-policy action, selected action, and
  root visit summaries. The patched `tetra_cli gpu-arena-protocol` emits these
  diagnostics as JSON on stderr after the binary Arena result.

## Existing metrics and limitations

- Existing `ArenaResult` metrics provide wins/losses/draws, Wilson CI, VS, APM,
  APP, PPS, survival, pieces, sent/received/cleared garbage, and top-out modes.
- The repository's pre-existing `make test` currently fails at link time in the
  dirty checkout because of multiple definitions in the existing test set;
  `make tools` and a standalone syntax check are separate verification gates.
- E2 is an experiment-only fixed placement-event schedule implemented in the
  shared Arena loop. It is not a new ruleset and is not used by training or
  production defaults.
