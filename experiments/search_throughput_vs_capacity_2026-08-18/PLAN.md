# Search throughput vs evaluator capacity

This experiment follows the external handoff at
`C:\Users\eddyf\Documents\Codex\2026-08-18\search-throughput-vs-capacity\HANDOFF.md`.

## Frozen primary contract

- A: formal Gen14 checkpoint `models/gen14_rank100_100_20260814.best.pt`.
- B: existing same-schema XS checkpoint
  `models/size_search_ablation_20260816/seed42/transformer_xs.final.pt`.
- Hardware: AMD Radeon RX 9070 XT, PyTorch device `cuda:1`.
- Precision: fp16; one shared search configuration; timing actions off.
- Search mode: current production Gumbel configuration, unless a config records a
  deliberate diagnostic control.
- Primary decision budgets: 10, 40, and 160 ms; curve budgets: 5, 20, 80 ms.
- Paired seeds and role/mirror swaps are fixed in `manifests/seeds.txt`.
- Existing simulator, move generator, ruleset, and GPU bridge remain authoritative.

## Execution order

1. Record worktree and runtime manifests.
2. Baseline same-checkpoint smoke.
3. Benchmark A/B forward latency for batch sizes 1, 2, 4, 8, 16, 32, 64,
   and 128 when memory permits, with at least 10,000 states per measurement.
4. Add only the minimum wall-clock and diagnostic hooks needed by the handoff.
5. Run A0/B0 raw-policy and equal-node diagnostics.
6. Run wall-clock pilot for every primary budget.
7. Run the main direct A-vs-B comparison and fixed-opponent comparisons where
   the pilot is healthy.
8. Run E1 no incoming interaction, E2 scripted incoming garbage, and E3 full
   versus using the same A/B and seeds.
9. Generate summary CSV/JSON and graphs, then run the final non-destructive
   verification checks.

The 4x throughput target is pursued through reasonable existing-model and batch
configuration checks. If it remains below 4x, the report will state the measured
ratio and the concrete limiting factors; the gameplay comparison still proceeds.

## Recorded execution status

- Phase 0 is complete. B/A is 4.635x at batch128 and 7.450x at batch256 with
  10,000 measured states per batch.
- Short Phase 1/2 diagnostics are complete for E1/E2/E3, raw-policy, and
  equal-node controls. These use one four-game paired block and max pieces 5 or
  20, plus the raw-policy 50-piece control.
- The long-run stop was diagnosed as undrained child stderr after the result
  frame. The runner and regular `gpu_arena.py` now drain stderr concurrently;
  the old mode remains an explicit reproduction control.
- The minimum stability gate passed: the same 300-piece/4-game/10 ms seed
  completed three consecutive times, the fixed-shape plus batch-prewarm run
  completed, `make tools` and `make test` passed, and no checkpoint changed.
- The E3/10 ms/300-piece 200-game Pilot then completed as 50/50 blocks with no
  timeout. B won 6/200 (3.0%, 95% CI 1.4--6.4%). This is a separate performance
  evaluation, not a promotion decision. The remaining 40/160 ms and E1/E2
  Pilot matrix and Main are still pending.
