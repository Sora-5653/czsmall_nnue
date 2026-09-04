# Deviations and pre-existing blockers

> 追加診断の更新は末尾の `Additional diagnosis update` に記録する。

## 2026-08-18 initial preflight

- `make test` failed before the experiment hook was added during link of the
  existing dirty checkout. The linker reported multiple definitions for
  `tetra::b2b_bonus_for(...)::bounds` and `tetra::shapes()::t` while including
  the existing `tests/test_human_replay.cpp` path. No reset, clean, or discard
  was performed. `make tools` succeeded.
- The first preliminary A/B forward benchmark measured B at roughly 2.5--3.6x
  A depending on batch size through 64. This is not the final 10,000-state
  microbenchmark; batch 128 and existing lightweight candidates remain to be
  checked before accepting a 4x shortfall.

## 2026-08-18 final measurement state

- The required 10,000-state evaluator benchmark was completed on RX 9070 XT
  `cuda:1`. B/A was 4.635x at batch 128 and 7.450x at batch 256. This satisfies
  the evaluator-only 4x target; no further architecture shrink was needed.
- The long-run Arena path did not meet the handoff's 200-game pilot gate. A
  300-piece, four-game 10 ms attempt with batch 128 was stopped after about 13
  minutes without an Arena result and with Windows GPU utilization near 1%.
  Batch 32 and batch 16 long attempts showed the same no-result behavior after
  hundreds of protocol requests. These runs are not gameplay evidence.
- A request trace showed variable token/action shapes (token widths roughly
  84--103 and action counts 9--75 in the observed prefix). A fresh direct probe
  of a previously unseen shape took about 4.6--4.7 seconds on its first call and
  about 9--11 ms thereafter. This supports ROCm shape-dispatch/setup as a
  contributing factor, but does not by itself prove it is the only cause of the
  long-run stall.
- Fixed masked padding to 128 tokens/actions was probed as a mitigation. A
  five-piece smoke completed, but the longer fixed-shape trace also stopped
  during a later batch-2 request. It was therefore not silently promoted to the
  primary protocol; stable short rows retain the existing variable-shape bridge.
- Completed gameplay rows are intentionally short: four games per row, max
  pieces 5 or 20, plus one raw-policy 50-piece row and one equal-node row. The
  resulting confidence intervals are descriptive only, not a strength claim.
- After the initial preflight failure, a clean incremental rebuild was allowed
  to finish. Final `make test` passed 292 tests/1,055,146 assertions plus 3
  reanalyse tests/17 assertions; the initial linker failure is retained only as
  historical dirty-checkout evidence.

## Additional diagnosis update

- The previous long-run no-result observation is now classified. With the old
  undrained `stderr=PIPE` control, the external watchdog observed
  `phase=waiting_child_exit` after the binary Arena result frame and after the
  final request had completed. The result file was not written. The child was
  therefore no longer in Python inference or C++ search at the point of the
  stall.
- The C++ child writes a final per-decision diagnostics JSON line to stderr.
  Its recorded volume was about 5.1--5.9 KiB for the four-game diagnostic rows.
  A dedicated stderr drain let the child exit under the same seed and settings;
  the old pipe control reproduced the stop. This is sufficient evidence for
  stderr pipe saturation as the primary hang cause.
- The stable default in the experiment runner and the regular `gpu_arena.py`
  path now drains stderr concurrently. The old undrained mode remains available
  only as an explicit reproduction control.
- With drain enabled and prewarming batches 1/2/4/8/16, the same seed completed
  300 pieces x 4 games three consecutive times. A 128x128 fixed masked shape run
  also completed. These are stability checks, not strength results.
- The four-game diagnosis rows remain descriptive only. After the stability gate,
  a separate E3/10 ms 200-game Pilot was run: 50/50 blocks completed, B won 6,
  lost 194, with no draws. No checkpoint promotion decision was made, and the
  remaining 40/160 ms and E1/E2 Pilot matrix plus Main remain out of scope for
  this pass.
