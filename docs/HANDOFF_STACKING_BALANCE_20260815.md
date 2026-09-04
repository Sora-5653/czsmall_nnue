# Stacking / Arena balance handoff — 2026-08-15

## Executive state

The active objective has changed from timing/cancellation ablation to **static stacking development while preserving competitive Arena strength**.

Hard constraints for the next session:

1. **Timing remains OFF.** Do not resume WAIT_FOR_EVENT/timing-head work yet.
2. Resume timing ablations only after the **clean stacking benchmark mean APP exceeds 0.5** on a fixed multi-seed evaluation.
3. `models/gen14_rank100_100_20260814.best.pt` remains the **model-strength champion** until a candidate passes frozen Arena convincingly.
4. Firepower specialists may be kept as teachers/staging models without promotion.
5. Candidate selection is two-dimensional: clean stacking APP **and** frozen Arena. Improving one while materially regressing the other is not a promotion.

Current roles:

- **Arena champion:** `models/gen14_rank100_100_20260814.best.pt`
- **Firepower specialist:** `models/gen24_stack_expertmix020x8_40_20260815.best.pt`
- **Current balance/staging candidate:** `models/gen25_stack_arena_recover20_20260815.best.pt`
- Do not promote Gen24 or Gen25 yet.

## Why the APP metric was changed

Ordinary 1v1 APP is contaminated by incoming garbage/cancellation: it measures attack actually sent after defensive interaction, not purely how efficiently the bot constructs attack.

A clean stacking benchmark was therefore added. `trainer/gpu_selfplay.py` now accepts:

```text
--no-attack-delivery
```

and the GPU export protocol passes this into the C++ self-play worker. Attack is still computed and recorded normally, but attacks are not delivered to the opponent. This removes incoming pressure/cancellation from the firepower measurement while preserving the normal rules, placement/search machinery, and attack calculation.

Use the clean benchmark with:

- timing OFF
- `--sims 64`
- `--determinizations 2`
- `--root-noise-fraction 0`
- `--policy-temperature 1.0`
- 300 pieces
- fixed seeds, currently `19000000`, `19000001`, `19000002`

Example:

```bash
./.venv-rocm714/Scripts/python.exe trainer/gpu_selfplay.py \
  models/CANDIDATE.pt data/benchmark/CANDIDATE_seed19000000.tetradat \
  --engine build-win/tetra_cli.exe --device cuda:1 \
  --games 1 --pieces 300 --sims 64 --batch 16 --seed 19000000 \
  --model-version 99 --determinizations 2 --root-noise-fraction 0 \
  --policy-temperature 1.0 --precision fp16 --no-attack-delivery
```

The **0.5 APP timing-restart gate should be applied to this clean benchmark**, not ordinary 1v1 APP.

## Frozen Arena protocol

Promotion/strength checks remain:

- candidate vs `models/gen14_rank100_100_20260814.best.pt`
- timing OFF for both
- Gumbel ON
- 32 simulations
- 1 determinization
- policy T=1.0
- default Gumbel c/noise (`0.01`, `0.05`)
- paired mirrored Arena
- seeds 42 and 1337 as the first screen

Current 4-game-per-seed sample is small. A 5-3 result is encouraging but **not sufficient for champion promotion**.

## Clean stacking results

### Gen14 champion baseline

| seed | APP |
|---|---:|
| 19000000 | 0.200 |
| 19000001 | 0.157 |
| 19000002 | 0.163 |
| **mean** | **0.173** |

### Gen24 firepower specialist

`models/gen24_stack_expertmix020x8_40_20260815.best.pt`

| seed | APP |
|---|---:|
| 19000000 | 0.207 |
| 19000001 | 0.230 |
| 19000002 | 0.220 |
| **mean** | **0.219** |

This is about +26% relative to Gen14 on the three-seed clean mean and was the first repeatable clean-firepower improvement.

However frozen Arena failed:

- seed42: **1-3**
- seed1337: **1-3**
- pooled: **2-6**

Arena APP also fell relative to Gen14. Therefore Gen24 is a useful firepower teacher/specialist, not a champion.

### Gen25 Arena-recovery candidate

`models/gen25_stack_arena_recover20_20260815.best.pt`

Construction: start from Gen24 and replay the full Gen15 competitive corpus for 20 steps, policy head only, policy+rank losses, LR `1e-5`, no elite replay.

Clean APP:

| seed | APP |
|---|---:|
| 19000000 | 0.200 |
| 19000001 | 0.193 |
| 19000002 | 0.130 |
| **mean** | **0.174** |

This essentially returns clean APP to the Gen14 mean, i.e. it recovers competitive behavior but gives back almost all of Gen24's firepower advantage.

Frozen Arena:

- seed42: **2-2**
- seed1337: **3-1**
- pooled: **5-3**

This is the current best *balance* point observed, but not a promotion due to the small sample and lack of clean APP improvement over Gen14.

## Rejected balance paths

### Gen25b: 10-step competitive replay

`models/gen25b_stack_arena_recover10_20260815.best.pt`

Despite being a nominal midpoint in training steps, performance was not an interpolation of Gen24 and Gen25. Clean seed19000001 collapsed to **APP 0.117**. Reject.

### Policy-weight checkpoint interpolation

Added `trainer/interpolate_policy_checkpoints.py`. It verifies that non-policy parameters are bit-identical and interpolates only:

- `action_in.*`
- `policy_attn.*`
- `policy_norm.*`
- `policy_out.*`

Gen14 vs Gen24 differs in exactly 9 tensors, all in this list.

Results on clean seed19000001:

- alpha 0.50: **APP 0.163**
- alpha 0.75: **APP 0.146**

Search behavior is strongly nonlinear in parameter interpolation, so neither was worth Arena testing. Keep the utility for diagnostics, but do not assume weight-space interpolation gives a behavioral Pareto curve.

### Simultaneous competitive + expert replay (Gen27)

Attempted from Gen14 with:

- high-firepower expert files from Gen6/7/11/15
- Gen15 competitive corpus included twice
- APP >= 0.2 trajectories repeated x4
- policy head only
- 30 steps, LR 2e-5

Validation worsened **4.0344 -> 4.3492**. The `.best.pt` file was saved at step 3312 before training improved anything and is effectively the initial Gen14 state for this run. Do not treat it as a new candidate.

This shows that merely mixing the two distributions by sample count does not solve the conflict.

## Historical expert-data discovery

A full scan of `data/production/*.tetradat` found:

- 303 files
- 4,327 player-trajectories
- mean historical APP ~0.067 (ordinary 1v1 sent APP)
- APP >= 0.20: **308 trajectories**
- APP >= 0.25: **86**
- APP >= 0.30: **18**
- APP >= 0.40: **1**
- APP >= 0.50: **0**
- maximum observed: **0.424** (`gen15_rank100prod_seed15082005`, player +1, 125 pieces, 53 attack)

Several old Gen6/7 raw-anchor trajectories are in the 0.30-0.36 range. This was important because recent generations had become safety-biased and contained much less high-APP diversity.

## Aux-schema replay compatibility change

`trainer/tetra_dataset.py::Dataset.concatenate` was extended to permit append-only aux schema mixing when all core input/action contracts match.

Verified case:

- v2 / 36-target data + v3 / 44-target data
- old samples are padded to 44 targets
- appended target values are zero
- appended `aux_valid_mask` is exactly zero

A real merge of Gen7 v2 data and Gen15 v3 data passed sanity checking, with appended-valid sum **0.0** for the old rows.

This is safe because missing auxiliary labels are not invented; they are explicitly masked invalid.

Do not generalize this to arbitrary schema changes. The code still requires matching tokenizer, observation, action, ruleset, and other core contracts.

## Elite APP replay curriculum

`trainer/train.py` now has default-OFF options:

```text
--elite-app-threshold FLOAT
--elite-app-repeat INT
```

It reconstructs per-player trajectory APP from the placement `0..1` attack auxiliary target and increases sampling multiplicity for trajectories at/above the threshold. Labels/rewards/loss formulas are unchanged and non-elite data remains in the replay.

This was useful for discovering Gen24, but ordinary 1v1 APP-selected trajectories should **not** be assumed to be pure stacking experts. The clean benchmark showed that an earlier 1v1-elite specialist could have high adversarial APP yet stack worse with attack delivery disabled.

## Important failed idea: ordinary 1v1 elite != clean stacking elite

A policy-only candidate trained from recent high-APP 1v1 trajectories appeared good under ordinary self-play (e.g. fixed seeds rose from ~0.09 to ~0.16 APP), yet in the clean no-delivery benchmark it could be worse than Gen14 and top out early.

Interpretation: high 1v1 sent APP contains opponent-pressure/cancellation/context selection. It is not an uncontaminated stacking target.

Future firepower curricula should increasingly use **clean no-delivery trajectories as the source of elite labels**, while Arena remains the independent strength constraint.

## Recommended next training direction

Do not resume broad hyperparameter ablations. The goal is still to learn stacking until clean mean APP > 0.5.

Recommended sequence:

1. Keep Gen14 frozen as strength champion.
2. Keep Gen24 as a firepower teacher, not a competitive parent.
3. Keep Gen25 as a reference balance point, not champion.
4. Generate a larger **clean no-attack-delivery corpus** using Gen14 and Gen24 at fixed search settings. Prefer enough seeds to find genuinely high clean-APP trajectories rather than reusing ordinary 1v1 elites.
5. Reconstruct clean per-player APP from those datasets and build the next stacking curriculum only from clean-domain elites.
6. Train from Gen14 (or a proven Arena-neutral checkpoint) with policy head only first. Avoid trunk/value/aux changes until the policy-only path is exhausted.
7. After every candidate, gate in this order:
   - clean fixed-seed APP must improve materially over Gen14;
   - frozen Arena seed42 + seed1337 must not show a clear regression;
   - only then spend more games to reduce confidence interval width.
8. If clean APP improves but Arena regresses again, do **not** blindly replay competitive data for many steps: Gen25 shows this can erase the whole firepower gain. Prefer smaller source-aware updates / constrained replay rather than sequential overwrite.
9. Timing remains off regardless of cancellation quality until the fixed clean multi-seed mean exceeds **0.5 APP**.

A sensible engineering improvement before the next large training run is source-aware sampling (explicit competitive-vs-clean-expert batch ratio) rather than approximating source balance by duplicating entire files. The Gen27 failure shows raw dataset-count mixing is too crude.

## Timing status — intentionally deferred

Earlier timing work discovered that raw Gen14 strongly extrapolates toward an unseen WAIT delay feature and can collapse PPS. Timing factorization/rank infrastructure and experimental timing-head code exist in the tree, but **do not continue this work now**.

The timing restart condition from the user is:

> fixed clean stacking benchmark mean APP > 0.5

Until then:

- no timing self-play for production
- no WAIT_FOR_EVENT training
- no cancellation-vs-timing ablation
- no timing-head promotion work

## Validation / build state

After the clean-benchmark protocol change:

```text
287 tests
1,055,113 assertions
0 failed
```

The Windows GPU engine `build-win/tetra_cli.exe` was rebuilt successfully afterward and real `--no-attack-delivery` GPU runs were completed.

`trainer/interpolate_policy_checkpoints.py` passes `py_compile` and was used to create the alpha 0.5 / 0.75 diagnostic checkpoints.

## Files created/modified during this phase

Important new/modified pieces:

- `trainer/train.py`
  - elite APP replay curriculum
  - timing flags/loss wiring from earlier phase remain default off
- `trainer/tetra_dataset.py`
  - append-only aux-schema concatenation with invalid-mask padding
- `trainer/gpu_selfplay.py`
  - `--no-attack-delivery`
  - timing diagnostic flags remain available but should stay off
- `tools/tetra_cli.cpp`
  - GPU export protocol accepts attack-delivery enable/disable
- `trainer/interpolate_policy_checkpoints.py`
  - safe policy-only checkpoint interpolation diagnostic
- timing-related files from the previous phase remain present but deferred

## Short status line for the next agent

**Gen14 is still champion. Gen24 raises clean APP from 0.173 to 0.219 but loses Arena 2-6. Gen25 recovers Arena to 5-3 but clean APP returns to 0.174. Timing is frozen until clean mean APP > 0.5. Next work should generate/source clean-domain high-APP teachers and use explicit source-aware competitive-vs-clean replay, with every candidate gated by both clean APP and frozen Arena.**
