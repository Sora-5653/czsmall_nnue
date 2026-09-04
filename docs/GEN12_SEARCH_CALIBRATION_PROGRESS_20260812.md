# Gen12 Search Calibration Progress — 2026-08-12

## Current conclusion

The biggest finding is that several apparent model regressions came from the evaluation search regime rather than from the network itself. `docs/adr/0008-search-gumbel-calibration.md` already states that PUCT is unreliable below roughly two simulations per legal action. Current TETR.IO positions have about 42 legal actions on average, so PUCT32 is below that regime and must not be used as the primary strength verdict.

Primary low-budget Arena should use **Gumbel sequential halving**. PUCT should be an explicit ablation, preferably around 128+ simulations.

## Important checkpoints

- `models/gen7_raw_aux_20260807.best.pt` — conservative stable baseline.
- `models/gen9_fixed_rank010_20.best.pt` — earlier rank-loss parent.
- `models/gen10b_rank010_30_20260810.best.pt` — policy-head-only descendant. Gen7 and Gen10b have bit-identical trunk/value weights; only 9 policy tensors differ.
- `models/gen12_calibrated_gumbel20_20260811.best.pt` — soft-CE distillation; offline metrics improved but Arena worsened. Do not promote.
- `models/gen12_hardrank12_20260811.pt` — hard-rank-only model; ranking and prior sharpness improved, and the frozen-search re-evaluation later promoted it over Gen7. **Current model-strength champion/reference.**
- `models/gen12_disagree12_20260811.pt` — disagreement-only chosen-action distillation; no useful Arena gain.

## VS Score

Exact TETR.IO-style VS is now tracked:

```text
VS = ((lines_sent + garbage_lines_cleared) / pieces) * PPS * 100
```

Use metrics in this order:

1. Win rate + Wilson confidence interval
2. VS
3. APM / APP / PPS

Several experiments produced higher VS but lower win rate, so VS is useful pressure diagnostics but not a substitute for direct match outcome.

## Aux schema v3

Aux schema widened from 36 to 44 outputs by appending future physical garbage-clear targets at 1/2/4/8-second and 1/2/4/8-placement horizons. Old 36 outputs keep their indices, so migration preserves old weights.

Gen11 v3 data currently totals about **44 independent games / 9,214 samples**.

The current aux/value experiments showed that dense ground-truth signals can be useful, but predicted aux outputs do not generalize well enough to blend into search value yet. Keep the targets for diagnostics/future representation work, but do not use current aux predictions as MCTS leaf value.

## Value-head finding

The earlier hypothesis that Gen10b's 32-sim weakness came from value/trunk was disproved:

```text
Gen7 vs Gen10b non-policy max parameter difference = 0.0
changed non-policy tensors = 0
changed policy tensors = 9
```

Current WDL value is informative only near terminal states. On held-out positions, AUC is about 0.67 inside <=8 placements to terminal, but about 0.50 at 9–32 and >32 placements. Value-only retraining lowered CE/MSE mainly by shrinking predictions toward neutral and did not improve Arena.

## Search-default bug found and fixed

`SearchConfig` already defaults to `use_gumbel=true`, but Python GPU evaluation scripts had been explicitly sending `gumbel=false` unless `--gumbel` was supplied. This silently forced many Arena runs into PUCT32.

`trainer/gpu_arena.py` and `trainer/gpu_match.py` now default to Gumbel. Use `--puct` only for an explicit PUCT experiment.

Direct Gen7 comparison at 32 sims:

- seed 42: Gumbel 7–1 PUCT
- seed 1337: Gumbel 7–1 PUCT
- combined: **14–2**

## Policy temperature

For PUCT32, `T=0.8` strongly improved the same checkpoint against itself:

- Gen10b: 31–17 combined vs T=1.0
- Gen7: 36–12 combined vs T=1.0

But under Gumbel32, Gen7 T=0.8 vs T=1.0 gave:

- 4–4
- 3–5
- combined **7–9**

Therefore `T=0.8` is mainly a low-budget PUCT calibration workaround, not a universal network or Gumbel improvement.

## Gen12 distillation results

Soft CE on the new Gumbel data improved held-out CE/ranking slightly but flattened the prior and worsened Arena.

Hard-rank-only training was more promising:

- teacher-best top1: 61.1% -> 62.2%
- MRR: 0.7905 -> 0.7954
- model top1 probability: 0.169 -> 0.191
- teacher-mass regret: 0.0043 -> 0.0041

This improved ranking and sharpness together, but Arena remained approximately even with Gen10b. Keep as an experimental checkpoint, not champion.

Gumbel `chosen_action` is also important: Gen10b agrees with it about 79.3% of the time even though visit-distribution argmax agreement is only about 61.1%. A simple disagreement-only loss did not improve chosen-action generalization enough.

## Gumbel noise calibration

Arena now supports separate candidate/champion `gumbel_noise_scale` values through:

- `include/tetra/arena.hpp`
- `tools/tetra_cli.cpp`
- `trainer/gpu_arena.py`

Current Gen7-vs-itself Gumbel32 results:

- noise 0.10 vs 0.05, seed42: **3–5**, VS 187.0 vs 226.1
- noise 0.02 vs 0.05, seed42: **5–3**, VS 231.8 vs 170.6
- noise 0.02 vs 0.05, seed1337: **4–4**, VS 186.2 vs 221.9

Combined 0.02 vs 0.05 is only 9–7, so there is not enough evidence to replace the existing 0.05 default.

### 2026-08-14 completion: noise 0.00 vs 0.05

The previously unfinished no-perturbation test is now complete under the same Gen7-vs-itself Gumbel32 setup:

- noise 0.00 vs 0.05, seed42: **5–3**, VS **201.4 vs 169.6**
- noise 0.00 vs 0.05, seed1337: **4–4**, VS **191.6 vs 216.1**
- combined: **9–7** for noise 0.00 over 16 games

This is effectively the same weak signal as noise 0.02 vs 0.05 (also 9–7), and the VS direction reverses between seeds. There is therefore no evidence strong enough to remove Gumbel perturbation or replace the existing 0.05 default.

**Freeze the primary low-budget search calibration at Gumbel32, `gumbel_noise_scale=0.05`, policy temperature `T=1.0`.** Search-calibration experiments may override these explicitly, but ordinary model-strength comparisons should not.

## Evaluation protocol from now on

Primary Arena:

- Gumbel sequential halving
- 32 simulations initially
- equal search settings on both sides unless explicitly doing search calibration
- policy temperature 1.0 by default
- at least two independent seeds
- paired/mirrored games
- win rate + Wilson CI primary
- VS secondary

The Gumbel32 search configuration is frozen and must remain the ordinary model-strength protocol. Promotion decisions below use the Arena implementation's rule: win rate at least 55% and Wilson 95% lower bound above 50%.

## 2026-08-14 continuation: frozen-search checkpoint re-evaluation

The requested Gen7-reference re-evaluation is complete. All runs used Gumbel32, `gumbel_noise_scale=0.05`, policy `T=1.0`, equal candidate/champion search settings, paired/mirrored games, and seeds 42 / 1337.

### Gen9 vs Gen7

- seed42, 16 games: **8-8**, VS **184.2 vs 196.9**
- seed1337, 16 games: **12-4**, VS **226.3 vs 176.4**
- combined: **20-12 = 62.5%**, Wilson 95% CI **45.3%-77.1%**
- equal-size weighted VS: about **205.3 vs 186.7**

This is directionally positive but not promotion-grade evidence because the confidence interval still crosses the threshold substantially.

### Gen10b vs Gen7

The first 4-pair seed42 attempt exceeded the DevSpace 300-second command limit, so the collected runs use 3 pairs / 12 games per seed.

- seed42: **8-4**, VS **200.3 vs 244.4**
- seed1337: **6-6**, VS **198.7 vs 246.5**
- combined: **14-10 = 58.3%**, Wilson 95% CI **38.8%-75.5%**
- weighted VS: about **199.5 vs 245.5**

There is no promotion case here. Win rate is mildly positive while VS is consistently worse.

### Gen12-hardrank vs Gen7

Initial sample:

- seed42, 12 games: **7-5**, VS **205.1 vs 261.8**
- seed1337, 8 games: **7-1**, VS **164.6 vs 240.0**
- initial combined: **14-6 = 70.0%**, Wilson 95% CI **48.1%-85.5%**

The seed1337 3-pair attempt exceeded the 300-second command limit and was rerun at 2 pairs. Because the initial lower bound was still below the Arena's >50% confidence requirement, the sample was expanded under the frozen protocol with independent seeds rather than tuning search.

Additional fixed-protocol runs:

- seed20260814, 8 games: **6-2**, VS **189.1 vs 234.9**
- seed424242, 8 games: **5-3**, VS **218.6 vs 177.1**
- after those two seeds: **25-11 = 69.4%**, Wilson 95% CI **53.1%-82.0%**

To avoid optional stopping immediately after crossing the implementation's confidence rule, two further seeds were fixed and run to a total of 52 games:

- seed271828, 8 games: **6-2**, VS **220.5 vs 235.5**
- seed8675309, 8 games: **6-2**, VS **238.7 vs 200.9**

Final pooled result:

- **37-15 = 71.2%** over 52 games
- Wilson 95% CI: **57.7%-81.7%**
- weighted VS: about **206.0 vs 227.9**

This now clears the Arena implementation's promotion rule (`win_rate >= 0.55` and `ci_lower > 0.50`) with margin. **Promote `models/gen12_hardrank12_20260811.pt` as the new model-strength champion/reference.** Gen7 remains the historical stable baseline.

The important unresolved result is that Gen12-hardrank wins much more often while still having lower aggregate VS than Gen7. That is now unlikely to be explained purely by one or two lucky seeds and should be investigated as a difference in survival/efficiency/timing rather than treated as a reason to reject the promotion.

There is currently no canonical `models/champion.pt` file in this checkout, so no binary alias was fabricated. Future evaluation/self-play should use `models/gen12_hardrank12_20260811.pt` directly as champion unless a canonical champion-file workflow is explicitly introduced.

## VS auxiliary loss: implementation completed and validated

The previously interrupted default-off VS auxiliary loss is now fully wired and tested. Its target is schema-v3 short-horizon pressure rate:

```text
VS / 100 = (attack + garbage_cleared) / seconds
```

Schema v3 stores `real_0_1s`, `real_1_2s`, `real_2_4s`, and `real_4_8s` attack/garbage-clear counts as disjoint intervals. `trainer/tetraformer.py` therefore cumulatively sums those intervals and divides by 1/2/4/8 seconds. A horizon is valid only if both attack and garbage-clear channels are valid, and validity is cumulative so an unknown earlier interval invalidates later cumulative horizons.

Completed plumbing:

- `trainer/tetraformer.py`: VS loss, exact unclamped valid-count reporting, schema-v3 guard, loss/gradient diagnostics.
- `trainer/train.py`: validation accumulation weighted by VS-valid horizon count, validation total integration, and start/step/interim/end reporting.
- `trainer/test_vs_aux.py`: 5 unit tests covering cumulative interval semantics, cumulative validity, zero-valid handling, total-loss contribution, and schema-v2 rejection.

Validation:

- Python: **12 tests passed** across `test_vs_aux`, `test_colab_generate`, and `test_colab_manual`.
- GPU smoke on RX 9070 XT, schema-v3 data, one step: held-out total **4.8406 -> 4.8076**, VS loss **0.0313 -> 0.0268**; non-zero VS shared-trunk gradient observed.
- Full C++ suite after the later Arena diagnostic extension is **286 tests / 1,050,435 assertions / 0 failed**.

`--vs-aux-weight` is now a functional, default-off experimental option. It is no longer an incomplete feature, but it is **not established as a strength improvement**.

## Gen13 paired VS-aux ablation

A controlled ablation used `models/gen10b_rank010_30_20260810.best.pt` as the common parent. Both arms widened aux 36 -> 44 from the same seed, reset optimizer/sampling identically, used the same 44-game / 9,214-sample Gen11 schema-v3 dataset, batch 256, LR `2e-5`, policy/value/aux/rank weights `1 / 0.05 / 1 / 0.1`, and differed only by `vs_aux=0` versus `vs_aux=0.1`.

Checkpoints:

- control: `models/gen13_vsaux_control30_20260814.best.pt`, then `models/gen13_vsaux_control100_20260814.best.pt`
- VS 0.1: `models/gen13_vsaux_w010_30_20260814.best.pt`, then `models/gen13_vsaux_w010_100_20260814.best.pt`

At 30 steps:

- control held-out: policy **2.7622**, value **0.7537**, VS loss **0.0143**
- VS 0.1 held-out: policy **2.7622**, value **0.7541**, VS loss **0.0140**
- direct Arena seed42: VS arm **5-3**, VS score **163.5 vs 177.6**
- direct Arena seed1337: VS arm **6-2**, VS score **201.6 vs 161.7**
- combined Arena: **11-5 = 68.8%**, Wilson 95% CI **44.4%-85.8%**

At 100 steps:

- control held-out: policy **2.7488**, value **0.7393**, VS loss **0.0138**
- VS 0.1 held-out: policy **2.7488**, value **0.7390**, VS loss **0.0134**
- direct Arena seed42: VS arm **3-5**, VS score **157.0 vs 194.1**
- direct Arena seed1337: VS arm **4-4**, VS score **207.1 vs 165.5**
- combined Arena: **7-9 = 43.8%**, Wilson 95% CI **23.1%-66.8%**

Interpretation: the dedicated VS term gives a small, repeatable improvement in the held-out pressure target while leaving policy/value metrics essentially unchanged, so the implementation appears safe at weight 0.1. However, the apparent 30-step Arena advantage disappears by 100 steps. **Do not promote either Gen13 arm and do not claim a strength gain from VS aux.** Keep the feature default-off for future representation/auxiliary experiments.

## Arena behavioral diagnostics: Gen12 wins by board safety, not raw pressure

The Arena result protocol now exposes aggregate behavioral diagnostics without changing search, game transitions, scoring, or the win/loss rule. Added fields include APM/APP/PPS, average pieces and duration, survival, sent/received/garbage-cleared lines per game, and BlockOut/LockOut/GarbageOut rates. The GPU binary protocol and Python reporter were extended accordingly, with aggregate-counter consistency covered by the C++ test suite.

Frozen-protocol Gen12-hardrank vs Gen7 diagnostic reruns:

### seed1337, 8 games

- result: **7-1** for Gen12
- VS: **164.6 vs 240.0**
- APM / APP / PPS: **81.6 / 0.098 / 13.873** vs **141.2 / 0.172 / 13.678**
- survival: **87.5% vs 12.5%**
- sent / received / garbage-cleared per game: **14.25 / 10.00 / 3.00** vs **24.75 / 4.50 / 0.50**
- topout BlockOut / LockOut / GarbageOut: **12.5% / 0.0% / 0.0%** vs **75.0% / 12.5% / 0.0%**

### seed42, 8 games

- result: **5-3** for Gen12
- VS: **231.0 vs 246.5**
- APM / APP / PPS: **123.3 / 0.149 / 13.820** vs **135.2 / 0.167 / 13.488**
- survival: **62.5% vs 37.5%**
- sent / received / garbage-cleared per game: **18.12 / 7.12 / 2.25** vs **19.88 / 4.62 / 1.88**
- topout BlockOut / LockOut / GarbageOut: **12.5% / 25.0% / 0.0%** vs **25.0% / 37.5% / 0.0%**

Across both diagnostic samples, Gen12 sends less and has lower VS/APM/APP, yet survives substantially more often. Gen7 receives **less** garbage than Gen12 in both samples but dies mainly by BlockOut/LockOut; neither side records GarbageOut in these runs. This rules against the simple explanation that Gen7 is merely being overwhelmed by incoming pressure. The strongest current interpretation is that Gen12's hard-rank training shifted policy toward materially safer stacking / garbage handling, sacrificing raw pressure for terminal survival. The exact learned tactical motif is still unknown, but the win-rate-vs-VS divergence is no longer diagnostically mysterious.

## 2026-08-14 continuation: initial Gen12-champion self-play batch

The production self-play configuration was re-checked before generation. It is deliberately different from the frozen low-budget Arena configuration:

- checkpoint: `models/gen12_hardrank12_20260811.pt`
- Gumbel search enabled
- **64 simulations**
- **2 determinizations**
- root exploration noise fraction **0.25**
- policy temperature **1.0**
- maximum **300 pieces**
- fp16 GPU inference on `cuda:1`
- timing actions remain off
- dataset model version **14**

A one-game production smoke succeeded first. Parallel 4-worker and 2-worker attempts then hit DevSpace's 300-second command limit because the outer process waits for the slowest game, but shards that had already completed were intact. Every retained shard was loaded through `trainer/tetra_dataset.py` and passed `sanity_check()` before use.

The production batch has now been expanded to **32 independent games / 9,665 samples / 165.3 MiB**. It contains seed `14082026` plus every seed from `14082030` through `14082060`, with no missing or duplicated seeds. Every retained shard was loaded independently and passed `Dataset.sanity_check()`.

Aggregate dataset statistics:

- games: **32**
- samples: **9,665**
- samples/game: mean **302.0**, median **279**, range **88-597**
- terminated games: **29**
- 300-piece truncated games: **3**
- terminated samples: **7,879**
- truncated samples: **1,786**
- schema: rectangular v4, aux schema v3 / 44 targets, model version 14
- loaded search-target alignment: **9,221 / 9,665 = 95.41%**
- mean chosen-action visit mass: about **0.2298**

The apparent discrepancy between exact `argmax(policy_target) == chosen_action` and the exporter alignment diagnostic was only tie handling: the exporter asks whether the chosen action has mass within `1e-6` of the maximum. Recomputing the same criterion after loading matches the exporter. There is no evidence of serialization corruption.

The expanded self-play also shows that Gen12 is not intrinsically a low-firepower policy. For example, seed `14082050` produced **APP 0.295 / APM 242.2 / VS 458.0** while winning. This supports the interpretation from the Arena diagnostics that Gen12 retains high-pressure capability but more selectively avoids unsafe pressure/stacking states, rather than simply having lost offensive capacity.

### Guarded-generation loss inheritance bug prevented

`gen12_hardrank12` stores an ablation objective in its checkpoint:

```text
policy=0, value=0, aux=0, policy_rank=1
```

Previously, `trainer/iterate.py` passed only a value override to `train.py`; therefore a promoted ablation checkpoint could silently carry experiment-specific loss weights into the next ordinary generation. With Gen12-hardrank this would have produced rank+value training instead of the intended production objective.

`trainer/iterate.py` now explicitly defines and forwards the current production fine-tuning profile instead of inheriting checkpoint loss weights:

```text
policy=1
value=0.05
aux=1
policy_rank=0.1
chosen_action=0
chosen_disagreement=0
policy_pair_rank=0
vs_aux=0
topout_aux=0
policy_target_temperature=1
lr=2e-5
```

It also exposes `--reset-optimizer` / `--reset-sampling`, which should be used for the first ordinary generation after an objective-changing ablation such as Gen12-hardrank. Generation summaries now record the self-play and training configuration so the run is reproducible.

### 30-step production-objective smoke

A deliberately non-promotion 30-step smoke was run from Gen12-hardrank on only the 2,699 new samples, with optimizer and sampling reset and the explicit production profile above:

- checkpoint: `models/gen14_prod_smoke30_20260814.best.pt`
- step 3212 -> 3242
- held-out total: **3.0647 -> 2.7993**
- held-out policy: **2.8117 -> 2.5785**
- held-out value: **0.8470 -> 0.7622**
- held-out value MSE: **1.1055 -> 1.0637**

This proves that ordinary policy/value/aux/rank training is wired correctly after the hard-rank champion, but **does not establish a strength gain**. Frozen-search smoke Arena against Gen12 was inconsistent:

- seed42, 8 games: candidate **2-6**, VS **188.7 vs 215.2**, survival **50.0% vs 87.5%**
- seed1337, 4-game reduced run after the 8-game command timed out: candidate **3-1**, VS **233.1 vs 163.9**, survival **75.0% vs 25.0%**
- collected total: **5-7** over 12 games, with unequal per-seed sample sizes and no promotion-grade interpretation

The seed42 run also showed the candidate returning to BlockOut/LockOut deaths, while seed1337 reversed direction. The correct conclusion is that **2,699 samples are too small for a serious next-generation strength verdict**. Do not promote this smoke checkpoint and do not tune around these Arena results.

### Serious Gen14 fit on the completed 32-game self-play set

Once the production set reached 32 games / 9,665 samples, a clean candidate was trained directly from Gen12-hardrank with optimizer and sampling reset, `new_data_repeat=1`, LR `2e-5`, and the explicit production objective `policy/value/aux/rank = 1 / 0.05 / 1 / 0.1`.

At 100 total steps (`models/gen14_selfplay32_100_20260814.best.pt`, step 3312):

- held-out total: **3.1169 -> 2.8005**
- policy: **2.8328 -> 2.5648**
- value MSE: **0.7987 -> 0.7183**
- frozen Arena seed42, 8 games: **5-3** vs Gen12; VS **157.4 vs 204.3**; survival **62.5% vs 37.5%**
- frozen Arena seed1337, 8 games: **3-5**; VS **159.2 vs 181.1**; survival **50.0% vs 62.5%**
- combined: **8-8** over 16 games; no promotion signal

Because validation was still improving and there was no catastrophic 100-step regression, the same optimizer/sampling state was continued to 200 total steps without resetting. At 200 steps (`models/gen14_selfplay32_200_20260814.best.pt`, step 3412), held-out total improved slightly again to **2.7902**, but independent Arena seeds moved in the wrong direction:

- seed20260814, 4-game reduced run after an 8-game command timeout: **1-3**; candidate survival **50.0%**, BlockOut **50.0%**
- seed424242, 4 games: **1-3**; candidate survival **25.0%**, BlockOut **75.0%**
- collected independent-seed total: **2-6**

Do not promote either Gen14 checkpoint. The 100-step candidate is approximately neutral in the initial two-seed sample; the 200-step candidate is directionally worse on two independent seeds despite better validation loss. The behavioral diagnostics make the failure mode especially relevant: longer training under the restored ordinary CE/value/aux/rank objective appears to reintroduce unsafe BlockOut behavior that Gen12-hardrank had suppressed.

This was enough to stop extending the same rank=0.1 objective and run a controlled rank-preservation sweep on the exact same 9,665 samples.

### Gen14 rank-preservation sweep: rank=0.3 vs rank=1.0

Both new arms resumed directly from `models/gen12_hardrank12_20260811.pt`, used the identical 32-game dataset, seed `20260814`, fresh optimizer/sampling state, 100 steps, batch 256, LR `2e-5`, and `policy/value/aux = 1 / 0.05 / 1`. Only `policy_rank` changed.

Rank 0.3 checkpoint: `models/gen14_rank030_100_20260814.best.pt`.

- held-out policy CE ended at **2.6011**
- frozen Arena seed42, 4-game reduced run: **0-4** vs Gen12; survival **25% vs 100%**; candidate BlockOut/LockOut **25% / 50%**
- seed1337, 4 games: **2-2**; survival **50% / 50%**
- collected total: **2-6**; no promotion case

Rank 1.0 checkpoint: `models/gen14_rank100_100_20260814.best.pt`.

The initial fixed seeds were then extended with four independent seeds, all under the same frozen Gumbel32/noise0.05/T1.0 protocol. Each block contains 4 games:

- seed42: **3-1**, VS **260.8 vs 220.6**, survival **75% vs 50%**
- seed1337: **3-1**, VS **203.4 vs 201.6**, survival **100% vs 50%**
- seed20260814: **3-1**, VS **247.4 vs 205.5**, survival **75% vs 50%**
- seed424242: **4-0**, VS **174.5 vs 115.9**, survival **100% vs 0%**
- seed271828: **3-1**, VS **199.2 vs 176.6**, survival **75% vs 50%**
- seed8675309: **3-1**, VS **212.2 vs 151.9**, survival **75% vs 25%**

Final pooled result: **19-5 = 79.2% over 24 games**, Wilson 95% CI **59.5%-90.8%**. This clears the Arena implementation promotion criterion with margin even after the two extra seeds fixed after the initial threshold crossing.

Equal-block aggregate behavioral diagnostics are also favorable:

- VS about **216.3 vs 178.7**
- survival about **83.3% vs 37.5%**
- candidate BlockOut / LockOut about **8.3% / 8.3%**
- Gen12 BlockOut / LockOut about **33.3% / 29.2%**
- GarbageOut remained 0% in these blocks

A held-out policy-ranking audit on 2,050 hashed samples helps explain the result. Compared with Gen12-hardrank, rank=1.0 improves teacher top-1 agreement **0.6146 -> 0.6273** and MRR **0.7830 -> 0.7885**. By contrast the rank=0.1 control achieves much better CE (**2.5552**) but slightly worse top-1 (**0.6073**) and MRR (**0.7760**) than Gen12. Rank=1.0 CE is **2.7302**, so the Arena gain is associated with preserving/improving action ordering rather than maximizing distributional CE fit.

**Promote `models/gen14_rank100_100_20260814.best.pt` as the new model-strength champion/reference.** This is the first Gen14 candidate that both preserves the Gen12 safety phenotype and improves direct Arena strength. `trainer/iterate.py` now defaults the production `--policy-rank-weight` to **1.0** so a subsequent ordinary generation does not silently fall back to the demonstrated-bad rank=0.1 profile.

### Gen15 completed self-play and failed next-generation policy-only improvement

Self-play from the promoted Gen14 champion was expanded from the initial 8 games to a complete **32 independent games / 13,146 samples / about 225 MiB** using seeds `15082000` through `15082031`.

Final Gen15 dataset properties:

- checkpoint: `models/gen14_rank100_100_20260814.best.pt`
- Gumbel64, 2 determinizations, root-noise fraction 0.25, T=1.0
- 300-piece cap, fp16, `cuda:1`, timing actions off
- model version 15
- games: **32**
- samples: **13,146**
- mean samples/game: **410.8**
- median samples/game: **380.5**
- range: **153-599**
- search-chosen / visit-max tie-aware alignment: **13,038 / 13,146 = 99.18%**
- mean chosen-action visit mass: about **0.2341**
- terminated games: **24**
- truncated games: **8**
- all 32 shards use schema v4 container / aux schema v3 with 44 targets and pass `sanity_check()`

The new champion therefore generated substantially longer trajectories than the Gen12-derived Gen14 corpus while maintaining almost perfect agreement between the action actually executed by search and the visit-max target.

Representative trajectories also confirm that Gen14's safety phenotype does not imply low offensive capacity. The most extreme early example, seed `15082005`, won with **APP 0.424 / APM 350.1 / VS 671.6**.

A separate dataset-density audit found that **5,104 / 13,146 = 38.8%** of samples had pending garbage. Of those pending-garbage states, about **54.8%** had at least one line-clearing legal action and about **26.5%** actually chose a clearing action. This later became important when deciding whether explicit cancellation supervision would be dense enough to be useful.

#### Gen15 ordinary rank=1.0 update

A 100-step closed-loop update from Gen14 used the 32-game Gen15 corpus, LR `2e-5`, and the current production objective `policy/value/aux/rank = 1 / 0.05 / 1 / 1`.

Checkpoint:

- `models/gen15_rank100_selfplay32_100_20260814.best.pt`

Held-out total improved slightly, **4.1310 -> 4.0643**, but frozen Arena against Gen14 over four independent 4-game blocks was exactly neutral:

- seed42: **1-3**
- seed1337: **2-2**
- seed20260814: **3-1**
- seed424242: **2-2**
- pooled: **8-8 / 16 games**

The policy-ranking audit exposed the same failure mode seen before Gen14's rank correction. Relative to the Gen14 parent on a fixed Gen15 held-out set:

- CE: **2.8679 -> 2.8488** improves
- teacher top-1: **65.53% -> 65.12%** worsens
- MRR: **0.8185 -> 0.8093** worsens
- search-chosen top-1 agreement: **87.93% -> 83.34%** worsens

Thus better soft-distribution CE was again not equivalent to better low-budget search priors.

#### Gen15 update ablations

Several controlled variants were run from the same Gen14 parent and fixed 13,146-sample corpus.

1. **50-step rank=1 update**
   - checkpoint: `models/gen15_rank100_selfplay32_50_20260814.pt`
   - still reduced top-1/MRR/chosen agreement relative to the parent
   - merely shortening the update did not solve the problem

2. **chosen-action CE = 0.1 / 0.3**
   - `chosen=0.3` improved teacher top-1 to **66.40%** and MRR to **0.8176**
   - Arena for `chosen=0.3` was **4-4 / 8 games** on seeds42/1337
   - sharpening the action search actually executed was therefore insufficient for a strength gain

3. **policy CE weight = 0.3, rank=1**
   - teacher top-1 fell to **64.83%**, MRR to **0.8059**
   - simply reducing soft CE was counterproductive

4. **rank=3**
   - teacher top-1 fell to **64.65%**, MRR to **0.8051**
   - the issue was not simply insufficient rank-loss magnitude

5. **Gen14+Gen15 replay window, rank=1**
   - 22,811 samples total
   - old Gen14-distribution top-1 was preserved almost perfectly (**62.73% -> 62.78%**)
   - new Gen15-distribution top-1 still fell **65.53% -> 63.73%**
   - catastrophic forgetting alone therefore does not explain the problem

6. **hard-rank + chosen-disagreement only**
   - objective: `policy=0, value=0.05, aux=1, rank=1, chosen_disagreement=1`
   - checkpoint: `models/gen15_hardrank_disagree100_20260814.best.pt`
   - teacher top-1 improved **65.53% -> 68.31%**
   - MRR improved **0.8185 -> 0.8298**
   - CE intentionally deteriorated to about **4.08**
   - nevertheless Arena on seeds42,1337,20260814,424242 was **2-2 on every seed, 8-8 pooled**

This is an important stopping result. Gen15 shows that **further improvement of policy ordering alone no longer reliably increases direct playing strength**. Gen14 remains the champion. Do not continue blind policy-loss/rank sweeps on this corpus.

### Cancellation / garbage-management supervision: schema v4 / 52 targets

Because the next missing strategic dimension appears to be garbage/cancellation management rather than static placement ranking, the auxiliary schema was extended without changing the game reward or enabling timing actions.

The simulator already computes exact cancellation ground truth as `LockResult.garbage_cancelled`. This avoids any heuristic label construction. `GameRecorder` now records cancellation as its own trace event, separate from attack sent and garbage cleared.

Auxiliary schema evolution is now:

- schema v1: 4 legacy targets
- schema v2: 36 interval targets
- schema v3: 44 targets, adding garbage-clear channels
- **schema v4: 52 targets, adding 8 garbage-cancellation channels**

The 8 new targets are disjoint cancellation counts over:

- real time: `0-1s`, `1-2s`, `2-4s`, `4-8s`
- placement distance: `0-1`, `1-2`, `2-4`, `4-8` placements

Implementation details:

- C++ schema and `TrainingSample` width updated to 52
- `GameRecorder` records exact `garbage_cancelled`
- real-time and placement-window aggregation added
- schema-v3 44-target files remain readable
- Python `tetra_dataset.py` and Colab manifest validation understand schema v4
- existing 44-output checkpoints can be widened with `--upgrade-aux-schema`
- migration copies the original 44 aux output rows exactly and initializes only the new 8 rows
- dedicated `--cancellation-aux-weight` added to `train.py`
- cancellation loss is extra emphasis on the 8 cancellation channels; these channels also remain part of generic aux MSE
- `trainer/iterate.py` explicitly forwards `cancellation_aux_weight=0` by default so experiment-specific weights cannot leak into production
- nonzero cancellation loss on a pre-v4 dataset fails explicitly rather than silently training nonsense
- timing actions remain **off**

A zero-step migration audit from `models/gen14_rank100_100_20260814.best.pt` confirmed:

- old aux weight rows: **bit-exact preserved**
- old aux bias rows: **bit-exact preserved**
- new rows: finite
- migrated config aux width: **52**

Migration checkpoint:

- `models/gen16_aux52_migrated_init_20260814.pt`

A 20-step cancellation-only GPU smoke (`policy=value=aux=rank=0`, `cancellation_aux=0.1`) confirmed that the learning path is live:

- held-out cancellation MSE: **0.0264 -> 0.0094**
- initial shared-trunk cancellation gradient norm: about **0.0164**
- no zero-gradient / disconnected-head problem

### Gen16 schema-v4 cancellation self-play corpus

Production self-play from the unchanged Gen14 champion was regenerated under the new 52-target schema. The policy/search configuration itself was unchanged, so differences are labels only; timing actions are still disabled.

Final fixed corpus:

- checkpoint: `models/gen14_rank100_100_20260814.best.pt`
- seeds: **16082000 through 16082031**
- games: **32 independent games**
- samples: **13,764**
- size: **236.6 MiB**
- schema: **aux v4 / 52 targets**
- no missing or duplicated seed
- every shard passes `sanity_check()`
- search-target alignment remained approximately 99% throughout generation

Cancellation target density over all 13,764 samples:

- any nonzero cancellation in any of the 8 windows: **89.66%**
- real-time nonzero rates:
  - 0-1s: **52.08%**
  - 1-2s: **49.26%**
  - 2-4s: **62.58%**
  - 4-8s: **57.61%**
- placement-window nonzero rates:
  - 0-1: **5.15%**
  - 1-2: **1.04%**
  - 2-4: **5.40%**
  - 4-8: **10.10%**

The real-time cancellation targets are therefore dense, not a mostly-zero auxiliary task.

Representative Gen16 trajectories show a healthy range of behavior:

- seed16082012: **300 pieces, 32 received, 25 garbage-clear, APP 0.153, VS 332.8**
- seed16082019: **300 pieces, 27 received, 25 garbage-clear, VS 281.4**
- seed16082023: **300 pieces, APP 0.230, APM 192.4, VS 371.8**
- seed16082031: win with **APP 0.222, VS 356.5**

Thus the cancellation corpus contains both heavy-defense long games and high-pressure offensive games; it is not merely teaching passive garbage absorption.

### Cancellation-loss calibration and full 32-game ablation

All arms below start from the same migrated 52-output checkpoint, use the exact same 32-game Gen16 corpus, identical train/validation split and RNG seed, 100 steps, LR `2e-5`, and otherwise use `policy/value/aux/rank = 1 / 0.05 / 1 / 1`.

#### cancellation_aux = 0 control

Checkpoint:

- `models/gen16_cancelaux32_control100_20260814.best.pt`

Held-out:

- total: **4.1532 -> 4.0359**
- cancellation MSE: **0.0213 -> 0.0109**

The generic aux loss alone already learns some cancellation because the new channels are included in the ordinary 52-target MSE.

#### cancellation_aux = 0.1

Checkpoint:

- `models/gen16_cancelaux32_w010_100_20260814.best.pt`

Held-out:

- total: **4.1553 -> 4.0369**
- cancellation MSE: **0.0213 -> 0.0107**

This is only about a **1.8%** cancellation-MSE improvement over control at the end, but policy/value behavior is essentially unchanged. On the fixed ranking audit:

- control top-1: **66.39%**
- cancel0.1 top-1: **66.39%**
- control MRR: **0.8195**
- cancel0.1 MRR: **0.8195**

The 8-game calibration stage had shown the same pattern: modestly improved cancellation prediction with policy metrics indistinguishable from control. A direct 4-game control-vs-0.1 Arena was **2-2**, survival **75%-75%**.

#### cancellation_aux = 1.0

Checkpoint:

- `models/gen16_cancelaux32_w100_100_20260814.best.pt`

Held-out:

- cancellation MSE: **0.0213 -> 0.0100**
- about **8% better** cancellation MSE than the no-extra-weight control
- policy CE/value remain essentially unchanged

Fixed policy-ranking audit:

- migrated parent top-1: **65.02%**, MRR **0.8161**
- control top-1: **66.39%**, MRR **0.8195**
- cancel0.1 top-1: **66.39%**, MRR **0.8195**
- cancel1.0 top-1: **66.44%**, MRR **0.8200**

Therefore strong cancellation supervision **does not appear to damage static policy ordering**. Initial cancellation gradient norm at weight1.0 was about **0.125**, then fell toward 0.03-0.04 later in training. Policy/cancellation gradient cosine stayed small in magnitude and showed no persistent strong conflict.

### Gen16 Frozen Arena: cancellation understanding alone is not yet a promotion

All candidates were evaluated with timing still off under the frozen low-budget Gumbel32/noise0.05/T1.0 protocol against the current Gen14 champion.

Control vs Gen14:

- seed42: **2-2**, survival **75% vs 75%**
- seed1337: **0-4**, survival **25% vs 100%**
- pooled current sample: **2-6**

Cancellation1.0 vs Gen14:

- seed42: **1-3**, survival **75% vs 75%**, garbage-clear **8.75 vs 6.50/game**
- seed1337: **2-2**, survival **50% vs 75%**
- pooled current sample: **3-5**

Do **not** promote either Gen16 arm. Gen14 remains the model-strength champion.

The important result is representational rather than competitive: cancellation1.0 predicts cancellation substantially better while leaving static policy ranking approximately intact, but this does not directly convert into wins while the action space has no explicit timing choice. This is consistent with the architecture: the network can learn that cancellation matters, yet cannot deliberately choose **when** to lock the same placement around a garbage-activation event.

### Interpretation and next strategic step: timing can now be introduced causally

The project is now at a cleaner transition point than before cancellation supervision existed.

We have established:

1. Gen14 rank=1.0 is a strong, safe champion.
2. Gen15 showed that further static placement-ranking improvements alone can become Arena-neutral.
3. Exact cancellation ground truth is available and dense.
4. A 52-target model can learn cancellation with no obvious damage to the existing policy ordering.
5. Cancellation-only representation improvement does not itself beat Gen14 while timing actions are unavailable.

Therefore the next experiment should **not** be another broad policy/rank sweep and should **not** immediately turn timing on for production. The next causal experiment is a bounded timing branch.

Recommended sequence:

1. Keep `models/gen14_rank100_100_20260814.best.pt` as champion/reference.
2. Keep the frozen Gumbel32/noise0.05/T1.0 Arena for every promotion verdict.
3. Keep cancellation schema v4 / 52 and use `cancellation_aux_weight=1.0` only as an experimental representation setting; production default remains 0 until a timing experiment demonstrates strength gain.
4. Enable `WAIT_FOR_EVENT` only in a **small timing-enabled self-play corpus**, leaving all other search settings fixed.
5. Before training, audit FASTEST/WAIT pairs: frequency, visit mass, search-chosen side, pending-garbage context, cancellation difference, attack difference, and whether WAIT actually straddles a relevant activation event.
6. Train a factorized timing experiment that compares the same base placement's FASTEST vs WAIT choice rather than letting timing noise distort ordinary placement learning. Existing `timing_pair` / `timing_rank` infrastructure should be reused.
7. Run paired ablations such as:
   - timing enabled + cancellation aux off
   - timing enabled + cancellation aux 1.0
   while keeping parent, data, seed, LR, ordinary policy/rank weights, and training steps fixed.
8. Do not enable broader delay bins or unrestricted timing until WAIT_FOR_EVENT alone produces a stable signal.
9. Preserve win rate, Wilson CI, survival, sent/received/garbage-clear, and BlockOut/LockOut diagnostics. Add cancellation-specific diagnostics before interpreting a timing gain as genuine counterspike/cancellation management.
10. Timing remains **off** in the current champion and ordinary production self-play until this bounded experiment passes Arena.

Non-promoted diagnostic/control checkpoints now include:

- `models/gen14_prod_smoke30_20260814.best.pt`
- `models/gen14_selfplay32_100_20260814.best.pt`
- `models/gen14_selfplay32_200_20260814.best.pt`
- `models/gen14_rank030_100_20260814.best.pt`
- `models/gen15_rank100_selfplay32_50_20260814.pt`
- `models/gen15_rank100_selfplay32_100_20260814.best.pt`
- `models/gen15_chosen010_rank100_100_20260814.best.pt`
- `models/gen15_chosen030_rank100_100_20260814.best.pt`
- `models/gen15_policy030_rank100_100_20260814.best.pt`
- `models/gen15_rank300_100_20260814.best.pt`
- `models/gen15_replay14_15_rank100_100_20260814.best.pt`
- `models/gen15_hardrank_disagree100_20260814.best.pt`
- `models/gen16_cancelaux32_control100_20260814.best.pt`
- `models/gen16_cancelaux32_w010_100_20260814.best.pt`
- `models/gen16_cancelaux32_w100_100_20260814.best.pt`

## Validation

Latest full C++ test result after cancellation-schema implementation:

```text
287 tests
1,055,113 assertions
0 failed
```

Latest focused Python regression set after cancellation implementation:

```text
16 tests
0 failed
```

The Windows GPU engine was rebuilt after the schema/cancellation changes. A real schema-v4 self-play file was generated and loaded successfully before the 32-game Gen16 corpus was produced. Existing schema-v3/44-target data was also explicitly reloaded after the change and remains compatible.

## 2026-08-15 continuation handoff

Timing work has now been intentionally deferred while static stacking/firepower is improved under an Arena-strength constraint. The authoritative continuation document is:

`docs/HANDOFF_STACKING_BALANCE_20260815.md`

Key current state: Gen14 remains champion; Gen24 is the clean-firepower specialist (3-seed clean APP 0.219 vs Gen14 0.173) but loses frozen Arena 2-6; Gen25 recovers the small frozen Arena sample to 5-3 but gives back nearly all clean APP gain (0.174). Timing must remain off until the fixed clean stacking benchmark mean APP exceeds 0.5.
