# CNN architecture ablation — 2026-08-08

## Question

Does an explicit local 2D board inductive bias improve the current TetraFormer policy when the data, split, optimiser, minibatches, number of steps, and seed are held fixed?

The existing Transformer is retained unchanged as the control architecture. The CNN and CNN+Transformer hybrid are additional comparison models; they do not replace the production model or champion.

## Fixed dataset

The ablation uses the previously generated 47,693-sample Gen-4 corpus:

- `data/production/gen4_bootstrap_20260807.part00..07.tetradat`
- `data/production/gen4_search32_[a-f]_20260807.part00..07.tetradat`

The concatenated rectangular dataset has a maximum of 102 state tokens and 106 legal actions.

The initial ablation used the existing game-seed split function once and shared it by all architectures:

- train: 37,966 samples / 512 game seeds
- held-out: 9,727 samples / 128 game seeds

No position-level random split is used. A later audit found that the production splitter sorts numeric game seeds before the 80/20 cut. Because these generation shards occupy consecutive seed ranges, the initial held-out set consisted entirely of `gen4_search32_e` and `gen4_search32_f`: bootstrap and search32 a-d contributed zero validation samples.

A second, still leakage-safe split hashes each game seed before the 80/20 assignment. This distributes validation games across every source:

- hashed train: 37,231 samples / 499 game seeds
- hashed held-out: 10,462 samples / 141 game seeds
- hashed validation includes bootstrap (4,383 samples) and every search32 group a-f (694-1,225 samples each)

## Fixed training conditions

- optimiser: AdamW
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size: 256
- steps: 400
- seed: 42
- exact sampled minibatch schedule: shared across architectures
- loss weights: policy `1.0`, value `1.0`, aux `0.1`
- gradient clipping: 1.0
- GPU: RX 9070 XT (`cuda:1` in the Windows ROCm PyTorch environment)

Parameter counts are intentionally close:

- TetraFormer-S control: 7,175,592
- CNN baseline: 7,163,880
- CNN+Transformer hybrid: 7,507,560

## Architecture details

### Transformer control

The current 8-layer, width-256 TetraFormer-S is instantiated without modification.

### CNN baseline

The tokenizer already stores the exact 10-wide occupancy bitmap in row-token features 8..17. Under the current two-player token order, the self board is the first 24 row tokens and the opponent board is recovered from the final 36 real tokens (24 rows + 10 columns + board summary + opponent counters).

The CNN receives five 24x10 planes:

1. self occupancy
2. opponent occupancy
3. self minus opponent occupancy
4. normalised y coordinate
5. normalised x coordinate

A residual 3x3 CNN encodes this board tensor. Non-board state is retained through a token-wise MLP plus masked mean pooling. Each legal action is scored conditionally on the fused state vector.

On this Windows ROCm build, ordinary `Conv2d` backward selected a MIOpen CK kernel that failed on gfx1201 with `invalid device function`. The ablation therefore implements the same spatially shared 3x3 convolution as `unfold + GEMM`, avoiding the failing MIOpen path while preserving the convolution operation.

### CNN+Transformer hybrid

A smaller board CNN produces one learned board token, which is appended to the normal TetraFormer state sequence before the 8 Transformer blocks. The existing variable-length cross-attention policy head remains intact.

## Held-out results

Primary metric: held-out policy cross-entropy after exactly 400 steps.

| Architecture | Held-out policy loss | Held-out total loss | Notes |
|---|---:|---:|---|
| Transformer | **3.244058** | 3.941302 | best policy at step 400 |
| Hybrid | 3.260342 | 4.204065 | best policy 3.255634 at step 300 |
| CNN | 3.267351 | **3.896739** | best policy at step 400 |

By policy cross-entropy alone, the original Transformer remains best. The difference is small: CNN is +0.023293 and hybrid is +0.016284 relative to the Transformer.

The CNN nevertheless has a lower final total loss than the Transformer, so the board inductive bias is helping some combination of value/auxiliary prediction even though the full policy distribution is fitted slightly less well.

## Arena

All Arena comparisons use the 400-step final checkpoints, identical search budgets, seed 42, fp16 inference, 300-piece cap, and no Gumbel search.

### Search Arena: 16 simulations each

| Candidate vs Transformer | Record | Win rate | 95% CI |
|---|---:|---:|---:|
| CNN | **34-6** | **85.0%** | 70.9%-92.9% |
| Hybrid | **28-12** | **70.0%** | 54.6%-81.9% |

The engine reports 40 games for 10 paired Arena units. CNN's interval is entirely above the 55% promotion threshold. Hybrid's point estimate is strong, but its lower confidence bound is just below 55%, so a larger Arena would be needed before treating that margin as robust.

### Policy-only Arena: 0 simulations each

| Candidate vs Transformer | Record | Win rate | 95% CI |
|---|---:|---:|---:|
| CNN | **27-13** | **67.5%** | 52.0%-79.9% |
| Hybrid | 18-22 | 45.0% | 30.7%-60.2% |

This initial policy-only result was subsequently found not to be stable enough to support the claim that the CNN raw policy is intrinsically stronger. Larger and re-split experiments below supersede that interpretation.

## Follow-up: ranking diagnostics, split audit, and independent seeds

### Teacher-ranking diagnostics

On the original 9,727-position held-out set, the Transformer is also slightly better than the CNN on direct teacher-ranking metrics:

| Metric | Transformer | CNN | Hybrid |
|---|---:|---:|---:|
| teacher argmax top-1 | **16.58%** | 16.07% | 16.91% |
| teacher-best top-3 recall | 38.46% | 38.20% | **39.64%** |
| mean reciprocal rank | 0.3355 | 0.3321 | **0.3414** |
| teacher mass at model argmax | **0.08910** | 0.08901 | 0.08865 |
| mean teacher-best rank | **8.49** | 8.58 | 8.52 |

The CNN therefore does not merely sacrifice tail calibration while ranking the teacher's best move better. The earlier policy-only Arena win is not explained by imitation ranking quality.

### Independent training seed

Repeating the ordered-split experiment with training seed 1337 preserves the policy-CE ordering:

- Transformer: 3.240603
- CNN: 3.270994

Yet the seed-1337 CNN again beats its paired Transformer in Arena:

- policy-only: 52-28 over 80 games (65.0%, 95% CI 54.1%-74.5%)
- 16 simulations: 32-8 over 40 games (80.0%, 95% CI 65.2%-89.5%)

Thus the original ordered-split result is not specific to one model-initialisation seed.

### Hashed game split

With the representative hashed game split and seed 42:

| Architecture | policy loss | value loss | value accuracy | value scalar MSE | total loss |
|---|---:|---:|---:|---:|---:|
| Transformer | **3.302256** | 0.693175 | 51.95% | 0.99975 | 3.997783 |
| CNN | 3.334443 | **0.582125** | **67.25%** | **0.80018** | **3.918749** |

The important structural separation is now clear: the Transformer is the better imitator of the search policy, while the CNN learns the WDL/value target much more successfully.

Hashed-split Arena is correspondingly different from the first policy-only result:

- policy-only, seed 777: CNN 37-43 over 80 games (46.3%, 95% CI 35.7%-57.1%)
- 16 simulations, seed 777: CNN 27-13 over 40 games (67.5%)
- 16 simulations, seed 5150: CNN 52-28 over 80 games (65.0%)

Pooling the two independent 16-simulation hashed-split trials gives 79-41 (65.8%); the Wilson 95% interval is approximately 57.0%-73.7%. The search advantage is considerably more stable than the raw-policy advantage.

### Teacher provenance

`gen4_search32_[a-f]` was generated by `models/gpu_gen_20260805_gen3.pt` at 32 simulations with no root noise. That checkpoint is itself an 8-layer, width-256 TetraFormer. Held-out policy cross-entropy therefore measures imitation of a Transformer-driven search teacher, not architecture-neutral game strength. This is one reason to give Arena substantially more weight than small CE differences.

### Split-head hybrid test

A diagnostic `split_hybrid` keeps the TetraFormer policy path structurally intact while moving value/aux to an independent compact CNN branch. At 100 steps its hashed held-out value metrics improve over the Transformer (value loss 0.6158, accuracy 64.6%, scalar MSE 0.8577), while policy CE is essentially identical (3.356448 vs 3.356361). However, the isolated value branch later overfits badly: by 400 steps value loss reaches 1.3318 despite continuing train-loss improvement. A 100-step 16-simulation Arena also loses 17-23 to the matched 100-step Transformer.

This indicates that a good static held-out value metric is not sufficient. Search probes counterfactual/off-policy positions, and the CNN baseline's shared policy/value representation appears to regularise the value function in a way the independent branch loses.

### Policy/value factorial diagnostic

Using the hashed seed-42 checkpoints, policy and value outputs were mixed at inference without retraining. On one common 20-game seed block:

- Transformer policy + CNN value vs Transformer/Transformer: 10-10
- CNN policy + Transformer value vs Transformer/Transformer: 11-9
- CNN policy + CNN value vs Transformer/Transformer: 11-9

Twenty games are too few for strength claims, but neither component swap alone produces an obvious gain. Combined with the larger full-model Arenas, this suggests policy-value compatibility/shared representation may matter, rather than one head being independently responsible.

## Spatial feature-map hybrid follow-up

The next experiment tested the architecture motivated by the CNN/Transformer division of labour directly: let a deep CNN learn exact local board geometry first, then give the Transformer a spatial feature map rather than asking attention to reconstruct cells from row bitmaps.

### Architecture

`spatial_hybrid` initially preserved all 24x10 CNN cells as 240 Transformer tokens. It used the same large local encoder as the successful CNN baseline (192 channels, 10 residual blocks), removed raw row/column board tokens from the Transformer sequence, and retained board summaries plus piece/hold/next/garbage/counter/rule/time/event state. This preserved the desired information flow, but the 240-token attention sequence made activation memory unnecessarily large; a batch-256 backward OOMed the 16 GiB RX 9070 XT.

The practical variant, `pooled_spatial_hybrid`, keeps the exact 24x10 board throughout the complete deep CNN and only then applies a 2x2 spatial bottleneck, yielding a 12x5 = 60-token CNN feature map. The Transformer therefore receives learned local-region features plus global game-state tokens. The CNN has already integrated exact cell geometry before pooling, while the Transformer does not spend quadratic attention on every cell.

### Microbatch / VRAM trade-off

The effective training batch remains 256 through gradient accumulation. On an 8-step representative benchmark, all microbatch sizes produced numerically identical learning updates; only speed and activation memory changed:

| microbatch | time / step | peak allocated | peak reserved |
|---:|---:|---:|---:|
| 32 | 0.84 s | 2.17 GiB | 2.38 GiB |
| 64 | 0.76 s | 4.13 GiB | 4.34 GiB |
| **96** | **0.74 s** | **5.97 GiB** | **6.27 GiB** |
| 128 | 0.73 s | 7.87 GiB | 8.38 GiB |

Microbatch 96 was selected: moving to 128 saved only about 1-2% wall-clock time while consuming roughly 2 GiB more VRAM. In the full 47,693-sample run, peak reserved memory remained 6.60 GiB.

The runner now saves AdamW state and supports exact continuation. The 400-step run was executed as 200 + 200 steps because of the DevSpace per-command timeout; the seed-42 minibatch schedule is regenerated for all 400 steps and the resumed run starts at schedule row 200, so this is the same update sequence as a single uninterrupted 400-step run. A separate 4-step control versus 2+2 resume check produced a maximum absolute parameter difference of exactly 0.0.

### Hashed-split 400-step result

| Architecture | policy loss | value loss | value accuracy | value scalar MSE | total loss |
|---|---:|---:|---:|---:|---:|
| Transformer | **3.302256** | 0.693175 | 51.95% | 0.99975 | 3.997783 |
| CNN | 3.334443 | 0.582125 | 67.25% | 0.80018 | 3.918749 |
| **Pooled spatial hybrid** | 3.310147 | **0.569333** | **68.25%** | **0.77934** | **3.881617** |

This is the first hybrid in the ablation series to preserve the CNN value-learning benefit without materially giving up the Transformer policy fit. Its policy CE is only +0.007891 behind the Transformer while its value metrics slightly exceed the full CNN baseline and its total loss is the best of the three.

### Arena

At 16 simulations per side, 300-piece cap, fp16, and otherwise matched search settings:

- seed 777: pooled spatial hybrid 23-17 Transformer (57.5%)
- seed 5150: pooled spatial hybrid 21-19 Transformer (52.5%)
- pooled: 44-36 over 80 games = 55.0%, Wilson 95% CI approximately 44.1%-65.4%

This is statistically compatible with parity. It does not establish a game-strength improvement over the Transformer, but unlike the earlier compact hybrids it also does not show a clear search regression.

### Inference cost

A forward-only fp16 benchmark at Arena batch size 16 and the production maximum tensor shapes gave:

| Architecture | latency / batch | throughput |
|---|---:|---:|
| Transformer | 8.271 ms | 1,934.5 positions/s |
| Pooled spatial hybrid | 13.746 ms | 1,163.9 positions/s |

The deep CNN therefore makes the hybrid about 1.66x slower per inference batch (about 40% lower position throughput) despite keeping Transformer sequence length under control. The remaining engineering question is whether later strength gains justify this extra inference cost, or whether CNN depth/channels can be reduced without losing the value-learning advantage.

## Current interpretation

The evidence now supports a narrower conclusion than the first Arena suggested:

1. **Transformer is consistently better at reproducing the Transformer-generated search policy.** This survives two training seeds and the corrected hashed split.
2. **CNN is substantially better at WDL/value learning on the same samples.** The difference is much larger than the policy-CE gap.
3. **CNN's strongest and most reproducible advantage appears under search, not policy-only play.** On the hashed split, 16-simulation Arena is about 66% across 120 games while policy-only is statistically compatible with parity.
4. **Naively bolting an independent CNN value branch onto Transformer policy does not recover the gain.** The shared CNN representation or policy-value consistency is likely important.
5. Compact shared-local-feature hybrids did **not** solve the problem. `fusion_hybrid` and `dual_policy_hybrid` still showed severe late value overfitting, so merely attaching a 64-channel/4-block CNN embedding to the Transformer was insufficient.
6. **A deep spatial CNN -> pooled feature-map -> Transformer hybrid does solve most of the representation trade-off.** The 192-channel/10-block CNN retains the strong value signal, while a post-CNN 12x5 spatial bottleneck lets Transformer policy fitting recover to within +0.0079 CE of the control. At 16-simulation Arena the current 80-game result is consistent with parity rather than a proven strength gain.
7. The main cost is now computational rather than statistical: forward inference is about 1.66x slower than the Transformer control. The next ablation should therefore reduce CNN depth/channels or spatial-token count while watching value loss, policy CE, Arena, and inference throughput jointly.

The evidence now points to the quality and spatial structure of the local representation as the critical variable. Compressing the entire board into one CNN vector was too aggressive, while preserving all 240 cell tokens was computationally wasteful. Computing rich local features first and then pooling to regional tokens is the first tested hybrid that preserves both sides of the earlier CNN-versus-Transformer trade-off.

## One-generation self-play adaptation experiment

The fixed-teacher ablations suggested that the hybrid policy was already competitive or better in move ranking, while its held-out value metrics did not translate cleanly to deeper search. To test the AlphaZero/KataGo-style hypothesis that policy/value/search need to co-adapt on the candidate's own distribution, one self-play generation was run from the seed-42 `augmented_pooled_hybrid` (128 CNN channels, 6 residual blocks, 60 pooled spatial tokens while retaining the raw Transformer token sequence).

### Self-play data

`gpu_match.load_model` was made architecture-agnostic for experimental checkpoints so the hybrid can drive the existing C++ GPU self-play protocol without changing the rules/search/dataset authority.

Generation used the RX 9070 XT (`cuda:1`), fp16 inference, Gumbel search, root exploration noise, model-version 5, and parallel C++ workers. Two search budgets approximate KataGo's playout-cap idea without changing the dataset schema:

- shallow/value-rich: 16 games, 8 sims, 2 determinizations -> 1,987 samples
- deep/policy-rich: 8 games, 32 sims, 2 determinizations -> 1,022 samples

The 3,009 new unique samples were mixed with the 47,693 Gen-4 samples. Because the current rectangular dataset has no per-sample policy-loss mask, the approximation weights the shallow shards x4 and the deep shards x8 during continuation. Effective concatenated sample count: 63,817. This is not a literal implementation of KataGo's policy-target masking; it is a first-generation test of candidate-distribution adaptation.

### Gen-5 continuation

The seed-42 augmented hybrid was resumed from step 400 with its AdamW state and trained for 200 additional steps (to global step 600), effective batch 256, microbatch 96.

On the weighted mixed validation distribution:

- start at step 400: policy 3.287463, total 3.853479
- step 600: policy 3.288556, total 3.899862

Thus conventional held-out total loss became worse, and the best validation-policy checkpoint remained the step-400 parent. Under a fixed supervised-selection rule this update would be rejected.

### Parent-vs-child causal check

This validation regression did **not** predict Arena strength.

Using 12 widely separated Arena base seeds, one paired block per seed (48 games total):

- policy-only Gen-5 vs Gen-4 parent: **25-23 = 52.1%**, Wilson 95% CI ~38.3%-65.5%
- 32-sim Gen-5 vs Gen-4 parent: **35-13 = 72.9%**, Wilson 95% CI ~59.0%-83.4%

This is the cleanest evidence in the experiment for policy/value/search co-adaptation. The raw policy did not materially improve against its parent, yet once both models were embedded in search the self-play-adapted child was much stronger.

### Transformer-control Arena

The repository does not currently contain `models/champion.pt`; therefore these results are explicitly against the fixed hashed-seed-42 Transformer ablation control used throughout this report, not an automatically promoted production champion.

Against that Transformer control:

- policy-only, 12 broad seeds / 48 games: **33-15 = 68.75%**, Wilson 95% CI ~54.7%-80.1%
- 16 sims, 12 broad seeds / 48 games: **32-16 = 66.67%**, Wilson 95% CI ~52.5%-78.3%
- 32 sims, first 12 broad seeds / 48 games: **30-18 = 62.5%**
- 32 sims, 12 additional broad seeds / 48 games: **35-13 = 72.9%**
- **32-sim aggregate: 65-31 over 96 games = 67.7%, Wilson 95% CI ~57.8%-76.2%**

For comparison, the step-400 augmented parent scored 26-22 (54.2%) against the Transformer on the original 12 broad 32-sim seeds. The one-generation child therefore shows a large improvement on the same style of Arena evaluation despite worse mixed validation total loss.

### Interpretation

The first-generation result strongly supports the hypothesis that fixed Transformer-generated supervision was masking the hybrid's real potential. The architecture's local CNN features improve action ranking, but a network trained only on the Transformer's search distribution is not automatically calibrated for the search tree induced by its own policy. Generating data with the hybrid itself and continuing training changes that interaction substantially.

The result should still be treated as an experimental generation rather than a production promotion. The shallow/deep mixing is an approximation to KataGo's playout-cap randomization and each generation adds only ~3k unique samples. However, the self-play-adaptation direction has now also been replicated from an independent training initialization, as described below.

### Independent training-seed replication (seed 1337)

A fresh augmented 128x6 parent was trained with seed 1337 on the same 47,693-sample hashed split. At step 400 it reached policy 3.308337 / total 3.885073. It then generated an independent Gen-5 corpus using non-overlapping game seeds:

- shallow 8-sim: 16 games / 1,837 samples (base seed 730000)
- deep 32-sim: 8 games / 1,064 samples (base seed 740000)

The same shallow x4 / deep x8 continuation recipe was applied for 200 additional steps. The seed-1337 Gen-5 child reached policy 3.197355 / total 3.865750 on its weighted mixed validation set.

At 32 simulations on the original 12 broad Arena seed blocks, the child beat its own seed-1337 parent **31-17 = 64.6%** over 48 games. Against the independently trained seed-1337 Transformer control, 24 broad seeds / 96 games gave **53-43 = 55.2%**. This is weaker than the seed-42 child but points in the same direction. Combining each Gen-5 child only with its corresponding same-training-seed Transformer control gives **118-74 over 192 games = 61.5%**, Wilson 95% CI approximately **54.4%-68.1%**.

The robust result across training seeds is therefore not that every Gen-5 child scores ~68% against Transformer, but that one generation of candidate-distribution self-play consistently improves the augmented hybrid relative to its own parent, while the size of the Transformer advantage varies with initialization.

### APM / APP diagnostic

A separate 32-sim self-play firepower diagnostic used the same 12 game seeds per model, steady garbage, one determinization, fp16, and the current root-noise self-play settings. These numbers are not Arena win rates; they measure simulator-time attack production under matched self-play conditions.

| training seed | model | APM | APP | PPS |
|---|---|---:|---:|---:|
| 42 | Gen-5 augmented hybrid | **86.94** | **0.105** | **13.78** |
| 42 | Transformer control | 38.14 | 0.049 | 12.94 |
| 1337 | Gen-5 augmented hybrid | **66.08** | **0.080** | **13.69** |
| 1337 | Transformer control | 31.86 | 0.040 | 13.23 |

Pooling the two 12-game blocks by total sent lines / pieces / simulator seconds yields approximately **78.21 APM, 0.095 APP, 13.74 PPS** for Gen-5 versus **35.02 APM, 0.045 APP, 13.08 PPS** for Transformer. The hybrid's advantage in this diagnostic is therefore predominantly attack efficiency rather than raw placement speed.

### Timing / deliberate non-cancelling audit

The tokenizer and action contract were already designed to make deliberate non-cancelling learnable: garbage tokens expose arrival/activation timing and cancellability, action feature 21 stores a delay bin, and `MoveGenerator::expand_delay_bins` implements timed action variants. However, an end-to-end audit found that the live `SelfPlayWorker`, `Searcher`, and `Arena` paths called ordinary placement generation directly and never invoked delay expansion.

This was confirmed empirically on all 3,009 seed-42 Gen-5 source samples: including 719 positions with pending garbage, both the stored MCTS policy and the network argmax placed **100%** of their timing mass on `FASTEST`. Therefore the existing Gen-4/Gen-5 agents have **not** learned deliberate non-cancelling despite the representation having been present.

An experimental, default-off timing path has now been wired through Search/SelfPlay/Arena. To keep the action-space cost bounded, positions with a future garbage activation branch only into `FASTEST` versus `WAIT_FOR_EVENT`; already-active garbage does not create a pointless wait branch. Existing behaviour remains unchanged unless the experiment flag is enabled. A dedicated search-wiring regression test was added, and the isolated Linux build passes **283 tests / 1,045,727 assertions / 0 failures**.

GPU training with the timing branch still requires rebuilding the Windows C++ child used by the ROCm Python environment. A separate Linux engine builds and tests correctly; the existing production Windows engine has deliberately been left untouched.

## Second self-play generation (Gen-6, seed 42)

To test whether the Gen-5 jump was a one-generation accident, the seed-42 Gen-5 child generated another independent replay generation:

- shallow 8-sim: 16 games / 2,049 samples (base seed 750000)
- deep 32-sim: 8 games / 1,510 samples (base seed 760000)

Step 600 was resumed to step 800 while retaining Gen-4 plus both Gen-5 and Gen-6 replay, with each recent generation using the same shallow x4 / deep x8 approximation. The weighted replay set contained 84,093 samples. Validation policy improved from 3.163491 to 3.147630, while total loss again worsened from 3.797595 to 3.882474 because value moved in the opposite direction.

The Arena signal nevertheless continued:

- Gen-6 vs Gen-5 parent, 12 broad seeds / 48 games / 32 sims: **31-17 = 64.6%**
- Gen-6 vs seed-42 Transformer control, same 12 broad seeds / 48 games / 32 sims: **32-16 = 66.7%**

Thus the self-play adaptation effect repeated for a second consecutive generation rather than appearing only at the first distribution shift.

On the same 12-game 32-sim firepower diagnostic used above, Gen-6 produced **65.71 APM / 0.084 APP / 12.99 PPS**. This is lower than Gen-5 seed42 (86.94 / 0.105 / 13.78) but still well above the seed-42 Transformer (38.14 / 0.049 / 12.94), while Gen-6 simultaneously beats Gen-5 in Arena. Raw attack rate is therefore not the sole objective being improved by the closed self-play/search loop.

## Added experiment tooling

- `trainer/ablation_models.py` — CNN baseline, hybrid, checkpoint loader
- `trainer/run_cnn_ablation.py` — fixed-data paired training ablation
- `trainer/run_ablation_arena.py` — Arena runner supporting all ablation checkpoint types and policy-only diagnostic overrides
- `trainer/analyze_policy_ranking.py` — teacher-ranking, top-k, entropy-stratified policy diagnostics
- `trainer/run_composite_arena.py` — inference-time policy/value factorial mixing for causal diagnostics
- `trainer/benchmark_ablation_inference.py` — forward-only latency/throughput comparison for ablation checkpoints
- `trainer/run_multiseed_ablation_arena.py` — wide-seed Arena aggregation, including policy-only diagnostics
- `trainer/gpu_match.py` loader — architecture-agnostic experimental checkpoint loading so hybrid checkpoints can drive the existing GPU self-play protocol

The existing `trainer/tetraformer.py`, production checkpoints, and champion are not replaced by this experiment.
