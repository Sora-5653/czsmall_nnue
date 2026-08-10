# CNN architecture ablation — 2026-08-08

> This file is experimental evidence, not the project architecture policy.
> [ADR 0013](adr/0013-architecture-ablation-and-local-geometry.md) records the
> accepted interpretation and the constraints on follow-up hybrids.

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

## Current interpretation

The evidence now supports a narrower conclusion than the first Arena suggested:

1. **Transformer is consistently better at reproducing the Transformer-generated search policy.** This survives two training seeds and the corrected hashed split.
2. **CNN is substantially better at WDL/value learning on the same samples.** The difference is much larger than the policy-CE gap.
3. **CNN's strongest and most reproducible advantage appears under search, not policy-only play.** On the hashed split, 16-simulation Arena is about 66% across 120 games while policy-only is statistically compatible with parity.
4. **Naively bolting an independent CNN value branch onto Transformer policy does not recover the gain.** The shared CNN representation or policy-value consistency is likely important.
5. Shared-local-feature hybrids were tested next and did **not** solve the problem. `fusion_hybrid` appends a CNN board token to the Transformer and also feeds that same board embedding directly to value/aux; `dual_policy_hybrid` additionally gives the CNN branch a direct 50/50 contribution to final policy logits. Both still show severe late value overfitting on the hashed split (400-step total losses 4.7677 and 4.6082 respectively). Thus simple policy regularisation of a compact CNN branch is insufficient.

The remaining plausible explanations are narrower: the full CNN baseline's much larger local encoder (192 channels, 10 residual blocks) may learn qualitatively better board features than the compact 64-channel/4-block hybrid branches; alternatively, the advantage may require policy and value to be produced from exactly the same CNN state rather than merely sharing some features. A future hybrid should test one of these hypotheses explicitly rather than adding more weakly-coupled heads.

## Added experiment tooling

- `trainer/ablation_models.py` — CNN baseline, hybrid, checkpoint loader
- `trainer/run_cnn_ablation.py` — fixed-data paired training ablation
- `trainer/run_ablation_arena.py` — Arena runner supporting all ablation checkpoint types and policy-only diagnostic overrides
- `trainer/analyze_policy_ranking.py` — teacher-ranking, top-k, entropy-stratified policy diagnostics
- `trainer/run_composite_arena.py` — inference-time policy/value factorial mixing for causal diagnostics

The existing `trainer/tetraformer.py`, production checkpoints, and champion are not replaced by this experiment.
