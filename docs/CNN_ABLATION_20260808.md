# CNNアーキテクチャ比較実験 — 2026-08-08

> この文書は**実験結果の記録**であり、project architecture policyそのものではありません。実験から採用した設計判断は [ADR 0013](adr/0013-architecture-ablation-and-local-geometry.md) を参照してください。

## 問い

同じdata、split、optimizer、minibatch、training steps、seedを固定したとき、**明示的な局所2D board inductive biasは現在のTetraFormer policyを改善するか**を検証しました。

既存Transformerは変更せずcontrolとして残します。CNNとCNN+Transformer hybridは追加比較modelであり、この実験だけでproduction modelやChampionを置き換えません。

## 固定dataset

既存の47,693-sample Gen-4 corpusを使用しました。

- `data/production/gen4_bootstrap_20260807.part00..07.tetradat`
- `data/production/gen4_search32_[a-f]_20260807.part00..07.tetradat`

連結したrectangular datasetは最大102 state tokens、106 legal actionsです。

### 最初のsplit

初期ablationでは既存のgame-seed split functionを1回適用し、全architectureで共有しました。

- train: 37,966 samples / 512 game seeds
- held-out: 9,727 samples / 128 game seeds

position単位のrandom splitは使っていません。

しかし後のauditで、production splitterがnumeric game seedをsortしてから80/20 cutしていることが分かりました。generation shardは連続seed rangeを使っていたため、held-out setが完全に `gen4_search32_e` と `gen4_search32_f` だけで構成され、bootstrapとsearch32 a-dからvalidation sampleが1つも入っていませんでした。

### Hashed game split

そこで、game seedをstable hashしてから80/20 assignmentする、leakage-safeなsplitを追加しました。

- hashed train: 37,231 samples / 499 game seeds
- hashed held-out: 10,462 samples / 141 game seeds

hashed validationには全sourceが入ります。

- bootstrap: 4,383 samples
- search32 a-f: 各694–1,225 samples

このsplitを、source rangeに依存しにくい代表条件として扱います。

## 固定training条件

- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size: 256
- steps: 400
- seed: 42
- sampled minibatch schedule: architecture間で完全共有
- loss weight: policy `1.0`, value `1.0`, aux `0.1`
- gradient clipping: 1.0
- GPU: RX 9070 XT（Windows ROCm PyTorch環境では `cuda:1`）

parameter countも意図的に近づけました。

- TetraFormer-S control: 7,175,592
- CNN baseline: 7,163,880
- CNN+Transformer hybrid: 7,507,560

## Architecture詳細

### Transformer control

既存の8-layer、width-256 TetraFormer-Sを変更せず使用します。

### CNN baseline

tokenizerはrow tokenのfeature 8..17にexact 10-wide occupancy bitmapを保持しています。

現行two-player token orderでは:

- self board: 最初の24 row tokens
- opponent board: 最後の36 real tokens中の先頭24 rows
  - 24 rows + 10 columns + board summary + opponent counters

から両盤面を復元できます。

CNN inputは5枚の24x10 planeです。

1. self occupancy
2. opponent occupancy
3. self - opponent occupancy
4. normalized y coordinate
5. normalized x coordinate

residual 3x3 CNNでboard tensorをencodeします。non-board stateはtoken-wise MLP + masked mean poolingで保持し、fused state vectorを条件として各legal actionをscoreします。

#### Windows ROCm上のConv2d workaround

このWindows ROCm buildでは通常の `Conv2d` backwardが、gfx1201で `invalid device function` になるMIOpen CK kernelを選ぶことがありました。

そのため同じspatially shared 3x3 convolutionを `unfold + GEMM` で実装しました。convolution operation自体は維持しつつ、失敗するMIOpen pathだけを避けています。

### CNN+Transformer hybrid

小型board CNNが1つのlearned board tokenを生成し、通常TetraFormer state sequenceへ追加して8 Transformer blocksへ通します。

既存のvariable-length cross-attention policy headはそのまま残します。

## 初期held-out結果

primary metricは400 steps後のheld-out policy cross-entropyです。

| Architecture | Held-out policy loss | Held-out total loss | 注記 |
|---|---:|---:|---|
| Transformer | **3.244058** | 3.941302 | step 400でbest policy |
| Hybrid | 3.260342 | 4.204065 | best policy 3.255634 at step 300 |
| CNN | 3.267351 | **3.896739** | step 400でbest policy |

policy cross-entropyだけならoriginal Transformerがbestです。

Transformerとの差:

- CNN: +0.023293
- Hybrid: +0.016284

一方CNNはfinal total lossがTransformerより低く、local board biasがvalue/auxiliary predictionの一部には寄与している可能性がありました。

## 初期Arena

400-step final checkpointを使用し、search budget、seed 42、fp16 inference、300-piece capを固定しました。ここではGumbelを使っていません。

### Search Arena — 16 simulations each

| Candidate vs Transformer | Record | Win rate | 95% CI |
|---|---:|---:|---|
| CNN | **34-6** | **85.0%** | 70.9%–92.9% |
| Hybrid | **28-12** | **70.0%** | 54.6%–81.9% |

engine report上、10 paired Arena unitsは40 gamesになります。

CNNのconfidence intervalは55% promotion thresholdを完全に上回りました。Hybridはpoint estimateこそ強いもののlower boundが55%をわずかに下回るため、この時点ではより大きなArenaが必要でした。

### Policy-only Arena — 0 simulations each

| Candidate vs Transformer | Record | Win rate | 95% CI |
|---|---:|---:|---|
| CNN | **27-13** | **67.5%** | 52.0%–79.9% |
| Hybrid | 18-22 | 45.0% | 30.7%–60.2% |

このinitial policy-only resultは、後続実験で「CNN raw policyが本質的にstronger」と断定するには不安定だと分かりました。以下のlarger / corrected-split experimentでinterpretationを更新します。

## 追試 — ranking diagnostic、split audit、independent seed

### Teacher-ranking diagnostic

original 9,727-position held-out set上でも、direct teacher-ranking metricではTransformerがCNNよりわずかに良い結果でした。

| Metric | Transformer | CNN | Hybrid |
|---|---:|---:|---:|
| teacher argmax top-1 | **16.58%** | 16.07% | 16.91% |
| teacher-best top-3 recall | 38.46% | 38.20% | **39.64%** |
| mean reciprocal rank | 0.3355 | 0.3321 | **0.3414** |
| teacher mass at model argmax | **0.08910** | 0.08901 | 0.08865 |
| mean teacher-best rank | **8.49** | 8.58 | 8.52 |

つまりCNNは単にtail calibrationを犠牲にしてteacher best moveのrankingを改善していたわけではありません。

初期policy-only Arena winはteacher imitation ranking qualityでは説明できません。

### Independent training seed

ordered-split experimentをtraining seed 1337で再実行してもpolicy CEの順序は維持されました。

- Transformer: 3.240603
- CNN: 3.270994

一方、seed-1337 CNNは対応するTransformerにArenaで再び勝ちました。

- policy-only: 52-28 / 80 games
  - 65.0%
  - 95% CI 54.1%–74.5%
- 16 simulations: 32-8 / 40 games
  - 80.0%
  - 95% CI 65.2%–89.5%

したがってordered-splitの現象は1つのmodel initialization seedだけには依存していません。

### Hashed game split

representative hashed split + seed 42では次の結果でした。

| Architecture | policy loss | value loss | value accuracy | value scalar MSE | total loss |
|---|---:|---:|---:|---:|---:|
| Transformer | **3.302256** | 0.693175 | 51.95% | 0.99975 | 3.997783 |
| CNN | 3.334443 | **0.582125** | **67.25%** | **0.80018** | **3.918749** |

ここで重要な構造分離がはっきりしました。

- Transformer: search policyのimitationが良い。
- CNN: 同じsampleからWDL/valueをかなり良く学ぶ。

hashed-split Arenaでは、initial policy-only resultとは違う姿になりました。

- policy-only, seed 777: CNN 37-43 / 80 games
  - 46.3%
  - 95% CI 35.7%–57.1%
- 16 simulations, seed 777: CNN 27-13 / 40 games
  - 67.5%
- 16 simulations, seed 5150: CNN 52-28 / 80 games
  - 65.0%

2つのindependent 16-simulation hashed-split trialをpoolすると:

- CNN 79-41 / 120 games
- 65.8%
- Wilson 95% CI ≈ 57.0%–73.7%

raw-policy advantageよりsearch advantageのほうがかなり安定しています。

## Teacher provenance

`gen4_search32_[a-f]` は `models/gpu_gen_20260805_gen3.pt` をteacherとして、32 simulations・root noiseなしで生成されました。

そのcheckpoint自身も8-layer、width-256 TetraFormerです。

つまりheld-out policy cross-entropyは**architecture-neutral game strength**ではなく、Transformer-driven search teacherをどれだけ模倣できるかを主に測っています。

このため、小さなCE差よりArenaへ大きなweightを置く理由があります。

## Split-head hybrid test

`split_hybrid` diagnosticでは、TetraFormer policy pathを構造的にそのまま残し、value/auxだけをindependent compact CNN branchへ移しました。

100 steps時点ではhashed held-out value metricがTransformerより改善しました。

- value loss: 0.6158
- accuracy: 64.6%
- scalar MSE: 0.8577
- policy CE: 3.356448 vs Transformer 3.356361

しかしisolated value branchはその後strongly overfitしました。

400 stepsでvalue lossは1.3318まで悪化し、train loss自体は改善を続けていました。

さらに100-step 16-simulation Arenaではmatched 100-step Transformerへ17-23で負けました。

この結果から、**static held-out value metricが良いだけでは不十分**と分かります。

searchはcounterfactual / off-policy positionをprobeします。full CNN baselineのshared policy/value representationがvalue functionをregularizeしており、independent branchではその性質を失った可能性があります。

## Policy / value factorial diagnostic

hashed seed-42 checkpointを使い、retrainせずinference時にpolicy/value outputをmixしました。

共通20-game seed blockで:

- Transformer policy + CNN value vs Transformer/Transformer: 10-10
- CNN policy + Transformer value vs Transformer/Transformer: 11-9
- CNN policy + CNN value vs Transformer/Transformer: 11-9

20 gamesではstrength claimには不足しますが、どちらか1 component swapだけで明確なgainが出るわけではありませんでした。

larger full-model Arenaと合わせると、1 headだけでなく**policy-value compatibility / shared representation**が重要である可能性があります。

## Shared-local-feature hybridの追試

追加で、local featureを共有するhybridも試しました。

- `fusion_hybrid`
  - CNN board tokenをTransformerへappend
  - 同じboard embeddingをvalue/auxへ直接入力
- `dual_policy_hybrid`
  - 上記に加えCNN branchがfinal policy logitsへ50/50で直接寄与

しかし両方ともhashed splitでlate value overfitを示しました。

400-step total loss:

- `fusion_hybrid`: 4.7677
- `dual_policy_hybrid`: 4.6082

したがって、compact CNN branchへ単純なpolicy regularizationを加えるだけではfull CNNの性質を再現できませんでした。

## 現在のinterpretation

証拠が支持するconclusionを、初期Arenaより狭くします。

1. **TransformerはTransformer-generated search policyを一貫して良く再現する。**
   - 2 training seeds
   - corrected hashed split
   でも維持。

2. **CNNは同じsampleからWDL/valueを大幅に良く学ぶ。**
   policy CE差よりvalue差のほうが大きい。

3. **CNNの最も再現性の高いadvantageはsearch下に現れる。**
   hashed splitの16-simulation Arenaでは120 games pooledで約66%。policy-onlyではparityと矛盾しない。

4. **independent CNN value branchをTransformer policyへ付けるだけではgainを回収できない。**
   shared CNN representationまたはpolicy-value consistencyが重要な可能性がある。

5. **small shared-local hybridでも解決しなかった。**
   full CNN baselineの192 channels / 10 residual blocksと、compact hybridの64 channels / 4 blocksではfeature qualityが違う可能性がある。またはpolicy/valueを同じCNN stateから直接生成すること自体が重要かもしれない。

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
したがって次のhybrid experimentでは、「もっとheadを足す」のではなく、次のような明示hypothesisを1つ選びます。

- full-capacity local encoderが必要なのか
- policy/valueがexact same CNN representationを共有する必要があるのか
- local CNN + global Transformerというrole separationが有効なのか

## 追加した実験tool

- `trainer/ablation_models.py` — CNN baseline, hybrid, checkpoint loader
- `trainer/run_cnn_ablation.py` — fixed-data paired training ablation
- `trainer/run_ablation_arena.py` — Arena runner supporting all ablation checkpoint types and policy-only diagnostic overrides
- `trainer/analyze_policy_ranking.py` — teacher-ranking, top-k, entropy-stratified policy diagnostics
- `trainer/run_composite_arena.py` — inference-time policy/value factorial mixing for causal diagnostics
- `trainer/benchmark_ablation_inference.py` — forward-only latency/throughput comparison for ablation checkpoints
- `trainer/run_multiseed_ablation_arena.py` — wide-seed Arena aggregation, including policy-only diagnostics
- `trainer/gpu_match.py` loader — architecture-agnostic experimental checkpoint loading so hybrid checkpoints can drive the existing GPU self-play protocol
- `trainer/ablation_models.py`
  - CNN baseline
  - hybrid variants
  - checkpoint loader
- `trainer/run_cnn_ablation.py`
  - fixed-data paired training ablation
- `trainer/run_ablation_arena.py`
  - ablation checkpoint対応Arena runner
  - policy-only diagnostic override
- `trainer/analyze_policy_ranking.py`
  - teacher ranking
  - top-k
  - entropy-stratified policy diagnostic
- `trainer/run_composite_arena.py`
  - inference-time policy/value factorial mixing

既存 `trainer/tetraformer.py`、production checkpoint、Championはこの実験によって置換していません。
