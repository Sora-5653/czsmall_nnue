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

したがって次のhybrid experimentでは、「もっとheadを足す」のではなく、次のような明示hypothesisを1つ選びます。

- full-capacity local encoderが必要なのか
- policy/valueがexact same CNN representationを共有する必要があるのか
- local CNN + global Transformerというrole separationが有効なのか

## 追加した実験tool

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
