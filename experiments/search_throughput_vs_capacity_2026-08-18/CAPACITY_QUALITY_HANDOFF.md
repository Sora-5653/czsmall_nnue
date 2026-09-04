# Evaluator Capacity × Quality × Search Throughput 再設計実験手順書

作成日: 2026-08-18
対象ディレクトリ: `experiments/search_throughput_vs_capacity_2026-08-18/`

---

## 0. このタスクの目的

前段の実験では、軽量評価器 B (`transformer_xs.final.pt`) は評価器単体で高速であり、E3・10 ms の200局Pilotでは A より約4.66倍多い `nodes/decision` を処理したにもかかわらず、Bの勝率は3.0%だった。

この結果から、現在のBは「少し品質が低い代わりに大きく探索量を増やせる評価器」という比較用候補ではなく、**評価品質差が大きすぎて、モデル容量と探索量の交換関係を測るA/B比較を歪めている可能性が高い**。

ただし、この時点で

> XSはモデル容量が小さすぎるから弱い

と結論してはいけない。

リポジトリ調査により、現在のBはサイズだけでなく、学習条件も現Champion Aと大きく異なる可能性があることが分かっている。

このタスクの目的は、次の4つを分離して検証することである。

1. **モデル容量不足**
2. **学習量不足 / 学習データ世代差**
3. **policy品質不足**
4. **value品質不足**

そのうえで、最終的に「同一wall-clockで探索量を増やす価値」を比較できる、品質差が過大でない中間サイズ評価器を選抜する。

本タスクは、既存XSをChampionへ昇格させることを目的としない。

---

# 1. 現在までに確定している事実

## 1.1 Champion側 A

現在の比較基準:

```text
models/gen14_rank100_100_20260814.best.pt
```

前段実験でのパラメータ数:

```text
7,176,624
```

このcheckpointを本タスク中に上書き・再学習・昇格・置換してはならない。

## 1.2 現在のB

```text
models/size_search_ablation_20260816/seed42/transformer_xs.final.pt
```

既存 `train_results.json` から確定している条件:

```text
parameters       = 132,744
width            = 64
layers           = 2
heads            = 4
ffn              = 192
training steps   = 400
batch            = 256
training samples = 37,231
validation       = 10,462
total samples    = 47,693
split            = hashed-game-80/20
source           = gen4_bootstrap + gen4_search32_* datasets
```

最終validation値:

```text
policy loss       = 3.3088097915
value loss        = 0.6941435917
value accuracy    = 0.5194991397
value scalar MSE  = 0.9995663937
```

したがって、現在のA/B比較は単なる

```text
7.18M parameters vs 0.133M parameters
```

ではない可能性がある。

Bは古いGen4系データで400 stepだけ学習されたサイズablation checkpointである。
AはGen14系Champion checkpointである。

**容量差と学習履歴差が交絡している可能性を最優先で検証すること。**

## 1.3 既存の中間サイズcheckpoint

同じサイズablationに次が存在する。

### M

```text
models/size_search_ablation_20260816/seed42/transformer_m.final.pt
```

```text
parameters = 946,920
width      = 128
layers     = 4
heads      = 4
ffn        = 384
steps      = 400
```

validation:

```text
policy loss      = 3.3121173320
value loss       = 0.6935713032
value accuracy   = 0.5194991397
```

### S

```text
models/size_search_ablation_20260816/seed42/transformer_s.final.pt
```

```text
parameters = 7,175,592
width      = 256
layers     = 8
heads      = 8
ffn        = 768
steps      = 400
```

validation:

```text
policy loss      = 3.3022562281
value loss       = 0.6931745939
value accuracy   = 0.5194991397
```

このSはパラメータ数がAにほぼ等しいにもかかわらず、同じ400-step size-ablation学習条件である。

**このSは極めて重要なcontrolである。**

もしSもAに大敗するなら、「XSが小さすぎる」だけでは説明できず、学習データ・学習量・checkpoint世代差が主因である可能性が高い。

---

# 2. 現在のXS結果をどう扱うか

前段のE3・10 ms・200局Pilot:

```text
B wins/losses/draws = 6 / 194 / 0
B win rate          = 3.0%
95% CI              = 1.4–6.4%
B nodes/decision    = 16.63
A nodes/decision    = 3.57
B/A nodes           ≈ 4.66x
```

この結果から言えること:

- XSは現在のAより大幅に弱い。
- XSは探索量不足で負けているわけではない。
- 約4.66倍の探索量でも品質差を補えていない。

この結果からまだ言えないこと:

- 0.133Mという容量そのものが弱さの主因。
- 小型モデル一般が探索評価器として不適切。
- throughput-first仮説が否定された。
- Transformerを小さくすると必ず弱くなる。

現在のXSは今後、**Pareto curve上の極端な小容量点 / failure boundary** として保存する。

Main 2,000局へは進めない。

---

# 3. 新しい中心仮説

## H1: 容量不足仮説

同じ学習条件を与えても、XSはpolicy/value表現力が不足し、探索量増加では補えない。

予測:

- 同一データ・同一学習量でXS < M < S/Aに強さが単調に近く改善する。
- S-sizeを十分学習すればAに近づく。
- XSを長く学習しても大差が残る。

## H2: 学習不足 / 世代差仮説

現在のXSの弱さの主要因は、容量ではなく400-step・Gen4系学習条件にある。

予測:

- 既存S-size-ablationもAに大敗する。
- XS/Mを現行データ・現行教師で再学習すると大幅に改善する。
- 同サイズでも古い400-step checkpointと再学習checkpointの差が大きい。

## H3: Policyボトルネック仮説

XSはvalueよりpolicy priorが壊れており、探索の入口で有望手を十分に残せない。

予測:

```text
A-policy + B-value >> B-policy + A-value
```

## H4: Valueボトルネック仮説

XSは候補手生成はそこそこだが、leaf ranking / outcome estimationが弱い。

予測:

```text
B-policy + A-value >> A-policy + B-value
```

## H5: 両head / shared representation不足

policy/valueの片方だけ差し替えても十分回復せず、shared trunkの表現不足または両headの品質低下が主因。

---

# 4. 実験の原則

この段階ではwall-clock Mainを急がない。

実験順序は必ず次とする。

```text
品質測定
    ↓
学習条件交絡の除去
    ↓
policy/value分離
    ↓
equal-node品質ゲート
    ↓
throughput測定
    ↓
equal-wall-clock Pilot
    ↓
Main
```

順序を逆転しない。

---

# 5. Phase 0: 既存checkpointだけで交絡を検出する

新規学習を始める前に、既存checkpointのみで以下を測る。

対象:

```text
A      = gen14_rank100_100_20260814.best.pt
S400   = size_search_ablation/.../transformer_s.final.pt
M400   = size_search_ablation/.../transformer_m.final.pt
XS400  = size_search_ablation/.../transformer_xs.final.pt
```

可能ならseed1337版もsecondaryとして測るが、primaryはseed42固定。

## 5.1 No-search policy control

探索なしで、A/S400/M400/XS400を固定opponentまたはpaired direct comparisonで評価する。

最低200局 / pair。

優先比較:

```text
A vs S400
A vs M400
A vs XS400
S400 vs M400
M400 vs XS400
```

ここで最重要なのは **A vs S400**。

S400はAとほぼ同パラメータ数なので、S400がAに大敗する場合、現在のA/XS差を「容量差」と解釈してはならない。

## 5.2 Equal-node control

同じ探索アルゴリズム・同一node budgetで比較する。

node budgets:

```text
16
32
64
128
```

最低でも64 nodesで各pair 200局。

Primary:

```text
A vs S400 @64
A vs M400 @64
A vs XS400 @64
```

目的はthroughput差を消し、pure evaluator/search guidance qualityを測ること。

## 5.3 判定

### S400がAに大敗

→ 学習条件差を主交絡として扱う。

### S400≈AだがXS400のみ大敗

→ 容量仮説が強まる。

### S400≈A、M400もそこそこ、XSだけ崩壊

→ 0.13M近辺にcapacity cliffがある可能性。

---

# 6. Phase 1: Held-out evaluator qualityを直接測る

Arenaだけでは原因が粗いので、同一held-out state set上で各checkpointの評価品質を比較する。

## 6.1 データセット

可能な限り、現在のAの学習世代に近いreplay / reanalyse / self-playデータから、**trainに使わない固定held-out game set** を作る。

要件:

- game/replay単位でsplit。
- trainとvalidationを局面単位で混ぜない。
- A/Bどちらか専用のデータにしない。
- manifestにsource hashを保存。
- 少なくとも10,000局面。
- E1/E2/E3相当の局面種別を可能ならタグ付け。
- garbage pending / high stack / cancellation / clean stack等のsubgroupを記録可能なら記録。

既存の信頼できるheld-out splitがあれば再利用する。

## 6.2 測るもの

### Policy

単なるcross-entropyだけでなく:

```text
policy cross entropy
teacher/search top-1 accuracy
top-3 recall
top-5 recall
top-10 recall
rank of teacher/search best action
NDCG or rank correlation if available
probability mass on searched-good actions
```

### Value

```text
WDL cross entropy
value accuracy
scalar MSE
Brier score
calibration curve / ECE
Spearman rank correlation
pairwise ordering accuracy
```

valueは絶対値だけでなく、**同一rootの複数childを正しく順位付けできるか**を重視する。

探索用途ではchild rankingが重要だからである。

## 6.3 Teacher target

可能ならAそのもののraw出力だけを正解とせず、次の優先順位とする。

1. deeper search / reanalyse target
2. game outcome / WDL target
3. strong search policy target
4. A raw output（distillation用secondary）

Aをteacherにする場合は、`teacher_agreement`として別指標にし、ground truthと混同しない。

---

# 7. Phase 2: Policy / Value swap ablation

これは本タスクの重要部分。

現在のsearch codeが同じforwardからpolicy/valueを受け取るため、診断モードとしてhead sourceを分離できる最小フックを追加する。

少なくともAと候補Bについて:

```text
AA = A-policy + A-value
AB = A-policy + B-value
BA = B-policy + A-value
BB = B-policy + B-value
```

ここでtrunkまで完全分離する必要がある場合は、2モデルforwardを行って必要headだけ採用する診断実装でよい。

**速度比較には使わない。品質原因の診断専用。**

## 7.1 Primary B

まずXS400で行う。

最低:

```text
64-node equal-node
E3
200局 / comparison
paired seeds
```

必要ならM400でも同じ実験を行う。

## 7.2 解釈

### AB ≈ AA, BA ≈ BB

B-valueは比較的使えるがB-policyが問題。

### BA ≈ AA, AB ≈ BB

B-policyは比較的使えるがB-valueが問題。

### AB, BAとも中間

両方に品質差。

### AB, BAともBBに近い

shared representation / head interaction / calibration / search integrationの問題を疑う。

---

# 8. Phase 3: 公平な再学習系列を作る

ここで初めて新規学習を行う。

## 8.1 最重要ルール

**サイズだけを変え、それ以外の学習条件を揃える。**

最低候補:

```text
XS = 0.133M class
M  = 0.947M class
S  = 7.176M class
```

ただし最終的にAと同等のS architectureが現Aと細部で異なる場合、その差をmanifestに記録する。

可能ならMとSの間に1つ中間サイズを追加してもよい。

推奨例:

```text
~2M–3M parameters
```

ただし、新しいarchitecture familyを発明しない。width/layers/ffnを既存Transformer family内で調整する。

## 8.2 学習データ

現在Champion Aの能力に近づけることが目的なので、Gen4だけではなく、現在使用可能な最新の信頼できるproduction replay/self-play/reanalyse mixを使う。

重要:

- 全sizeで完全に同じdataset manifest。
- 同じsample order policy。
- 同じtrain/validation split。
- 同じloss semantics。
- 同じlabel source。
- 同じoptimizer family。
- 同じaux loss。
- 同じtiming loss設定。

## 8.3 学習量の公平性

単純に全sizeを同じ400 stepsで止めない。

400 stepsではcapacity比較ではなくundertraining比較になる可能性がある。

primaryとして **validation convergenceまで学習** する。

停止条件例:

```text
validation policy/value compositeがN evaluation連続で改善しない
```

または既存trainerのbest-save / early stopping相当を使用する。

同時に、計算効率比較用として以下も記録する。

```text
same updates
same samples seen
same training wall-clock
converged
```

主たる「モデル品質比較」にはconverged checkpointを使う。

## 8.4 Seed

最低2 seed。

推奨:

```text
42
1337
```

計算量が許せば3 seed。

1 seedだけの偶然をcapacity cliffと解釈しない。

---

# 9. Phase 4: Distillation系列を別枝として測る

小型評価器を探索用に使う目的なら、同じlabelだけでゼロから学ぶより、Champion A / strong searchからdistillする方が実用上自然である。

したがって、supervised系列と混ぜずに、secondary branchとして次を作ってよい。

```text
XS-distilled
M-distilled
(optional intermediate)-distilled
```

teacher:

```text
strong search policy distribution
A value/WDL distribution
```

可能ならhard top-1だけでなくsoft distributionを保存する。

Distillation branchは必ず通常学習branchと分けて報告する。

目的は:

> 小型モデルの容量限界

と

> 小型モデルへ知識を詰める学習手法の限界

を分離することである。

---

# 10. Phase 5: Quality Gate

Wall-clock比較へ進める前に、候補ごとに品質ゲートを設ける。

これは絶対的なChampion昇格基準ではなく、**throughputで逆転可能な範囲まで品質差が小さいか**を見るための選抜である。

## 10.1 Gate A: Held-out quality

候補がAに対して以下を満たすことを目安とする。

- policy top-k recallが壊滅していない。
- value pairwise orderingが大幅に崩れていない。
- subgroupで特定局面だけ崩壊していない。

数値閾値は測定前に実装者が勝手に後付けしない。

まずA/S/M/XSの曲線を出してから、相対的なcliffを判断する。

## 10.2 Gate B: Equal-node Arena

Primary:

```text
64 nodes
E3
>= 400 games / candidate vs A
paired seeds
```

解釈の目安:

```text
45–50%   非常に有望
35–45%   throughputで逆転しうる候補
20–35%   厳しい。速度差が大きい場合のみsecondary
<20%     primary wall-clock候補から除外
```

これは事前の硬い統計的法則ではなく、wall-clock試験の計算資源をどこへ配分するかの実務ゲート。

**3%級のモデルをMainへ流さない。**

## 10.3 Gate C: No-search sanity

raw policyが完全に壊れていないこと。

no-searchでAに極端に大敗する候補は、search priorとして危険なのでpolicy swap結果も確認する。

---

# 11. Phase 6: Throughput Curve

Quality Gateを通った候補だけ evaluator benchmark / real-game forward benchmarkを行う。

batch:

```text
1
2
4
8
16
32
64
128
256
```

ただしreal-gameで128/256が現れないことは前段で分かっているため、主評価は以下。

```text
batch 1
batch 4
batch 8
batch 16
```

記録:

```text
states/s
per-state latency
p50/p95
VRAM
real-game actual batch histogram
model_forward_us/state
service_us/state
```

**batch128/256の倍率を実ゲーム倍率として使わない。**

---

# 12. Phase 7: Equal-wall-clock Pilot

Quality Gateを通った候補だけ実施。

budgets:

```text
10 ms
40 ms
160 ms
```

E3をprimaryとする。

各budget:

```text
>= 200 games
paired seeds
max 300 pieces
stderr drain enabled
batch prewarm enabled
same stable runner
```

測定:

```text
win rate + 95% CI
nodes/decision
nodes/ms
APM
APP
PPS
survival
topout
actual batch histogram
forward time
service time
```

ここで初めて、

> evaluator quality lossをthroughput gainが上回るか

を判定する。

---

# 13. 最終的に欲しいグラフ

単一A/B表で終わらせない。

## 13.1 Parameter count vs quality

x:

```text
log(parameters)
```

y:

```text
equal-node win rate vs A
```

## 13.2 Parameter count vs throughput

x:

```text
log(parameters)
```

y:

```text
real-game forward states/s
or nodes/ms
```

## 13.3 Quality vs throughput Pareto frontier

x:

```text
real-game nodes/ms or evaluator latency
```

y:

```text
equal-node strength
```

## 13.4 Wall-clock strength surface

可能なら最終的に、モデルsize Nとsearch budget tに対して

```text
strength(N, t)
```

を可視化する。

最低でも:

```text
A, candidate1, candidate2 × 10/40/160 ms
```

を同一グラフへ載せる。

---

# 14. 重要な診断: 「容量cliff」と「training cliff」を分ける

以下の表を必ず最終READMEへ作る。

| model | params | training data | steps/samples | converged? | equal-node WR vs A | 10ms WR vs A | real forward ratio |
|---|---:|---|---:|---|---:|---:|---:|
| A | ... | ... | ... | ... | 50% | 50% | 1x |
| S400 | ... | Gen4 | 400 | no/unknown | ... | ... | ... |
| M400 | ... | Gen4 | 400 | no/unknown | ... | ... | ... |
| XS400 | 132744 | Gen4 | 400 | no/unknown | ... | 3% existing | ... |
| S-new | ... | current | converged | yes | ... | ... | ... |
| M-new | ... | current | converged | yes | ... | ... | ... |
| XS-new | ... | current | converged | yes | ... | ... | ... |

これにより、例えば:

### ケース1

```text
S400 << A
S-new ≈ A
M-new ≈ A-ish
XS-new << A
```

→ 旧比較は学習不足で汚染されていたが、XSにはcapacity cliffもある。

### ケース2

```text
S400 << A
S-new ≈ A
M-new ≈ A
XS-new ≈ A-ish
```

→ 旧XSの弱さは主に学習不足。小型化自体は有望。

### ケース3

```text
S400 ≈ A
M400 moderately lower
XS400 catastrophically lower
```

→ capacity cliffが強く支持される。

### ケース4

```text
all new small models still << A
```

→ 現行representationではcapacity依存が強い。またはAが最新reanalyse/self-play等から持つ情報を小型モデルへ移せていない。

---

# 15. Policy/value分離結果の最終表

最低限:

| policy source | value source | equal-node WR | policy top-k | value ordering | note |
|---|---|---:|---:|---:|---|
| A | A | baseline | ... | ... | control |
| A | XS | ... | A | XS | isolate value |
| XS | A | ... | XS | A | isolate policy |
| XS | XS | ... | XS | XS | current B |

M候補でも必要に応じて行う。

---

# 16. 実装上の禁止事項

## 禁止1: 旧XSの3%結果だけで「小型モデルはダメ」と結論する

学習条件が交絡している。

## 禁止2: 旧400-step Mをそのまま「1Mモデルの最終性能」と扱う

まずS400 vs Aでtraining-generation gapを検証する。

## 禁止3: 全sizeを400 stepsだけ学習して容量曲線を描く

undertrainingの影響がsizeごとに異なる可能性がある。

## 禁止4: wall-clock MainをQuality Gate前に実行する

大幅に弱いモデルへ計算を浪費しない。

## 禁止5: 小型モデルだけ追加の強いデータを与える

公平なcapacity比較branchではdataset/labelsを揃える。

Distillationは別branchとして明示する。

## 禁止6: policy/value swapを速度比較として使う

2-model forwardは原因診断専用。

## 禁止7: batch128/256 throughputを実ゲーム代表値として報告する

実ゲーム平均batchは前段Pilotで4.76、batch1 requestが55.92%だった。

## 禁止8: Champion Aを上書きする

本タスクは研究評価でありpromotion taskではない。

## 禁止9: 既存dirty worktreeをreset/clean/stashする

既存変更を保持する。

---

# 17. 保存先

本タスクの追加成果物は既存実験ディレクトリ内に分離する。

推奨:

```text
experiments/search_throughput_vs_capacity_2026-08-18/
└── capacity_quality/
    ├── README.md
    ├── configs/
    ├── manifests/
    ├── results/
    │   ├── heldout/
    │   ├── equal_node/
    │   ├── head_swap/
    │   ├── throughput/
    │   └── wallclock/
    ├── scripts/
    └── notes/
```

新規trained checkpoints自体は既存project conventionに従い `models/` 配下へ置き、manifestから参照する。

---

# 18. Manifest要件

各checkpointについて最低限:

```text
path
sha256
architecture
parameters
training dataset manifest hash
train samples
total optimizer steps
samples seen
seed
best step
validation metrics
teacher/distillation source if any
commit hash
```

旧S400/M400/XS400も同様にmanifest化する。

「古いcheckpointなので不明」は許容するが、不明項目はnull/unknownとして明記する。

---

# 19. 推奨作業順

厳守する。

1. git status / instructions確認。
2. 既存A/S400/M400/XS400のmetadata manifest化。
3. S400 vs A no-search/equal-node。
4. M400/XS400を同条件で測定。
5. training-generation gapの有無を判定。
6. 共通held-out set作成。
7. policy/value/ranking quality測定。
8. A/XS head swap。
9. 必要ならA/M head swap。
10. 最新共通dataset manifest確定。
11. XS/M/Sを公平条件で再学習。
12. validation convergence確認。
13. 2 seed以上で再現。
14. new modelsのheld-out quality測定。
15. equal-node 64で>=400局 quality gate。
16. 候補を1〜2個に絞る。
17. real-game throughput測定。
18. 10/40/160ms 200局Pilot。
19. Pareto frontier作成。
20. Mainを行う価値がある候補だけ提案する。

---

# 20. このタスクの停止条件

次の場合はMainへ進まず、一度報告する。

### Stop A

S400がAに大敗する。

→ 旧size-ablation系列はcapacity比較として汚染されている可能性が高い。再学習を優先。

### Stop B

S-newですらAに大敗する。

→ architecture/config/label/training pipeline差を調査。size比較を続けない。

### Stop C

全小型new modelがequal-nodeでAに20%未満。

→ throughput Mainへ進まず、distillation/architecture改善を検討。

### Stop D

M-newがequal-node 35%以上かつ実ゲームforwardが有意に高速。

→ M-newをwall-clock primary candidateとして採用。

### Stop E

XS-newがequal-node 35%以上まで回復。

→ 旧XSの3%は主にtraining gapだった可能性が高い。XS-newもwall-clock候補へ戻す。

---

# 21. 成功条件

本タスクは以下が揃って初めて完了。

- [ ] 旧S400 vs Aでtraining-generation confoundを測定した。
- [ ] 旧XSの弱さを容量だけのせいにしていない。
- [ ] 共通held-outでpolicy/value品質を比較した。
- [ ] policy/value swap ablationを行った。
- [ ] 少なくともXS/M/Sの公平再学習系列を作った、または作れない明確な理由を記録した。
- [ ] convergenceを確認した。
- [ ] 最低2 seedで容量傾向を確認した。
- [ ] equal-node quality gateを行った。
- [ ] throughputとqualityのPareto curveを作った。
- [ ] wall-clock比較へ進む候補を品質根拠付きで選んだ。
- [ ] 旧XS 3%をMainへそのまま持ち込んでいない。

---

# 22. 最終READMEで必ず答える質問

1. **旧S400は、ほぼ同サイズのAに対してどれほど弱かったか。**
2. **旧A/XS差のうち、training-generation gapで説明できる部分はどの程度か。**
3. **XSを現行データで収束まで学習すると、どこまで回復したか。**
4. **MはAに対してequal-nodeで何%勝てるか。**
5. **policyとvalueのどちらが小型化で先に壊れるか。**
6. **real-game batch 1–16で各sizeは何倍速いか。**
7. **quality lossとthroughput gainのPareto frontierはどこにあるか。**
8. **10/40/160 msのwall-clockで最も強いsizeはどれか。**
9. **0.133Mはcapacity cliffなのか、旧学習条件が悪かっただけなのか。**
10. **次にMain 2,000局を回す価値がある候補は何か。**

---

# 23. 最終的な研究上の問い

この実験で答えたいのは、

> 「7Mモデルと0.13Mモデルのどちらが強いか」

ではない。

本当に答えたいのは、

> **対戦テトリスにおいて、評価器容量をどこまで減らすと、得られる探索throughputの増加が評価品質低下を上回るのか。**

である。

したがって、最終成果物は二点比較ではなく、最低限

```text
model capacity
× evaluator quality
× real-game throughput
× equal-wall-clock strength
```

の関係が見える形にする。

XSが失敗点なら、それ自体を重要な境界条件として残す。
Mや中間サイズが最適なら、そこを新しいB候補とする。

結果がthroughput-first仮説を否定しても構わない。
しかし、**学習不足のXSを使った不公平な比較によって否定したことにはしない。**
