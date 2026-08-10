# ADR 0007: Searchより先にEvaluator interfaceを定義する

## 状態

採用。

## 背景

当初は、legal-move generationがbottleneckだと仮定し、dummy evaluationでMCTS skeletonを先に作る予定でした。しかし実測すると前提が逆でした。

初期sandbox（2 CPU cores）、約100 state tokens・約40 legal actionsでの測定:

| 処理 | 1 positionあたり |
|---|---:|
| `generate_for_piece` | **0.088 ms** |
| TetraFormer-S（当時5.3M params、spec §9.5構成） | **8.6 ms** |
| half-size variant（0.67M） | 1.18 ms |
| tiny dev proxy（0.09M） | 0.35 ms |

network evaluationは周辺処理より約**100倍高価**でした。

64-simulation searchを素朴にscalar inferenceすると、movegen約5.6 msに対してinference約550 msとなり、spec §19.4の30 ms級budgetを大幅に超えます。movegenを0 costへ近づけても本質的な解決にはなりません。

## 決定

MCTSを実装する前に `Evaluator` interfaceを固定し、searchをそのinterface前提で設計します。後からbatchingをretrofitしません。

### 1. work unitをbatchにする

`evaluate()` はposition vectorを受け取ります。`evaluate_one()` はそのthin wrapperにし、scalar専用pathを作りません。

Spec §11.1はbatched leaf inference、§19.4は高いbatched evaluation比率を要求しています。MCTSでのbatchingは単なるoptimizationではなく、次を含む**search structureそのもの**です。

- leaf collection
- virtual loss
- pending-node bookkeeping
- evaluator batch scheduling

### 2. Policyは合法actionごとの可変長vectorにする

固定 `(x, rotation)` gridへ押し込みません。

これにより、合法action数がpositionごとに違っても同じinterfaceを使え、spin provenanceやhold/delayなど「同じ座標だけでは区別できないaction」を別にscoreできます。

### 3. baseline evaluatorを同時に用意する

- `UniformEvaluator`: null hypothesis。search testの基準fixtureとして使う。
- `HeuristicEvaluator`: deterministicな意見を持つbaseline。初期search opponent / teacherとして使う。

`Observation` は `RulesetConfig` をvalueで保持し、evaluatorがwrong rulesetと組み合わされないようにします。

## 帰結

- trained weightがなくてもsearchを実装・test・profileできる。
- trained modelへの差し替えは `Evaluator` implementationの変更に閉じる。
- `HeuristicEvaluator` は初期測定で、greedy priorだけでも10 seeds中8回500-piece gameを生存し、約197 lines clearするbaselineになった。
- `ChunkedEvaluator` により、fixed-shape backendのmaximum batch sizeをsearch側へ漏らさず制限できる。
- evaluator自身がposition数とbatch数を記録するため、batch-efficiencyを最初から測定可能。

## 初期sandboxについて

このADRのbenchmarkは、2-core CPU・GPUなしの初期sandboxで得たhistorical measurementです。当時はPyTorchをvenvへ導入できたものの、full-scale trainingやstrength evaluationは現実的ではありませんでした。

現在はROCm GPU bridgeとGPU Arenaが実装済みです。このため「M2に必要なGPU strength evaluationは将来課題」という当時の制約は解消されていますが、**Evaluator-first / batch-firstという設計判断自体は現在も有効**です。
