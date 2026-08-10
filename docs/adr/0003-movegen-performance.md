# ADR 0003: movegenをMCTS inner loopとして最適化する

## 状態

**旧project-side generatorについては置換済み。** 現在の合法配置生成はpure Cobra integrationを使用します。以下の測定と最適化は、削除済みgeneratorのhistorical recordです。

## 背景

Spec §19.4は、policy-onlyで5 ms、64–128 simulationsで30–50 msというinference targetを置いています。各simulationはlegal move listを必要とするため、movegen throughputはsearch budget全体の上限になります。

最初のcorrect implementationは1 callあたり279 µsで、約180 callsだけでsearch budget全体を消費する水準でした。

## 決定

旧generatorへ3つのtargeted optimizationを導入しました。各変更について、同一benchmarkの777,136 placements、determinism hash、全testがbyte-identicalであることを確認しました。

1. **Parent-linked BFS path**
   - BFS expansionごとに `vector<Input>` をcopyする方式を廃止。
   - nodeはparent indexと単一inputだけを持ち、実際にemitするplacementについてのみpathを再構築。
   - 279 → 163 µs。

2. **Flat generation-stamped visited table**
   - 20-bit packed stateを `unordered_map` ではなくdirect-mapped arrayへindex。
   - 毎callの全clearを避け、generation counterでstale entryをO(1) invalidation。
   - 163 → 124 µs。

3. **Scratch buffer再利用**
   - arena、frontier、landing vectorを `thread_local` scratchへ移動。
   - steady-state callでheap allocationを行わない。
   - 124 → 117 µs。

collision detectionもprecomputed per-row bitmaskへ変更しました。これは主要bottleneckではありませんでしたが、naive implementationとの5.9M differential casesで一致を確認しています。

## 帰結

旧generatorでは次の結果を得ました。

- 初期実装比で約2.4倍高速化。
- `thread_local` scratchにより、`MoveGenerator` 自体をthread間で共有しつつbufferはthread-localとなった。
- bitmask collisionではbounding boxの `x` がnegativeでもfilled cell自体はfield内にあるcaseがあり、shift directionの扱いが必要だった。このbugはdifferential testで発見した。

## 現行実装

現在の `MoveGenerator` は、Cobraの固定10x40 boardを直接使用します。

- Cobra `MoveList` がlegal targetを列挙する。
- all-target `PathFinder` がpiece/input modelごとのcanonical input pathをまとめて供給する。
- removed legacy generatorへのfallbackはない。

同じ標準10x40・20,000 call CLI benchmarkでは次の測定でした。

| 実装 | µs / call |
|---|---:|
| pure Cobra移行前のhybrid adapter | 132.4 |
| pure Cobra | **99.0** |

したがって、このADRの「movegen throughputをsearch architecture上の重要制約として測る」という原則は維持しつつ、具体的な旧BFS optimizationはpure Cobraによって置き換えられています。
