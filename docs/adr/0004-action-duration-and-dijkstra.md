# ADR 0004: action durationと、旧movegenをDijkstraへ変更した理由

## 状態

route enumerationの実装方式はpure Cobra integrationにより置換済みです。

一方で、**actionに実時間costを持たせること、deliberate waitを別に表現すること、bounded delay binをaction spaceへ含めること**は現在も採用中のcontractです。

## 背景

Spec §8.4はplacementのtime costとdelay binを定義し、spec §12は「今相殺する / garbageを受ける / attackを保留する」といったtimingを、hand-written ruleではなく正しいclock上で**学習**させることを要求しています。

この変更以前は全actionを定数時間として扱っていました。CLIから `lock_piece` へhardcoded 20 ticksを渡していたため、timing差をmodelが表現できませんでした。

## 決定

### 1. すべてのactionに実時間を持たせる

`HandlingModel` が `RulesetConfig::movement` のDAS、ARR、SDF、ARE、lock delayから各inputのcostを導出します。

`PlacementAction` は次を持ちます。

- `base_duration`: input sequenceを実行するcost
- `delay_ticks`: deliberate wait
- `total_duration()`: 両者を合わせた総時間

### 2. 旧project-side generatorではBFSをDijkstraへ変更した

Spec §8.2はBFS / Dijkstraのどちらも許容します。しかしinputごとにcostが異なると、plain BFSは**depth順**にnodeをsettleするため最短時間を保証しません。

たとえばDAS slideとtapは同じ1 edgeでも時間が違います。高価なrouteから先に同じnodeへ到達すると、そのcostを保持したままchildへ伝播し、後からparentが安くなってもstale sumが残りました。

実際、emitしたinput sequenceを最初から再実行したcostとincremental cost modelが最大5 ticks食い違うcaseがありました。

旧実装ではedge weightがsmall integerでsettled costがmonotoneだったため、binary heapではなくDial's algorithmのbucket queueを採用しました。

### 3. Airborne nodeもlanding candidateにする

`L HD` と `SD L HD` が同じcellへ到達するとき、前者のほうが安い場合があります。

soft drop後のstateだけをlanding candidateにすると、より安い空中routeを見つけていても採用できません。そのためmove/rotationで到達したairborne nodeからimplicit hard dropした結果もlanding candidateへ含め、trailing redundant `SoftDrop` をemit sequenceから除去しました。

### 現行Cobra implementation

上記Dijkstra traversal自体は現在削除されています。

現行move generatorは次の流れです。

1. Cobra `MoveList` でlegal targetを列挙する。
2. Cobra `PathFinder` からtap pathとfinesse pathを得る。
3. 両pathをproject側 `HandlingModel` で再実行・価格付けする。
4. validなrouteのうち安いものを採用する。
5. gravity reachability、canonical sequence、delay bin contractをproject側で維持する。

したがって、「合法状態探索をDijkstraで行う」という具体実装は置換されましたが、**action durationを実行可能なinput sequenceから厳密に計算し、安いvalid routeを採る**という意味論は残っています。

## 帰結

旧Dijkstra implementationでは次を確認しました。

- length 4までの全input sequence brute forceと比較し、`action_cost_is_the_true_shortest_path` でoptimalityを検証。
- T placementのcheapest costが2→1 ticks、dearestが12→7 ticksへ修正。
- exact pricingによりgenerationは117→155 µs/callへ遅くなった。
- landing deduplicationを `evaluate_placement` より前に行い、213→155 µsまで回復。
- I pieceで異なる `(x, y, rot)` が同一cell集合を作るcaseから、merge時にinputだけを置換して座標を残すself-consistency bugを発見。5191 actions中162 actionsが「自身のcanonical inputで自身の宣言位置へ行けない」状態だった。
- `duration_matches_replaying_the_canonical_sequence` が、positionとpriceの両方を再実行で確認するようになった。

## 検討した代替案

### BFSのまま各sequenceを再実行してpriceだけ計算する

単純でcorrectですが、pathを二重にwalkし、BFS自体がroute selectionを誤るためnon-optimal costを返します。

### Airborne candidateをhorizontal move後だけに制限する

少し速い一方、mid-air rotationがdrop後rotationより安くなるcaseを落とすため不正確でした。optimality testでrejectしました。
