# ADR 0006: Gravityをreachability constraintとして扱う

## 状態

採用。

## 背景

`MovementCfg::gravity_num/den` はrulesetに含まれhashにも反映されていましたが、初期move generatorはgravityを無視していました。

通常速度（1/60 G）では、placementが数ticksで終わる一方gravityが1 cell落とすまで60 ticksかかるため、実害はほぼありません。しかしhigh gravityでは違います。generatorがplayerには物理的に到達不能なplacementをemitでき、20Gではspawnした瞬間にpieceがfloorへ落ちます。

## 決定

ADR 0004でactionをtick単位に価格付けできるようになったため、gravityをreachability constraintとして導入します。

旧generatorでは、`cost` ticks後にgravityが

\[
\left\lfloor \frac{cost \cdot gravity\_num}{gravity\_den} \right\rfloor
\]

cells落下させたとみなし、その時点でpieceが存在できる高さより上にあるnodeをunreachableとしてexpandしませんでした。

現行Cobra pathでも同じ不変条件を維持しています。Cobraが返したcanonical path prefixをproject側 `execute_inputs` / `HandlingModel` で再生し、各prefixの経過tickに対してgravity上の到達可能性を検査します。

Gravityはexact rationalのまま保持し、comparisonもinteger arithmeticで行います。これによりspec §5.2 / §18.1のbit reproducibilityを維持します。

common caseのoverheadを避けるため、gravityが `gravity_check_threshold` より遅い場合はcheck自体をskipします。defaultは8 ticks/cellです。

## 帰結

旧実装導入時の測定:

- empty board・20Gでplacement countが162→58へ減少。
- 1G以下では変化なし。`normal_gravity_does_not_restrict_placements` で固定。
- faster gravityはoptionを増やさないというmonotonicityをtest。
- benchmark上は174.8→173.2 µs/callで、測定可能なoverheadは見られなかった。

数値は導入当時のhistorical measurementです。現行Cobra implementationのthroughputはADR 0003を参照してください。

## 制限

### Lock delayと `reset_limit` は完全にはmodelしていない

pieceがstackへ接触した後も `lock_delay` ticksの間は操作でき、reset回数には上限があります。high gravityではこのwindowが主要なmanoeuvring timeになります。

現在のgravity modelはこの点で保守的です。本来到達可能なhigh-gravity placementを一部rejectする可能性があります。

### Gravity中のhandling近似

現行checkはcanonical input prefixを再実行して経過時間と位置を比較しますが、完全なcontinuous handling/lock-delay simulatorではありません。したがって、gravity correctnessの残課題は `ROADMAP.md` のengine correctness項目として追跡します。
