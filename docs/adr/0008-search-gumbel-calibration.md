# ADR 0008: PUCTとGumbel、およびGumbelを既定とする理由

## 状態

採用。

## 背景

Spec §11.1はPUCTを定義し、§11.2はlow simulation countでpolicy improvementを安定させるためGumbel sequential halvingを推奨しています。本botが主に扱う16–128 simulations / 30–50 ms級budgetはまさにその領域です。

PUCTとGumbelをADR 0007のbatched `Evaluator` 上へ実装しました。Gumbelの校正では、同じbehavioural testが複数のbugを発見しました。

使ったtest positionは、bottom 2 rowsが1 columnだけ欠けており、**clearできるplacementが実質1つ**という局面です。correct searchならbudgetにかかわらずその手を見つける必要があります。

## 発見した問題

### 1. Heuristic priorが極端すぎた

softmax temperatureの設定によりtop priorが0.9918となり、searchが事実上1 actionへcollapseしていました。

score spreadに応じてtemperatureをscaleし、boardごとにpriorが十分な情報量と探索余地を持つよう修正しました。

### 2. Unvisited actionを `q = 0` としていた

このvalue scaleでは多くのposition valueがnegativeです。そのためunvisited actionの0が常に「既に調べたnegative action」より良く見え、sequential halvingが**調べて情報を得たactionほど捨てる**状態になっていました。

unvisited actionはparent valueをinheritするよう変更しました。

### 3. Gumbel noiseがlogitを圧倒していた

weak policyで約34 placementsのlogit spreadが約0.6 natsしかない一方、Gumbel noiseは約5程度のspreadを持ち、root candidate selectionがほぼrandomになっていました。

実測ではGumbelが自身のpolicy-only baseline 200 piecesに対して平均27 pieces程度しか生存しない状態でした。

## 決定

`SearchConfig::gumbel_noise_scale` を追加し、defaultを **0.05** とします。また `use_gumbel` をdefault trueとします。

forced-clear positionでのcalibration:

| noise scale | 16 sims | 64 sims | 256 sims |
|---|---|---|---|
| 1.0 | wrong | wrong | wrong |
| 0.2 | wrong | wrong | correct |
| 0.05 | correct | correct | correct |

trained policyが十分confidentになった場合、このscaleを1へ近づける余地はあります。self-play explorationの主要knobとしてroot noiseも別に存在します。

## 当時の測定結果

steady garbage stream（8 placementsごとに2 lines）、6 games、最大250 placements:

| 条件 | pieces | survived | attack / piece |
|---|---:|---:|---:|
| policy-only | 228.7 | 5/6 | 0.169 |
| gumbel 32 | 250.0 | 6/6 | 0.207 |
| puct 32 | 62.7 | 0/6 | 0.136 |
| puct 128 | 250.0 | 6/6 | 0.249 |

初期2-core CPU + heuristic evaluatorでのtiming:

| 条件 | ms | mean batch |
|---|---:|---:|
| gumbel 32 | 11.5 | 5.5 |
| gumbel 64 | 20.5 | 4.9 |
| puct 64 | 23.5 | 13.0 |
| puct 128 | 46.7 | 14.3 |

これらはsearch algorithm校正時のhistorical measurementであり、現在のneural Championのstrength benchmarkではありません。

## 重要な制限

**PUCTは、legal actionあたりおおむね2 simulations未満のthin budgetで不安定になりやすい**という実測上の性質があります。

Tetris positionでは25–50 placements程度のbranchingがあり、flat prior + small budgetだとPUCTがarbitrary tie-breakへvisitを集中し、priorをそのままfollowするより悪化する場合があります。

上の測定ではpuct-32が平均62.7 pieces、policy-onlyが228.7 piecesでした。同budgetでGumbelは複数actionをsurveyできました。

`puct_needs_enough_simulations_to_beat_its_prior` がこのfailure modeをtestとして固定しています。したがってこれは「後に偶然壊れたregression」ではなく、Gumbelをlow-budget defaultとした具体的理由です。
