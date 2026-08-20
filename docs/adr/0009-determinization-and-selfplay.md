# ADR 0009: Determinizationと自己対局pipeline

## 状態

採用。

## 背景

searchからtrainable networkへ進む前に、2つの構造的問題がありました。

1. searchがhidden informationを読めていた。
2. training sampleを正しく保存するcontractがなかった。

## 情報漏洩

`Player` はpreviewより先を含むpiece queue全体を所有します。search nodeが `Player` をcopyするだけだと、tree depthが `preview_count` を超えた時点で**実際のhidden future**を使ってplanningできてしまいます。

例:

```text
preview_count = 5
playerが見える:   S O J Z I
searchが実際に持つ: S O J Z I L Z L O T
```

Spec §3.2 / §18.3はこれを禁止しています。

self-playでは特に危険です。crashもtest failureも起こさず、engineがpeekしたfutureでしか成立しないsetupのvalueを静かに過大評価し、そのpolicyをtraining dataとして増幅してしまうからです。

## 決定1: rootでdeterminizeする

`Player::determinize(seed)` はplayerが合法的に見えないfutureを捨て、再sampleします。

次の2つを維持します。

### Visible previewは変えない

Determinizationによって現在playerが見ているpreviewを変更しません。

### Bag informationを保つ

7-bagの残り構成はpublic informationであり、人間playerもcountできます。bagを無視してfutureをresampleすると、人間より情報の少ないagentになってしまいます。

そのためdiscardしたpieceをbagへ戻してreshuffleします。

初期版では戻し忘れがあり、14-piece window内にOが3つ、Lが1つというbag guarantee違反を実際に生成しました。このbugを修正してからbag balanceをtestで固定しました。

`SearchConfig::determinizations > 1` の場合は、independentにsampleした複数futureでsearchし、visit distributionを平均します。これはspec §11.3のchance nodeをparticle approximationした形です。

導入当時の64-simulation searchでは19.1 ms → 19.4 msで、resamplingがroot queue operationだけのためcostは小さく抑えられました。

## 決定2: sampleにはraw stateではなくtokenを保存する

`TrainingSample` は `Player` そのものではなく、tokenized observationとaction embeddingを保持します。

これにより:

- sampleをsimulator internal stateからdecoupleできる。
- observation maskで除外したhidden stateをlearnerが後から読む経路を構造的に閉じられる。
- on-disk contractをmodel inputへ近づけられる。

各sampleは `ruleset_hash` と `model_version` を持ち、`ReplayBuffer` はforeign rulesetをdropできます。spec §14が禁止する「ruleset identityなしの混合」を防ぎます。

> 後にADR 0012で、再構成可能な単盤面sampleについてはReplay+πをcompactに保存しload時にtokenを再生成する形式も追加しました。二盤面self-playは相手event streamの再構成が必要なため現行rectangular v3を使います。

## 決定3: rewardはgame resultだけにする

Spec §12.3の方針に従い、garbage penaltyやstack heuristicをterminal rewardへ混ぜません。

`GameRecorder` は `outcome` をwin/draw/lossだけから設定し、attack・garbageなどは別のauxiliary fieldへ保存します。

piece limitまで生存したgameも**winではなくdraw**として扱います。相手を倒していないtrajectoryをwinとしてrewardすると、stallingを最適行動として学ぶ危険があります。

## 帰結

導入時点で、

```text
self-play -> sample -> replay buffer -> training batch
```

がend-to-endで接続されました。

当時の2-core CPU・16-simulation searchでは約157 placements/sでした。この値はhistorical pipeline measurementです。

TetraFormerを接続するときは `Evaluator` implementationを追加すればよく、search、sample contract、buffer、token layoutを別々に書き換える必要がなくなりました。

## 2026-08の追補

本ADRの根本制約は現在も維持します。

- hidden informationはmask / determinizationする。
- terminal objectiveはactual game resultにする。

後続のsample-efficiency workはこの制約を弱めず、**supervision densityとcurriculum**だけを拡張します。

- [ADR 0014](0014-objectives-auxiliary-targets-and-vs-score.md): WDLをobjective anchorに残したままmulti-horizon auxiliary targetとVS Score diagnosticsを導入する。
- [ADR 0015](0015-selfplay-provenance-search-mixture-and-timing-curriculum.md): provenance-locked self-play、search-budget mixture、Champion保護、timing/相殺外しのstaged learningを定める。
