# サンプル効率 実装計画

最終更新: 2026-08-10

> この文書は実装順序と実験詳細を保持する作業計画である。原仕様は
> `SPEC.md`、後続の設計判断は ADR 0014/0015、短い運用契約は
> `TRAINING_AND_EVALUATION.md` を参照する。`SPEC.md` は後から書き換えない。
>
> 2026-08-10時点の `main` ではPhase 1とPhase 2a、およびPhase 2bの
> 基盤（aux schema v2、36補助目標、valid mask、target統計、gradient
> diagnostics）が実装済み。Phase 3のaction-conditioned target、VS Score
> の実装・auxiliary ablation、検索強度mixの本格運用は次段階である。

## 目的

少ない自己対局から、政策・勝敗価値・攻撃タイミング・相手の危険度を同時に学習できる状態を作る。優先順位は、モデルを大きくすることではなく、1局面から得られる有効な教師信号を増やすことと、観測に存在する情報をTokenizerで落とさないことである。

同時に、補助目標の追加によって方策学習を悪化させていないかを、損失値・勾配・Arena結果の三方向から検証できる実験系を作る。

## 現在の到達点

### 完了済み

- `Observation` には盤面、hold、queue、bag残量、garbage到着予定、combo、B2B、相手盤面および相手のgarbage/combo/B2B/aliveが存在する。
- `Tokenizer` に以下の明示tokenを追加した。
  - `TokenKind::Bag`: 7-bag / 14-bag の残り枚数をpieceごとのcountとして符号化。
  - `TokenKind::OpponentCounters`: 相手のpending garbage、combo、B2B、aliveを符号化。
- Uniform / OnePieceではbag情報を「利用不能」としてmissing flagで表し、空bagと混同しない。
- 新しい相手tokenを含むサンプルはCompact Replay形式へ誤って落とさず、現行どおりschema付きrectangular version 3へフォールバックする。
- Tokenizer回帰テストを追加した。
  - bag内容の変更がtokenを変える。
  - 相手カウンタの変更がtokenを変える。
  - soloでは相手tokenを出さず、duelでは相手カウンタtokenを1個出す。

### 検証済み

- `make test`
- 2026-08-10時点: 282 tests / 1,045,724 assertions / 0 failed
- C++ / PyTorchのfeature widthは変更していないため、`TOKEN_FEATURES=24`の契約は維持する。

ただし、feature widthが同一でもtoken kindやtoken順序の意味は変わり得るため、互換性判定にはwidthだけでなくTokenizer schemaを使用する。

これらのTokenizer/schema変更は現在の`main`へ取り込まれている。以後はfeature widthだけでなくschema version/hashを互換性判定の権威とする。

## 用語とデータ契約

### 終局と打ち切り

trajectory終了理由を明示的に区別する。

- `terminated`: 正常な勝敗・引き分け規則によってゲームが終局した。
- `truncated`: 時間制限、記録停止、shard切断、異常終了などにより、未来が不明なままtrajectoryが終了した。

正常終局後にイベントが存在しないことは既知であるため、`terminated`したtrajectoryでは、終局をまたぐ長い時間窓も原則としてvalidとする。`truncated`したtrajectoryでは、観測できていない未来を0 targetとして扱わず、該当horizonのvalid maskを0にする。

### プレイヤー視点

- 最終勝敗 `z` は固定プレイヤー視点のWDL targetとする。
- attack、garbage、top out、未来状態も同じplayer perspectiveで記録する。
- 相手のイベントを自分のattackとして加算しない。
- 視点反転時にself/opponentのchannel対応が壊れないことをfixtureで検証する。

## 実装順序

## Phase 1: 観測とデータ契約を固定する — 実装済み

1. 現在のTokenizer変更を再検証し、token kind名、token順序、solo/duelのtoken数を固定する。
2. Dataset export / import、GPU bridge、C++ evaluator、PyTorch入力で可変token長が維持されることを確認する。
3. C++/PyTorch parity fixtureを再生成できる状態にし、bagとOpponentCountersを含むfixtureを1つ追加する。
4. manifestおよびdataset headerへTokenizer関連schemaを追加する。
5. この段階の変更を単独コミットする。

### manifest / schemaに記録する項目

- `dataset_version`
- `tokenizer_schema_version`
- `token_kind_order`
- `max_token_count`
- `observation_schema_hash`
- `action_schema_version`
- `aux_target_schema_version`
- `aux_targets`
- `randomizer_type`
- `ruleset_hash`
- `model_version`
- `self_play_seed`
- `termination_reason`

### 受け入れ条件

- `make test`が通る。
- 同一局面・同一seedでtoken列が完全一致する。
- bag、hold、queue、garbage予定、B2B、combo、相手状態のいずれか一つを変更すると、対応する入力が変わる。
- 非対応randomizerでbagの偽情報を生成しない。
- 同じ`TOKEN_FEATURES=24`であっても、Tokenizer schemaが異なるdatasetを誤って互換と判定しない。
- Dataset round-trip後にtoken列、mask、player perspective、termination reasonが一致する。
- 異なるschemaのshardをvalidatorが拒否する。

## Phase 2a: 多段補助targetを生成する — 実装済み（aux schema v2）

aux schema v2は36 targetsを持つ。互換性維持のためのlegacy 4 targetsに加え、
real-timeとplacementの各4区間について、attack / garbage received /
self top out / opponent top outを記録する32 interval targetsを追加した。

追加の自己対局を要求せず、同じtrajectoryから多段targetを抽出する方針は維持する。

### 時間horizon

第一候補として、実時間と意思決定回数の二系統を扱えるschemaを用意する。

- real-time horizon: `1s / 2s / 4s / 8s`
- decision horizon: `1 / 2 / 4 / 8 placements`

初期実装では一方だけを学習に使用してもよいが、dataset上は両者を混同しない。

### 累積値ではなく区間値を基本targetとする

実時間について、次の区間ごとの差分を記録する。

- `0-1s`
- `1-2s`
- `2-4s`
- `4-8s`

各区間について、少なくとも次を持つ。

- 自分が送ったattack量
- 自分が受けたgarbage量
- 自分がtop outしたか
- 相手がtop outしたか

placement horizonについても同様に、

- `0-1 placements`
- `1-2 placements`
- `2-4 placements`
- `4-8 placements`

の区間targetを持てる形式とする。

累積値が必要な場合は、区間値を加算して導出する。

\[
A_{\leq 4s}
=
\Delta A_{0-1s}
+
\Delta A_{1-2s}
+
\Delta A_{2-4s}
\]

top out targetはself/opponentを別channelにし、単一のOR targetへまとめない。

### top out時間の表現

独立した累積確率を直接予測するとhorizon間で非単調な出力が生じ得るため、実装可能なら区間hazardまたは離散time-to-event分類として表現する。

例:

- `survive 0-1s`
- `top out in 1-2s`
- `top out in 2-4s`
- `top out in 4-8s`
- `survive beyond 8s`

初期実装でbinary targetを使用する場合も、horizon間の単調性違反率をログへ出す。

### 実装上の注意

- 最終勝敗 `z` と補助targetを混同しない。
- 未来量は同じplayer perspectiveで集計する。
- `terminated`では終局後のイベントなしを既知として扱う。
- `truncated`では観測不能なhorizonにvalid maskを使用する。
- countは既存のsquashまたはclip規則で正規化する。
- `aux_targets`とtarget順序はコード内定数、dataset header、ドキュメントで固定する。
- 旧4 targetを読む互換性は残すが、新旧schemaを同一batchへ暗黙に混ぜない。
- 同じゲームの隣接局面をtrain/validationへ跨がせない。

### split規則

validation leakageを避けるため、splitは局面単位ではなく少なくとも次のいずれかで行う。

- game単位
- self-play seed単位
- shard単位

可能なら、opponent checkpointを分けたout-of-opponent-distribution評価集合も用意する。

### Phase 2aの受け入れ条件

- 同じtrajectoryから、各区間targetが手計算fixtureと一致する。
- 区間targetを加算した累積値が直接集計した累積値と一致する。
- `terminated`した終局直前サンプルでは、終局後を含むhorizonが不必要にinvalidにならない。
- `truncated`したtrajectoryでは、未知の未来を0 targetとして出力しない。
- player視点を反転したとき、self/opponent attack、garbage、top outの意味が一致する。
- 自分と相手のtop out channelが分離されている。
- 同一gameのサンプルがtrainとvalidationの両方へ入らない。
- `TensorBatch`、dataset serialization、Colab shard validatorが新しいaux幅で一致する。

### 学習前に出力するtarget統計

- mean / standard deviation
- percentile
- zero率
- valid mask率
- self/opponent別出現率
- horizon間相関
- `terminated` / `truncated`別件数
- top outのclass比率
- squash / clip前後の分布

target統計が不自然な場合は、aux headの実装へ進まない。

## Phase 2b: 多段aux headとlossを接続する — 基盤実装済み、ablation継続

PyTorch側はaux schema v2を読み、valid mask付きloss、target統計、shared trunkのgradient norm/cosineを記録できる。残る課題は「学習できるか」ではなく、policy/search/Arenaに実益がある重みとtarget集合を固定データで切り分けることである。

総損失は概念的に次の形とする。

\[
L
=
L_\pi
+
\lambda_v L_v
+
\sum_i \lambda_i L_{\mathrm{aux},i}
\]

初期値は固定せず、policy lossへの干渉を観測しながら調整する。

### 学習ログ

各stepまたは一定間隔で、少なくとも次を記録する。

- `L_policy`
- `L_value`
- aux targetごとのraw loss
- 重み付きloss
- shared trunkに対する各lossのgradient norm
- policy/value間のgradient cosine
- policy/aux間のgradient cosine
- predictionのmean / variance
- valid sample数
- NaN / Inf検出

記録対象の例:

\[
\left\|\nabla_{\theta_{\mathrm{shared}}}L_\pi\right\|,
\qquad
\left\|\lambda_v\nabla_{\theta_{\mathrm{shared}}}L_v\right\|,
\qquad
\left\|\lambda_i\nabla_{\theta_{\mathrm{shared}}}L_{\mathrm{aux},i}\right\|
\]

補助lossの値が小さくても勾配がpolicyを支配し得るため、loss scaleだけで重みを判断しない。

### Phase 2bの受け入れ条件

- 補助targetを有効にした学習でlossとgradient normがfiniteになり、NaN / Infが出ない。
- valid maskが0の要素はlossとgradientへ寄与しない。
- aux headを無効化した場合、従来のpolicy/value学習と数値的に整合する。
- 補助headを追加しても、既存checkpointのpolicy/value推論が壊れない。
- 固定batch上で、各aux targetが学習可能であることを小規模overfit testで確認する。
- policy lossへの勾配干渉をログから観測できる。

## Phase 3: Action-conditionedな補助学習を追加する

Phase 2のstate-level targetが安定してから、各legal actionについて「その手を選んだ直後に何が起こるか」を学習する。

まずは既存の`PlacementOutcome`から追加ロールアウトなしで作れる、次の低コスト特徴を対象にする。

- attack
- cleared lines
- cleared garbage
- resulting holes
- stack height
- all clear
- spin
- immediate top out
- top out riskの明示的定義が存在する場合はその値

### 入力特徴と教師targetを分離する

次の二設計を混同しない。

#### A. afterstate特徴をaction入力へ与える設計

engineで合法actionのafterstateを計算し、その特徴をaction embeddingへ含める。

この場合、入力済みのattack、holes、heightなどを同じheadの補助targetとして予測させない。単なる入力コピーになるためである。

#### B. 現状態とactionから結果を予測する設計

入力は現在状態とactionのみとし、

\[
(s,a)
\longmapsto
\text{attack, holes, height, top out}
\]

を予測させる。

内部表現学習を目的とする場合は、まずこちらを優先する。

### action-conditioned head

Action embeddingに結果特徴を追加するだけで終わらせず、shared state representationとaction representationの双方からheadへ情報が流れる構造にする。

例:

\[
h_s = E(s),
\qquad
h_a = A(a),
\qquad
\hat y(s,a)=H(h_s,h_a)
\]

候補actionのlabelは、実際に選ばれた手だけでなく、合法な他actionも同じ局面のengine outcomeから正確に作れる場合に限って利用する。

探索由来のvalueやvisit countと、決定論的engine outcomeを同一targetとして混ぜない。

### 計算量の記録

全合法actionについてengine outcomeを生成する場合、次を記録する。

- 1局面あたりのlegal action数
- target生成時間
- dataset容量増加
- GPU転送量
- training step時間
- policy改善量あたりの追加計算量

sample-efficiencyの改善をwall-clock efficiencyと混同しない。

### Phase 3の受け入れ条件

- 同一局面の候補action間で、line clear・穴・stack height・top out targetが分離する。
- 学習時と推論時でaction順序・mask・target対応が一致する。
- 同じ特徴が入力と教師の両方へ重複していない。
- legal action permutationを変えても、actionとtargetの対応が維持される。
- masked actionはlossへ寄与しない。
- action-conditioned headを切っても既存checkpointの推論が壊れない。
- 小規模overfit testで候補actionごとの結果を識別できる。

## Phase 4: 学習・Colab運用へ接続する

1. `aux_targets`、dataset version、Tokenizer schema、action schemaをmanifestに記録する。
2. Local / Colab shardのtarget schema、ruleset hash、model versionをvalidatorで検査する。
3. 異なるtarget schemaのshardを同一学習に混ぜない。必要なら変換処理を明示的に行う。
4. train / validation splitをgameまたはseed単位で固定する。
5. 固定されたself-play datasetと固定seedでablationを実行する。
6. Arenaでは同じseed protocol、同じsearch budget、同じ対戦順序を使用する。

## 実験条件

Tokenizer改善、value head、aux headの効果を分離するため、最低限次を比較する。

| 条件 | Tokenizer | value | aux |
|---|---|---:|---|
| A | 旧 | なし | なし |
| B | 新 | なし | なし |
| C | 新 | あり | なし |
| D | 新 | あり | 既存4 aux |
| E | 新 | あり | 多段aux |
| F | 新 | あり | 多段aux + action-conditioned |

計算量が限られる場合でも、value/auxの寄与を切り分ける基準としてpolicy-only条件Bは省略しない。

### 比較時に固定または記録する量

- self-play game数
- training sample数
- optimizer update数
- batch size
- optimizerとlearning-rate schedule
- model parameter数
- search simulations
- replay buffer構成
- opponent checkpoint集合
- training seed
- arena seed
- dataset schema
- wall-clock時間
- GPU時間

複数seedを使用し、単一runだけで結論を出さない。

## 評価指標

### held-out評価

- held-out policy loss
- held-out value loss
- aux targetごとのheld-out loss
- value calibration
- top out予測のAUROC / AUPRCまたは適切な分類指標
- horizon間の単調性違反率
- targetごとのbaseline比較
- gradient normとgradient cosine

### Arena評価

- 勝率
- 95%信頼区間
- VS Score（実装後は標準報告。win rateの代替ではない）
- APM
- APP
- PPS
- 平均生存時間
- top outまでの手数
- attack / garbage cancellation
- 同一seedでのpaired result
- raw policyとsearch policyの双方

比較の主指標は、同じself-play game数または同じ生成計算量に対するheld-out性能とArena性能である。

学習時間短縮だけをsample-efficiencyの改善とはみなさない。また、補助lossが改善してもArena性能が改善しなければ、戦略的に有効な内部表現を獲得したとは結論しない。

## 実装しないこと

- この段階でTransformerを捨てて別アーキテクチャへ移行しない。
- プレイヤー視点で観測できないhidden queueやgarbage holeをTokenizerへ漏らさない。
- self-playの勝敗targetをheuristic scoreへ置き換えない。
- resulting holesやstack heightを直接「良さ」の報酬として扱わない。
- 同じ特徴をaction入力と補助targetの両方へ置いて、表現学習の効果と誤認しない。
- `terminated`と`truncated`を同一視しない。
- game内の隣接局面をtrain/validationへ無作為に分割しない。
- schemaが異なるshardをwidth一致だけで結合しない。
- Google Apps Scriptをseed、label、dataset mergeの権威にしない。Colab shardのseedとschemaはmanifestおよびvalidatorで管理する。

## 次の実行順序

1. `make test` とdataset/schema round-tripを維持し、現行36-target contractを壊さない。
2. policy-only / WDL / legacy aux / multi-horizon auxを、同一dataset・split・minibatch schedule・複数seedで比較する。
3. Arena出力へVS Scoreを追加し、APM/APP/PPSと同じpaired条件で記録する。
4. VS Scoreをaux targetとして使う実験は、計算式・ruleset解釈・schemaを固定してから独立ablationとして行う。WDL rewardは変更しない。
5. CNN/Transformer系の比較はADR 0013の方針に従い、held-out policy loss単独で結論しない。
6. Phase 3のaction-conditioned targetを小規模overfit testから実装し、入力特徴の自己コピーになっていないことを確認する。
7. self-play loopが安定したら、浅いsearch中心 + 少量の深いsearchという生成mixをprovenance別に試す。
8. 基本的なstacking/Quad/T-spinが確認できてからtiming/相殺外しの探索を強め、delay action頻度・cancellation・VS Score・Arenaで検証する。
9. MoE/SAEはADR 0016どおり、強いdense baselineの後段に置く。
