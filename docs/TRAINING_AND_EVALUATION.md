# 学習・評価プロトコル

学習実験を比較可能に保つための運用契約です。この文書が、改善の判断基準、目的関数、datasetの由来、比較実験、Arena、timingの段階導入、Champion promotionの規則を一元的に管理します。

- 判断の理由は[ADR 0013（構造比較）](adr/0013-architecture-ablation-and-local-geometry.md)、[ADR 0014（目的と補助目標）](adr/0014-objectives-auxiliary-targets-and-vs-score.md)、[ADR 0015（自己対局の由来）](adr/0015-selfplay-provenance-search-mixture-and-timing-curriculum.md)を参照してください。
- 補助目標の生成の詳細は[サンプル効率の実装計画](SAMPLE_EFFICIENCY_PLAN.md)を参照してください。
- 実行コマンドは[GPU手順](GPU_WORKFLOW.md)を参照してください。

## 1. 改善の判断基準

目的は単一のoffline lossを最小化することではなく、管理された計算予算のもとで実際に強いplayerを作ることです。証拠の優先順位は次のとおりです。

1. 固定した探索予算とpaired seedによるCandidate対ChampionのArena。勝率と信頼区間を報告します。
2. 同じpaired条件でのVS Scoreと戦闘診断。
3. APM、APP、PPS、生存時間、相殺、timingの統計。差が出た理由の説明に使います。
4. 検証用の方策、価値、補助目標の指標。表現学習と学習の健全性の診断に使います。

検証用の方策損失が低くても、強いplayerであるとは限りません。2026-08-08のCNN ablationでは、TransformerがTransformer由来の教師方策をよく模倣した一方、CNNは探索内でより強い挙動を示しました（[実験レポート](CNN_ABLATION_20260808.md)）。

モデルサイズと探索量は、同じ推論予算を取り合う関係として扱います。パラメータ数そのものは目的にしません（[ADR 0014（サイズと探索予算）](adr/0014-model-size-vs-search-budget.md)）。

## 2. 目的関数

終局の勝敗を価値とrewardの基準として維持します。密な教師信号を追加しても、reward shapingへ置き換えません。

\[
L = L_{\pi} + \lambda_v L_v + \sum_i \lambda_i L_{\mathrm{aux},i}.
\]

補助目標には次の制約を課します。

- 記録済みのtrajectoryと、定義済みのplayer視点から導出できること。
- `terminated` と `truncated` を区別すること。
- 観測できない未来のhorizonを0として学習せず、valid maskで除外すること。
- 推論時に見えないhidden stateを補助labelの生成に使わないこと。
- loss weightは、損失値だけでなくshared trunkの勾配ノルムとcosineも見て決めること。
- 管理されたablationで、方策とArenaに対して中立以上の場合だけ残すこと。

## 3. 密な補助目標

同じtrajectoryから複数の教師信号を抽出し、1局面あたりの学習情報量を増やします。ただし、label数が増えても独立したゲーム数は増えません。

現在のaux schema v4は52 targetsです。

| 系統 | 内容 |
|---|---|
| legacy（4） | 互換性のためのtarget |
| 区間target（32） | 実時間4区間（0–1、1–2、2–4、4–8秒）と配置数4区間（0–1、1–2、2–4、4–8配置）それぞれで、attack、受けたgarbage、自分のtop out、相手のtop out |
| garbage消去（8） | 同じ実時間4区間と配置数4区間で、消したgarbageライン |
| garbage相殺（8） | 同じ実時間4区間と配置数4区間で、相殺したgarbage |

今後の候補は、エンジンから厳密に計算できるaction-conditionedな直接結果と、ablationを経た戦闘統計の予測です。VS Scoreを補助目標に使う実験は、Arena reportへのVS Score導入とは別のablationとして行い、WDL rewardとpromotion ruleは変更しません。

## 4. Datasetの由来

本格的なrunでは、各sampleがどのcode、checkpoint、ruleset、schema、探索設定、seedから生成されたかを後から特定できる状態にします。

最低限、次を記録・検証します。

- repository commit
- checkpointの識別子とhash
- ruleset hash
- dataset、tokenizer、action、補助目標のschema version
- 探索アルゴリズム、simulation数、determinization数、root noise
- ゲームとseedの区間、shardの識別子
- 終了理由
- sample数、生成元のgeneration

生成済みshardは変更しない入力として扱います。datasetをバイト連結せず、schemaが異なるshardを暗黙に混ぜません。

次のsourceは、通常の自己対局と別のsourceとしてmanifestに記録し、混合比を独立に制御します。

- **Reanalyse:** 履歴を厳密に再生し、選んだrootだけを再探索した方策を別shardに出力します。元のshard、`chosen_action`、勝敗、補助labelは変更しません（[ADR 0015（Reanalyse）](adr/0015-reanalyse-historical-target-refresh.md)）。
- **人間リプレイ:** C++の合法手生成で検証した配置だけをdatasetにします（[人間リプレイによる事前学習](HUMAN_REPLAY_PRETRAINING.md)）。
- **position-start、recovery curriculum:** 導入する場合は別のsource classとして記録します。

## 5. 学習用と検証用の分割

隣接局面を無作為に分けず、ゲーム、seed、shardの単位で分割します。

連続したseed範囲を持つshardで、seedを並べて上位80%と下位20%に切ると、検証集合が特定のshardだけになる危険があります。既定ではゲームseedの安定hashで分割します。

構造比較では同じ分割を共有し、可能ならminibatchのscheduleも共有します。

## 6. 構造比較実験

Transformer、CNN、hybrid、将来のMoEを比較するときは、次を固定または記録します。

- 学習sample集合と分割
- optimizerとlearning-rate schedule
- update数とbatch size
- パラメータ規模
- 学習seed
- 探索予算とArena seed
- replayの混合比、相手とChampionのcheckpoint
- 実時間とaccelerator時間

1 seed、1指標の結果で基準モデルを置き換えません。構造の変更は、実験、Candidate checkpoint、通常のChampion promotion gateの順に進めます。

## 7. Arena report

Arena reportには最低限次を含めます。

| 分類 | 指標 |
|---|---|
| 昇格判定 | 勝ち、負け、引き分け、勝率、95%信頼区間、paired seedの条件 |
| 戦闘 | VS Score、APM、APP |
| 速度 | PPS |
| 生存 | 平均生存時間、top outまでの配置数 |
| garbageの応酬 | 送った、受けた、相殺したattack、相殺効率 |
| timing | 最速actionとdelay actionの比率、`WAIT_FOR_EVENT` の使用率、timingに関係する相殺 |
| 探索依存性 | 必要に応じて、探索なしの方策と探索後の方策の両方 |

VS Scoreは `include/tetra/stats.hpp` で、TETR.IOの式「(送ったライン + 消したgarbageライン) / piece数 × PPS × 100」として実装されています。GPU Arenaが両者のVS Scoreを報告します。VS Scoreは勝率の代わりではなく、APM/APPだけでは区別しにくい戦い方を説明する診断指標です。

## 8. Timingと相殺外しのカリキュラム

合法手生成は `WAIT_FOR_EVENT` を含むdelay binを出せますが、networkが使いこなすまではtimingを学習済みとみなしません。

1. 安定した積み、Quad、T-spinなどの基本戦術を先に獲得します。
2. 単一の数値ではなく、対局の観察と戦闘指標を併用して準備状況を見ます。
3. timingのablationは、固定したclean benchmarkの平均APPが0.5を超えるまで再開しません。
4. 再開後に、garbage timingと相殺外しを含むdelay actionの探索とデータ網羅を強めます。
5. action頻度、相殺の応酬、VS Score、paired Arenaで能力が実在するか検証します。

clean benchmarkは、attackを相手へ届けない自己対局（`--no-attack-delivery`）を、timing無効、64 sims、2 determinizations、root noise 0、300 pieces、固定seed `19000000`、`19000001`、`19000002` で実行する条件です（[2026-08-15 handoff](HANDOFF_STACKING_LC3_20260815.md#6-clean-benchmark-protocol)）。通常の1v1 ArenaのAPPとは区別します。

APP約0.5は平積みQuadの理論的な目安です。ADR 0015では粗い診断として導入し、2026-08-15以降はtiming再開の条件として使っています（[ADR 0013（LC3）](adr/0013-lc3-search-runtime.md)）。

待つこと自体に正のrewardを付けません。待つべき局面かどうかは、探索と実際の勝敗から学習させます。

## 9. 自己対局の探索混合

自己対局のloopが安定した後は、すべての局面を一様に深く探索するより、次の混合を基本候補とします。

- 大部分を浅い探索にして、安価に広い状態を網羅します。
- 少量を深い探索にして、質の高い方策と価値のtargetを加えます。
- 複数の探索強度や、意図的に不完全な局面、recoveryが必要な局面を含め、現在の方策の分布だけに閉じないようにします。

サンプル効率を比較するときは、ゲーム数または生成の計算量を揃えます。

## 10. Champion promotion

Championは保護された成果物です。学習とablationから得られるものはCandidateです。CandidateがChampionを置き換えられるのは、設定済みのpaired Arena gateを満たした場合だけです。

次のいずれか単独では、promotionの理由になりません。

- 検証用の損失が下がった。
- VS Score、APM、APPが高い。
- 1回の短いArenaで勝った。
- 補助目標の損失が改善した。

この分離により、積極的な実験を行っても、比較対象のChampionが途中で変わりません。
