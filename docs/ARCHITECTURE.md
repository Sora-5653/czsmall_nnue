# 現在のアーキテクチャ

現在の `main` に存在する構成と、各層の責務の境界を説明します。当初仕様は[SPEC](SPEC.md)、仕様策定後の設計変更の理由は[ADR](adr/README.md)、実装の進み具合と次の作業は[ロードマップ](ROADMAP.md)で管理します。

## 1. シミュレーションとルール

ゲーム上の事実に影響する状態遷移は、すべてC++エンジンが担当します。

- ruleset設定とhash（spawn位置、kick table、gravity、garbageの設定を含む）
- 盤面、piece、spin、attack、garbage、B2B、combo、Surge
- Cobraによる合法配置生成
- action duration、gravity reachability、delay bin
- 観測の秘匿情報maskと、tokenおよびactionの生成
- PUCT/Gumbel探索、determinization、二盤面のevent順序
- replayの検証、dataset serialization、Arenaのゲーム進行

学習側では、近似したゲームを別に実装しません。学習labelと推論時の意味を1つのシミュレータに固定するためです。

## 2. 観測とactionの契約

エンジンは、mask済みのplayer観測を、可変長のstate token列と合法action列へ変換します。`TensorBatch` はそれらをpaddingし、maskを保持します。同じ契約をC++推論、PyTorch、dataset readerが共有します。

観測には公開情報だけを含めます。preview以降のhidden queueや、未着弾garbageのhidden hole columnは、tokenにも探索にも渡しません。`tests/test_observation.cpp` で回帰を検証しています。

互換性はfeature widthだけでは判定しません。`include/tetra/schema.hpp` が、tokenizer、observation、action、補助目標のversionとhashを定義します。同じ `TOKEN_FEATURES=24` でも、意味やtokenの順序が違うdatasetは互換として扱いません。

関連: [ADR 0009](adr/0009-determinization-and-selfplay.md)、[ADR 0010](adr/0010-cpp-python-handover.md)、[ADR 0012](adr/0012-compact-dataset-replay-pi.md)。

## 3. 評価関数（neural evaluator）

`trainer/tetraformer.py` のTetraFormerが基準モデルです。Transformerのstate encoder、可変長のaction scoring、WDL価値head、補助予測headを持ちます。

| preset | 役割 |
|---|---|
| `s` | TetraFormer-S。構造比較の基準 |
| `xs` | 高速探索の比較基準。人間リプレイからの蒸留先 |
| `teacher1m` | XSへ蒸留する教師。観測、action query、方策/価値headはXSと同じ |

モデルサイズと探索量は、同じ推論予算の配分として評価します（[ADR 0014（サイズと探索予算）](adr/0014-model-size-vs-search-budget.md)）。

`trainer/ablation_models.py` には、同じforward契約を持つCNNとCNN+Transformerの実験モデルがあります。2026-08-08の比較では、教師方策の模倣、WDLの学習、探索下の強さが一致しませんでした。結果の詳細は[実験レポート](CNN_ABLATION_20260808.md)、採用した方針は[ADR 0013（構造比較）](adr/0013-architecture-ablation-and-local-geometry.md)を参照してください。

## 4. 探索

評価関数が方策と価値の推定を返し、C++の探索が改善されたroot方策へ変換します。

- 低simulation数ではPUCTの探索が合法手数に対して薄くなるため、Gumbel sequential halvingを既定とします（[ADR 0008](adr/0008-search-gumbel-calibration.md)）。
- 見えない未来のpieceはroot determinizationで扱います。自己対局の教師は、人間playerが観測できる情報の範囲で生成されます（[ADR 0009](adr/0009-determinization-and-selfplay.md)）。
- 探索runtimeはLC3の構造を取り入れています（[ADR 0013（LC3）](adr/0013-lc3-search-runtime.md)）。
  - 複数のC++探索が、Python GPU側の共有 `StreamingInferenceQueue` へ要求を送り、互換なmodelの要求を1回のforwardにまとめます。
  - `SearchPolicy` を探索本体から分離しています。
  - edgeごとに `P`、`Q`、`N`、`N_inflight`（virtual loss）を保持します。
  - Gather、Eval、Backpropの段階を分け、queue待ち時間やbatch充填率などのtelemetryを記録します。

## 5. 自己対局とdataset

基本のgeneration loopは次のとおりです。

1. checkpointと、repository、ruleset、schemaの由来を固定します。
2. 探索設定とseed範囲を記録して自己対局を生成します。
3. shardとmanifestを、変更しない成果物として保存します。
4. 管理されたreplay混合で学習します。
5. paired seedのArenaでCandidateとChampionを比較します。
6. promotion gateを満たした場合だけChampionを更新します。

データの由来と比較の規則は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md#4-datasetの由来)で管理します。

### dataset version

| version | 形式 |
|---|---|
| 4（現行） | rectangular形式。schemaと終了理由に加え、Reanalyseが履歴を再生するための `game_seed`、`move_number`、`player_perspective`、`chosen_action` を保持します |
| 3 | 以前のrectangular形式。readerは読み込みを維持します |
| 2（`VERSION_COMPACT`） | Replay+πを保存し、再生可能な単盤面trajectoryからtokenとactionをload時に再生成する形式 |
| 1 | 初期のrectangular形式。後方互換のためreaderを残しています |

二盤面の自己対局は相手側のevent streamを観測に含むため、compact形式のmetadataだけでは正確に再構成できません。この経路ではrectangular形式を使います（[ADR 0012](adr/0012-compact-dataset-replay-pi.md)）。

### Reanalyse

`trainer/reanalyze.py` と `include/tetra/reanalyse.hpp` は、seedから各ゲームを再生し、記録された `chosen_action` を順に適用します。再構成したtokenと合法手埋め込みが保存済みの行と一致したdatasetだけを対象に、KL divergenceで選んだrootを深い探索で再評価し、更新した方策を別shardへ出力します（[ADR 0015（Reanalyse）](adr/0015-reanalyse-historical-target-refresh.md)）。

### 人間リプレイ

人間のリプレイからの事前学習では、Pythonがリプレイを正規化してTETR.IO v19の状態を再構成し、C++が合法手生成で各配置を検証してからdatasetを書き出します。手順は[人間リプレイによる事前学習](HUMAN_REPLAY_PRETRAINING.md)を参照してください。

### Colab

Colabは追加のworkerであり、ルール、label、seed割り当ての権威にはしません。ruleset、schema、checkpointの契約が異なるshardを暗黙に混ぜません。手順は[Colab手順](COLAB_MANUAL.md)を参照してください。

## 6. 学習目的

目的を混同しないよう、層を分けます。

- **方策:** 探索で改善されたaction分布を学びます。
- **価値:** 固定したplayer視点からWDLを予測します。
- **補助head:** 将来のattack、受けるgarbage、top outまでのhorizonなど、同じtrajectoryから得られる密な教師信号を学びます。現行のaux schema v4は52 targetsで、未知の未来はvalid maskで除外します。
- **終端目的:** 実際の勝敗を維持します。

VS Score、APM、APP、PPS、相殺などは測定値です。WDLやArenaによるpromotionを置き換えません。詳細は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md#2-目的関数)と[ADR 0014（目的と補助目標）](adr/0014-objectives-auxiliary-targets-and-vs-score.md)を参照してください。

## 7. Timingの表現と学習

合法手生成にはdelay actionと `WAIT_FOR_EVENT` があります。ただし、action空間に存在することは、networkがtimingを学習したことを意味しません。GPU自己対局ではtiming actionは既定で無効で、`--timing-actions` で明示的に有効にします。

timingを段階的に導入する条件は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md#8-timingと相殺外しのカリキュラム)で管理します。

## 8. 将来の研究

強く指すplaying agentと、人間に戦略を説明するteaching agentを分ける構想、自然言語を媒介層とする設計、Sparse MoEとSparse Autoencoderの扱いは、[ロードマップ](ROADMAP.md#その後の構想)と[ADR 0016](adr/0016-defer-sparse-moe-and-build-for-interpretability.md)で管理します。現在のproduction pathには含まれていません。
