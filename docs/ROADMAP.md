# ロードマップ

プロジェクトの現在の実装状況と、次に進める作業の順序を示します。当初仕様は[SPEC](SPEC.md)に保存し、方向の変更は[ADR](adr/README.md)に記録します。比較実験の規則は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md)で管理します。

Championと個別実験の最新状況は、各実験のhandoffで管理します。直近の記録は[2026-08-15 handoff](HANDOFF_STACKING_LC3_20260815.md)です。

## Milestoneの状況

| 領域 | 状態 |
|---|---|
| **M0 — rule core** | 完了。盤面、piece、SRS/SRS+/180 kick、spin、line clear、attack、garbage、ruleset versioning、event logを実装済み。spawn位置はrulesetの設定で、TETR.IO v19とguidelineを区別します。 |
| **M1 — 方策入力と合法手生成** | 現行の契約では完了。Cobraによる合法配置生成、action timingとdelay bin、mask済み観測、row/column/global token、bagと相手カウンタのtoken、schema識別子、可変長の合法action埋め込みを実装済み。 |
| **M2 — 探索、自己対局、学習** | end-to-endのloopを実装済み。batch評価、PUCT/Gumbel、determinization、dataset、PyTorch学習、C++推論、GPU自己対局、GPU Arena、再開可能なcheckpoint、replay混合、guard付きiteration、LC3型の探索runtime、Reanalyseを利用できます。 |
| **M3 — garbageを考慮した自己対局** | 有効。二盤面をtimestamp順に進め、探索およびArenaと同じevent処理でattackを届けます。attackを届けない条件（`--no-attack-delivery`）も明示的に選べます。 |
| **M4 — 相手を考慮したモデル** | 観測の基本経路は有効。相手の盤面とカウンタをtoken化し、二人用の探索も実装済み。相手の意図のモデル化とleague学習は未実装。 |
| **M5 — マルチモーダル** | 未着手。画像・動画入力は現在の優先順位外です。 |

## 2026-08-10以降に入った主な実装

| 項目 | 内容 | 根拠 |
|---|---|---|
| LC3型の探索runtime | 共有の推論queue、`SearchPolicy` の分離、edgeごとの統計、Gather/Eval/Backpropの分離とtelemetry | [ADR 0013（LC3）](adr/0013-lc3-search-runtime.md) |
| モデルサイズと探索予算 | XS/Sの比較。XSを高速探索の比較基準として維持 | [ADR 0014（サイズと探索予算）](adr/0014-model-size-vs-search-budget.md) |
| Reanalyse（最小構成） | 履歴の厳密な再生、token/actionの一致確認、KLによる選択、選んだrootの再探索、非破壊の出力 | [ADR 0015（Reanalyse）](adr/0015-reanalyse-historical-target-refresh.md) |
| source別のsampling | `trainer/train.py` の `--secondary-source-*` で、別sourceを固定比率で混ぜる | [2026-08-15 handoff](HANDOFF_STACKING_LC3_20260815.md#5-source-aware-sampling-is-already-implemented) |
| VS Score | GPU ArenaでCandidateとChampionのVS Scoreを報告 | `include/tetra/stats.hpp` |
| aux schema v4 | 52 targets。garbageの消去と相殺の区間targetを追加 | `include/tetra/schema.hpp` |
| dataset v4 | Reanalyse用に `chosen_action` などの再生情報を保持 | [アーキテクチャ](ARCHITECTURE.md#dataset-version) |
| 人間リプレイの事前学習 | TETR.IO v19の厳密な再構成、C++による配置検証、`teacher1m` からXSへの蒸留 | [人間リプレイによる事前学習](HUMAN_REPLAY_PRETRAINING.md) |

## 優先度1 — サンプル効率と教師targetの質

2026-08-15時点の次の作業は、元のtargetとReanalyse後のtargetを比較する管理された学習ablationです。構造は固定したまま行います。

1. 元のtargetとReanalyse後のtargetを、同じdataset、分割、学習予算、複数seedで比較します。
2. 方策のみ、WDL、legacy aux、区間auxを、同じ条件で比較します。
3. VS Scoreを補助目標として試す場合は、Arena reportとは別のablationとして行います。WDL rewardは変更しません。
4. action-conditionedな結果targetは、エンジンから厳密なlabelを得られ、入力特徴の自己コピーにならないものから追加します。

関連: [ADR 0014（目的と補助目標）](adr/0014-objectives-auxiliary-targets-and-vs-score.md)、[サンプル効率の実装計画](SAMPLE_EFFICIENCY_PLAN.md)。

## 優先度2 — 由来を保った自己対局loopの運用

ローカルとColabでの生成、manifestのvalidator、source別のsamplingは実装済みです。課題は、generation間を正しく比較できる状態を保つことです。

1. すべてのshardを、commit、checkpoint、ruleset、schema、探索設定、重複しないseed区間へ結び付けます。
2. ローカルだけの生成とローカル+Colabの生成を、同じArena条件で比較してからpromotionします。
3. 自己対局が安定したら、浅い探索を中心に少量の深い探索を混ぜます。
4. Championは設定済みのArena gate以外では変更しません。

関連: [ADR 0015（自己対局の由来）](adr/0015-selfplay-provenance-search-mixture-and-timing-curriculum.md)。

## 優先度3 — 基本戦術の後にtimingと相殺外し

timing actionは現在無効です。固定したclean benchmarkの平均APPが0.5を超えてから、timingのablationを再開します。段階と条件は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md#8-timingと相殺外しのカリキュラム)を参照してください。

## その後の構想

長期目標は、特定のinterpretability構造ではなく、AIから人間への学習の流れを作ることです。

- 最大限強く指す**playing agent**と、その知識を抽出・検証・説明する**teaching agent**を分けます。teaching agentは、trace、探索統計、activation、counterfactual probeなどを使えれば、playing agentと同じ構造である必要はありません。
- 自然言語は、人間の問いを再現可能なgame/model probeへ落とし、発見を検証可能な説明へ戻す双方向の媒介層として使います。根拠は厳密なシミュレータ、intervention、由来に置きます。
- Sparse MoEとSparse Autoencoderは候補手法であり、目標そのものではありません。強いdense baselineができるまで延期します。

関連: [ADR 0016](adr/0016-defer-sparse-moe-and-build-for-interpretability.md)、[ADR 0014（サイズと探索予算）](adr/0014-model-size-vs-search-budget.md)。

## ToolingとCIの既知の課題

### vendored Cobraと `-Werror`

通常の `make test` はC++23で成功します。一方、`docs/ci.yml` のwarning-as-error条件をそのまま適用すると、vendored Cobra内部の `#pragma unroll` と `-Wshadow` のwarningがerrorになり失敗します。

これはproject側のcorrectnessの失敗ではなく、third-partyのコードを同じwarning policyでcompileしている範囲の問題です。Cobraのsourceを無条件に書き換えず、次のいずれかを決めます。

- project側とvendored側でwarning policyを分ける。
- upstreamと互換な最小patchを用意する。
- compilerごとのpragmaとwarningの扱いをCI側で明示する。

解消するまで、`docs/ci.yml` はそのままgreenになるworkflowとして扱いません。

### Colab validatorのdataset version

`trainer/colab_generate.py` のvalidatorはdataset v1/v3だけを受け付け、現在のv4を拒否します。[Colab手順](COLAB_MANUAL.md)を参照してください。

## エンジンのcorrectnessの未解決事項

### lock delayと `reset_limit`

gravityはreachabilityの制約として実装済みです。一方、stackに接触した後のlock delayの操作猶予と、`reset_limit` による回数制限は合法手生成に反映されていません。

高gravityではこの猶予が主な操作時間になるため、本来到達可能な一部の配置を保守的に除外する可能性があります。

### TETR.IOとの一致

- **kick tableの出典:** SRS+と180のtableは構造テスト済みです。合法に取得した実際のTETR.IO replayとのkick単位の差分テストは未実施です。
- **高B2Bと高comboの丸め:** 公開されている式に従っていますが、非整数の中間値と極端な組み合わせで差が出る可能性があります。
- **garbageの乱れ方の定数:** 動作は設定可能ですが、実際の定数は非公開です。
- **Surgeの派生:** reversedやQUICK PLAY系を完全にはモデル化していません。

これらはルールの一致の問題であり、決定論的なruleset、hash、schemaの契約を緩める理由にはしません。

## 文書の更新規則

方向が変わった場合は、次の順で更新します。

1. ADRを追加します。
2. このROADMAPを更新します。
3. 必要なら運用ガイドを更新します。

[SPEC](SPEC.md)は当初仕様として残し、このROADMAPに合わせて書き換えません。文書間の役割分担は[文書案内](README.md)を参照してください。
