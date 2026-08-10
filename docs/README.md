# ドキュメント案内

このディレクトリでは、**当初仕様・設計判断の履歴・現在の実装状況・実験結果・運用手順**を意図的に分離しています。開発速度が速いため、これらを1つの文書へ混ぜると「当初からそう設計されていたこと」と「後から実験で変わったこと」が区別できなくなるためです。

## 文書の権威と更新規則

質問の種類に応じて、次の順で参照先を使い分けます。

1. **`SPEC.md` — 当初の設計仕様。** プロジェクト開始時の設計基準を保存する歴史的文書です。後の判断を、最初から書かれていたように見せるための書き換えは行いません。
2. **`adr/` — 仕様策定後の設計判断。** 採用・却下・延期・置換の理由を記録します。`SPEC.md` から方向を変えた場合、その変更理由の権威はADRです。
3. **コードとテスト — 実装済み挙動。** ADRが将来方針を記録している場合もあります。「実装済み」と断定できるのはコードとテストで確認できる事項だけです。
4. **`ROADMAP.md` — 現在地と次の実行順。** `main` に存在するもの、評価中のもの、意図的に延期しているものを追跡します。
5. **実験レポート — 判断材料。** `CNN_ABLATION_20260808.md` のような文書は測定結果や失敗仮説を保存します。実験そのものは方針ではなく、方針変更に至った場合にADRから引用します。
6. **運用ガイド — 実行方法。** `SETUP.md`、`COLAB_MANUAL.md`、リポジトリ直下の `AGENTS.md` がbuild、GPU学習、自己対局、shard運用を扱います。
7. **`POLICY.md` — 利用範囲。** 本プロジェクトはローカルシミュレータであり、TETR.IOへ接続しません。

新しい構造上の判断を行う場合、古いADRや `SPEC.md` を現在形へ書き換えるより、**新しいADRを追加してROADMAPを更新する**ことを優先します。既存ADRへ追記する場合は、過去の判断を消さず「追補」として新しいADRへのリンクを追加します。

## 現在のシステム構成

責務は大きく2層へ分けています。

- **C++がゲーム上の真実を所有する:** ruleset、Cobra合法手生成、timing、不可視情報のmask、探索、自己対局の状態遷移、dataset serialization、replay verification、Arenaのゲーム進行。
- **Python/PyTorchが学習を所有する:** neural evaluator、GPU batch、学習、checkpoint、モデル構造ablation、ROCm/CUDA inference worker。
- **境界をテストする:** 学習と推論が同じtoken/action contractを使い、C++ forwardはPyTorch forwardとのparity testで固定します。

基準モデルは現在もTetraFormerです。CNNおよびCNN+Transformerは、既存モデルを黙って置き換えるものではなく、共通契約上の実験候補です。2026-08-08のablationでは、**方策cross-entropy、WDL学習、探索下の強さが一致しない**ことが確認されたため、モデル構造の採否をheld-out loss 1つだけで決めない方針になりました。詳細はADR 0013を参照してください。

## 学習・評価の原則

現在の学習方針では、目的を次のように分離します。

- 対局結果のWDLを価値・終端目的の基準に残し、戦闘統計で置換しない。
- trajectoryから得られる密な補助目標は、表現学習とサンプル効率を改善するために使う。ただし方策学習を悪化させるなら採用しない。
- CandidateからChampionへの昇格条件は、固定条件のpaired Arena勝率と信頼区間を中心に置く。
- VS Scoreは、計算式・ruleset解釈を固定して実装した後、APM/APP/PPS、生存、相殺、raw-policy/search-policy比較と並ぶ**戦闘診断指標**として追加する。勝率の代替にはしない。
- モデル構造や補助headの効果は、同一データ・同一計算budget・複数seedのablationで確認する。

詳細は `TRAINING_AND_EVALUATION.md`、ADR 0014、ADR 0015を参照してください。

## 用語方針

文書では、コード識別子を除き、可能な範囲で日本語を優先します。

| 英語の概念 | 本文での基本表記 |
|---|---|
| policy | 方策 |
| value | 価値 |
| self-play | 自己対局 |
| held-out / validation | 検証用 / 検証集合 |
| auxiliary target | 補助目標 |
| loss | 損失 |
| checkpoint | チェックポイント |
| dataset | データセット |
| promotion | 昇格 |
| inference | 推論 |
| routing | ルーティング |

`Candidate`、`Champion`、`Arena`、`TetraFormer`、`Cobra`、CLI optionやクラス名など、コード上の固有名はそのまま表記します。

## ADR索引

[`adr/README.md`](adr/README.md) に全ADRの一覧があります。2026年8月に追加された主要判断は次のとおりです。

- **ADR 0013:** CNN/Transformer系の構造比較は、教師模倣lossだけでなく探索下Arenaの強さを必須証拠とする。TetraFormerは比較基準として残す。
- **ADR 0014:** WDLを目的の基準に保ち、multi-horizon補助目標やVS Scoreは補助教師・診断として扱う。
- **ADR 0015:** 自己対局データのprovenanceを固定し、浅い探索中心＋少量の深い探索、timing/相殺外しの段階導入を明示する。
- **ADR 0016:** 長期のAI→人間学習構想では、「強く指すagent」と「教えるagent」を分け、自然言語を人間の問いと再現可能なgame/model probeの双方向媒介層として扱う。MoE/SAEは候補手段であり、現時点の必須構造ではない。

## 目的別の参照先

| 知りたいこと | 文書 |
|---|---|
| 当初想定した最終設計 | `SPEC.md` |
| なぜ設計が変わったか | `adr/README.md` と各ADR |
| 現在のアーキテクチャ | `ARCHITECTURE.md` |
| 学習・Arena比較の規約 | `TRAINING_AND_EVALUATION.md` |
| 現在の優先順位 | `ROADMAP.md` |
| サンプル効率の実装詳細 | `SAMPLE_EFFICIENCY_PLAN.md` |
| CNN/CNN+Transformerの測定結果 | `CNN_ABLATION_20260808.md` |
| GPU・ローカル環境 | `SETUP.md`、直下 `AGENTS.md` |
| Colab shard生成 | `COLAB_MANUAL.md` |
| 利用範囲 | `POLICY.md` |

## 日本語化の対象外

- `SPEC.md` は当初仕様を保存するため原文のまま維持します。
- `third_party/**/LICENSE` などの正式なlicense textは、法的意味を翻訳で変えないため原文を維持します。`THIRD_PARTY_NOTICES.md` の説明文自体は日本語化しています。
- コード、CLI option、file format field、class/function名は翻訳しません。
