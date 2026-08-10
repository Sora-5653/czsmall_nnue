# ロードマップ

この文書は、プロジェクトの**現在の実行状況**を示します。`SPEC.md` は当初仕様として保存し、後の方向変更は `adr/` に記録します。

## Milestone状況

| 領域 | 状態 |
|---|---|
| **M0 — rule core** | **完了。** Board、piece、SRS/SRS+/180 kick、spin、clear、attack、garbage、ruleset versioning、event logを実装済み。 |
| **M1 — policy input / move generation** | **現行contractでは完了。** Cobra legal placement、action timing/delay bin、masked observation、row/column/global token、bag/opponent-counter token、schema identifier、variable legal-action embeddingを実装済み。 |
| **M2 — search / self-play / training** | **end-to-end loop実装済み。** batched evaluator、PUCT/Gumbel、determinization、replay/dataset、PyTorch training、C++ weight inference、GPU self-play、GPU Arena、resumable checkpoint、replay mixing、Candidate→Champion guarded iterationが利用可能。 |
| **M3 — garbage-aware self-play** | **有効。** 二盤面をtimestamp順に進め、search/Arenaと同じevent machineryでattack deliveryを行う。no-attack curriculumも明示的に選択可能。 |
| **M4 — opponent-aware model** | **core observation pathは有効。** Opponent board/counterをtokenizeし、two-player searchも実装済み。Dedicated opponent-intent modellingとleague trainingは未実装。 |
| **M5 — multimodal** | **未着手。** image/video inputは現在の優先順位外。 |

## 現在のモデル構造

TetraFormerをreference/controlとして維持し、CNNとCNN+Transformerを同じsimulator・tensor contract上の実験候補として扱います。

2026-08-08のablationから、architecture選定方針を次のように変更しました。

- TransformerはTransformer由来teacher policyをわずかに良く模倣した。
- corrected hashed splitではfull CNNがWDL/valueを大幅に良く学習した。
- CNNの最も安定した優位はraw policy-onlyではなく**search内**に現れた。
- 小型CNNを付け足したhybridはその優位を再現できず、value late-overfitも起こした。

したがって次のarchitecture実験は「TransformerをCNNで置換する」でも「CNN headをまた足す」でもありません。

**具体的な表現仮説**を検証します。最有力候補は、full CNN相当のlocal encoderをpolicy/valueで共有し、その特徴をTransformerのglobal/opponent interactionへ接続するhybridです。

関連: [ADR 0013](adr/0013-architecture-ablation-and-local-geometry.md)、[CNN_ABLATION_20260808.md](CNN_ABLATION_20260808.md)。

## 優先度1 — サンプル効率と目的関数の診断

現在 `main` には次が存在します。

- tokenizer / observation / action / auxiliary schema identifier
- dataset contract上の `terminated` / `truncated` 区別
- bag tokenとopponent-counter token
- 36 auxiliary targets
  - legacy 4 targets
  - real-time 4区間 × attack / garbage received / self top-out / opponent top-out
  - placement 4区間 × 同じ4 channel
- unknown future horizon用valid mask
- trainer-side auxiliary target statistics
- shared-trunk gradient norm
- policy/value、policy/aux gradient cosine diagnostics

次の実行順:

1. multi-horizon targetをより大きなmixed-generation datasetで再検証し、split leakage checkを固定する。
2. policy-only / WDL / legacy aux / multi-horizon auxを、同一dataset・split・training budget・複数seedで比較する。
3. **VS Scoreをmatch/Arena reportへ実装する。** 計算式とruleset interpretationを文書・testで固定する。
4. VS Scoreをauxiliary prediction targetとして試す場合は、report実装とは別のablationとして行う。WDL rewardは変更しない。
5. action-conditioned consequence targetは、engineからexact labelを得られ、入力特徴の自己コピーにならないものから追加する。

関連: [ADR 0014](adr/0014-objectives-auxiliary-targets-and-vs-score.md)、[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)、[SAMPLE_EFFICIENCY_PLAN.md](SAMPLE_EFFICIENCY_PLAN.md)。

## 優先度2 — provenanceを保った自己対局loopの運用

local/Colab generationとmanifest validatorは実装済みです。次の課題は「sampleを増やせること」から「generation間を正しく比較できること」へ移っています。

1. すべてのshardをcommit、checkpoint、ruleset、schema、search setting、non-overlapping seed intervalへ結び付ける。
2. local-only generationとlocal+Colab generationを、同じArena protocolで比較してからpromotionする。
3. generationが安定したら、**浅いsearch中心 + 少量の深いsearch**というmixtureを試す。
4. position-start / recovery curriculumを導入する場合は、manifest上で別provenance classとして記録する。
5. Championはconfigured Arena gate以外から変更しない。

Drive/GASのresumable transferは運用上有用ですが、seed・label・dataset mergeの権威にはしません。

関連: [ADR 0015](adr/0015-selfplay-provenance-search-mixture-and-timing-curriculum.md)。

## 優先度3 — 基本戦術の後にtiming / 相殺外し

move generatorは `WAIT_FOR_EVENT` を含むdelay binをすでに表現できます。しかし初期trained policyはdelay actionをほぼ利用しておらず、**表現可能であることと学習済みであることは別**です。

段階的に進めます。

1. stable stacking、Quad、T-spin、通常attack constructionが明確に成立する状態を先に作る。
2. Arena、qualitative play、APP、将来のVS Scoreを合わせてreadinessを見る。
3. 平積みQuadの理論baselineであるAPP約0.5は粗いdiagnosticとしてのみ使い、hard gateにはしない。
4. その後、delay action・garbage timing・相殺外しのexploration/data coverageを強める。
5. delayed-action frequency、`WAIT_FOR_EVENT` usage、cancellation interaction、VS Score、paired winsを測り、能力の実在を確認する。

「waiting」自体へpositive rewardを与えません。delayの有用性はsearchとgame outcomeから学習させます。

## その後 — 強いplaying agent、teaching agent、自然言語媒介

長期目標は、特定のinterpretability architectureそのものではなく、**AI→人間学習pipeline**です。

役割を分離します。

- 最大限強く指す**playing agent**
- 強いagentのknowledgeを抽出・検証・説明する**teaching/analysis agent**

teaching agentはplaying agentと同一architectureである必要はありません。trace、search statistics、activation、learned feature、counterfactual probeなどを利用し、人間にとって有用な戦略概念へ変換できればよいとします。

自然言語は双方向の媒介層として使います。

- 人間の問い → reproducible game/model probe
- engine/model discovery → human-readable explanation / curriculum

exact simulator・intervention・provenanceを根拠として残し、fluentな説明だけで戦略を正しいとみなしません。

### Sparse MoE / Sparse Autoencoder

これらは候補手法であり、長期目標そのものではありません。

Sparse MoEはstrong dense baselineができるまで延期します。将来試す場合はsmall expert count、shared trunk、observable routingから始めます。

Sparse Autoencoderもstrong checkpoint後のfeature extraction候補です。candidate featureをteachingへ使う前に、counterexample・intervention・ablationで検証します。

関連: [ADR 0016](adr/0016-defer-sparse-moe-and-build-for-interpretability.md)。

## Tooling / CIの既知課題

### Vendored Cobraと `-Werror`

通常の `make test` はC++23で通りますが、`docs/ci.yml` のwarning-as-error条件をそのまま適用すると、vendored Cobra内部の `#pragma unroll` と `-Wshadow` warningがerrorへ昇格して失敗します。

これは現在のproject-side sourceのcorrectness failureではなく、third-party codeを同じwarning policyでcompileしていることによるscope問題です。

次に決めるべきなのは、Cobra sourceを無条件に書き換えることではなく、次のいずれかです。

- project codeとvendored codeでwarning policyを分離する。
- upstream-compatibleな最小patchを用意する。
- compilerごとのpragma/warning扱いをCI側で明示する。

この問題を解消するまでは `docs/ci.yml` を「そのままgreenになる完成workflow」とは扱いません。

## 残っているengine correctness課題

### Lock delayと `reset_limit`

Gravityはreachability constraintとして実装済みですが、stackへ接触した後のfull lock-delay manoeuvring windowとbounded resetは未完成です。

高gravityではこのwindowが主要な操作時間になるため、現在のmodelは一部の本来到達可能なplacementを保守的にrejectする可能性があります。

### TETR.IO parityに関する未解決事項

- **Kick-table provenance:** SRS+ / 180 tableはstructural test済みだが、合法的に取得したreal TETR.IO replayとのdiff testは未実施。
- **High B2B × high combo rounding:** documented formulaに従うが、実ゲームの非整数intermediateと極端な組み合わせで差が出る可能性がある。
- **Garbage messiness constants:** behaviourはconfigurableだがreal constantsは非公開。
- **Surge variants:** reversed / QUICK PLAY系を完全にはmodelしていない。

これらはrule parityの問題であり、deterministic ruleset/hash/schema contractを緩める理由にはしません。

## ドキュメント更新規則

このROADMAPへ合わせるために `SPEC.md` を書き換えません。

方向が変わった場合は次の順で更新します。

1. ADRを追加する。
2. ROADMAPを更新する。
3. 必要ならoperational guideを更新する。
4. `SPEC.md` は当初仕様のhistorical baselineとして残す。

文書間の役割分担は [docs/README.md](README.md) を参照してください。
