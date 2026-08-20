# 現在のアーキテクチャ

この文書は、現在の `main` に存在する構成と、production code・実験系の境界を説明します。`SPEC.md` の置き換えではありません。仕様策定後の設計変更はADRに記録します。

## 1. シミュレーションとルール: C++を権威とする

ゲーム上の真実へ影響する状態遷移はC++エンジンが所有します。

- ruleset設定とhash
- 盤面、piece、spin、attack、garbage、B2B/combo/Surge
- Cobraによる合法配置生成
- action duration、gravity reachability、delay bin
- observation maskingとtoken/action生成
- PUCT/Gumbel探索、determinization、二盤面event ordering
- replay verification、dataset serialization、Arena game mechanics

学習側が別の近似ゲームを再実装することは避けます。これにより、学習labelと推論時の意味論を1つのシミュレータへ固定します。

## 2. 観測・action contract

エンジンは、mask済みのplayer observationを可変長のstate sequenceとlegal-action sequenceへ変換します。`TensorBatch` はそれらをpaddingし、同時にmaskを保持します。同一contractをC++ inference、PyTorch、dataset readerが共有します。

観測にはpublic informationだけを含めます。hidden queueや未着弾garbageのhidden情報はtokenや探索へ漏らしません。

現行contractはfeature widthだけでは識別しません。`schema.hpp` がtokenizer、observation、action、auxiliary targetのversion/hashを定義し、同じ `TOKEN_FEATURES=24` でも意味やtoken順が違うdatasetを互換扱いしないようにします。

関連: ADR 0009、ADR 0010、ADR 0012、`SAMPLE_EFFICIENCY_PLAN.md`。

## 3. Neural evaluator

### TetraFormer — 基準モデル

`trainer/tetraformer.py` が現在の基準architectureです。Transformer state encoder、可変長action scoring、WDL value head、auxiliary prediction headを持ちます。

新しい構造が短いArenaで勝っただけではTetraFormerを捨てません。比較のcontrolとして残し、同じdataset・split・optimizer・search budgetで候補構造を評価します。

### CNN / hybrid — 実験モデル

`trainer/ablation_models.py` には、同じpublic forward contractを維持するlocal-board CNNとCNN+Transformer系を実装しています。目的は、シミュレータやdataset契約を変えずに、明示的な2次元局所帰納バイアスがゲーム強度へ寄与するかを切り分けることです。

2026-08-08のablationでは、次の分離が見えました。

- Transformerは、Transformer-search由来のteacher policyをわずかに良く模倣した。
- 修正済みhashed splitでは、CNNがWDL/valueを大幅に良く学習した。
- CNNの最も再現性の高い優位は、raw policy-onlyより**search内で使ったとき**に現れた。
- 小さなbolt-on CNN branchを持つhybridはfull CNNの性質を再現できず、valueのlate overfitも起こした。

そのため、次のhybridは「CNNを少し足す」のではなく、**何の表現仮説を検証するのか**を明示する必要があります。候補は、full CNN相当の局所encoderをpolicy/valueで共有し、Transformerを長距離・global・opponent interactionへ使う構成です。

関連: ADR 0013、`CNN_ABLATION_20260808.md`。

## 4. 探索

Evaluatorはpolicy/value estimateを返し、C++ searchがそれを改善されたroot policyへ変換します。

低simulation数では、合法手数に対してPUCTの探索が薄くなりやすいため、Gumbel sequential halvingを既定としています。virtual lossとtransposition tableもC++側にあります。

不可視なfuture pieceはroot determinizationで扱います。したがって自己対局教師は、人間playerが観測可能な情報の範囲で生成されます。

## 5. 自己対局とdataset

基本generation loopは次のとおりです。

1. checkpointと、repository/ruleset/schema provenanceを固定する。
2. search設定とseed範囲を記録して自己対局を生成する。
3. shardとmanifestをimmutable artifactとして保存する。
4. 管理されたreplay mixtureで学習する。
5. paired seedのArenaでCandidateとChampionを比較する。
6. promotion gateを満たした場合だけChampionを更新する。

Colabは追加workerであり、rules、label、seed allocationの第二の権威ではありません。ruleset/schema/checkpoint contractが異なるshardを暗黙に混ぜません。

### dataset version

- **Version 3:** 現行のrectangular dataset。schema/termination metadataを保持し、二盤面自己対局で使用します。
- **Version 2 (`VERSION_COMPACT`):** Replay+πを保存し、再生可能な単盤面trajectoryからtoken/actionをload時に再生成するcompact形式です。
- **Version 1:** 旧rectangular形式。readerは後方互換のため残しています。

二盤面self-playは相手側event streamを観測へ含むため、現在のcompact metadataだけでは正確に再構成できません。この経路を無理にcompact化せずversion 3を使う判断は意図的です。

関連: ADR 0012、ADR 0015、`COLAB_MANUAL.md`。

## 6. 学習目的

目的を混同しないよう、階層を分けます。

- **方策:** search-improved action distributionを学ぶ。
- **価値:** 固定player perspectiveからWDLを予測する。
- **補助head:** future attack、future garbage、survival/top-out horizonなど、同一trajectoryから得られる密な教師信号を学ぶ。
- **終端目的:** 実際のgame resultを維持する。

VS Score、APM、APP、PPS、cancellationなどは測定値です。将来、ablationを経てauxiliary prediction targetへ使う可能性はありますが、WDLやArena promotionを置き換えません。

現行aux schema v2は36 targetsを持ち、未知未来にはvalid maskを使います。trainerはshared trunk上のpolicy/value/aux gradient normとcosineも記録できます。

関連: ADR 0014。

## 7. Timingを「表現可能」と「学習済み」に分ける

move generatorにはdelay actionと `WAIT_FOR_EVENT` がすでにあります。しかしaction spaceに存在することは、networkがtimingを学習したことを意味しません。

初期の学習済みpolicyでは最速actionへ偏る傾向が見られたため、garbage timingや相殺外しは、基本stacking・Quad・T-spinなどが成立した後に段階的に強める能力として扱います。

timing能力は、delay action頻度だけでなく、cancellation interaction、VS Score、paired Arena resultと合わせて検証します。

関連: ADR 0015。

## 8. 将来研究: 強いplayer、teaching agent、自然言語媒介

長期のAI→人間学習構想では、同じモデルへすべてを押し込みません。

- **playing agent:** Arenaで最大限強く指すことを主目的とする。
- **teaching/analysis agent:** 強いagentのtrace、search、activation、featureなどを使い、人間に役立つ戦略概念を抽出・検証・説明することを主目的とする。

teaching agentの質はArena strengthだけでは測りません。counterfactual questionへ答えられるか、概念を再現可能な局面集合へ落とせるか、不確実性を表現できるか、人間の学習に実際に役立つか、といった別の評価が必要です。

自然言語は**双方向の媒介層**として位置づけます。

- human → engine/model: 「このstackでなぜこの手が悪いのか」といった問いを、position slice、search probe、intervention、comparisonへ落とす。
- engine/model → human: discovered regularity、feature、routing pattern、counterfactual resultを、検証可能な説明や教材へ変換する。

ゲーム上の事実の権威は引き続きexact simulatorとprovenanceに置きます。言語modelは説明interface・仮説生成器であって、rule engineの代替ではありません。

### MoE / Sparse Autoencoder

Sparse MoEとSparse Autoencoderは、この構想の**候補手法**であり必須構造ではありません。

MoEは強いdense baselineができるまで延期します。最初に試す場合は、小数expert・shared trunk・observable routingから始め、opening/midgame/finisher、高stack pressureなどのhuman labelをsemantic guaranteeとみなしません。

Sparse Autoencoderも強いcheckpointができた後にactivation feature抽出へ試します。featureを人間概念と結びつける際は、強くactivateする例だけでなくcounterexample、intervention、ablationで検証してからteaching layerへ渡します。

関連: ADR 0016。
