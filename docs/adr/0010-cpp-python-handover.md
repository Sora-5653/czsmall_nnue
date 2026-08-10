# ADR 0010: C++ / Python境界とfixed-shape batching

## 状態

採用。

## 背景

search、self-play loop、replay bufferができた後、Transformerを接続する前に1つ大きな構造問題が残っていました。

**上流データはraggedだが、network backendはrectangular batchを必要とする。**

当時720 self-play samplesを測定すると:

| | min | max | mean |
|---|---:|---:|---:|
| state tokens | 48 | 65 | 63.6 |
| legal actions | 9 | 70 | 42.5 |

どこかでpaddingとmaskを行う必要があります。問題は「どこをcontract boundaryにするか」でした。

## 決定1: engine側で1度だけpad / maskする

`batch.hpp` の `TensorBatch` がrectangular formを生成し、同じstructureを次のconsumerで共有します。

1. C++ inference
2. on-disk training dataset
3. PyTorch trainer

backendごとに別々のpadding ruleを実装しない理由:

- padding conventionを1か所で定義・testできる。
- maskをdataと同時生成するため、backendがpaddingを誤ってattention対象にするfailureを減らせる。
- trainerとengineのshape / feature order driftを防げる。

layoutはrow-major contiguousです。

```text
tokens  [B, T, TOKEN_FEATURES]   token_mask  [B, T]
actions [B, A, ACTION_FEATURES]  action_mask [B, A]
```

`pad_tokens` / `pad_actions` を指定するとfixed-shape backend用のstatic dimensionを強制できます。0ならbatch内容に合わせて必要なsizeだけ使います。

## 決定2: `.tetradat` をC++/Python共通contractにする

初期versionでは、small self-describing headerの後ろにbatch bufferをflat float32で保存しました。

Protobufを採用しなかった理由はADR 0005と同じです。当時のenvironmentではtoolchainを取得できず、必要のないcross-language dependencyを増やすより、simple binary contractを優先しました。

Python reader `trainer/tetra_dataset.py` はload時に次を検証します。

- non-finite value
- non-zero padding
- policy rowがdistributionになっているか
- padded actionへpolicy massが乗っていないか
- feature/schema compatibility

当初headerは `token_features` / `action_features` を記録しました。後のsample-efficiency workでは、**widthが同じでも意味が違う場合**を検出するため、tokenizer/observation/action/aux schema version/hashも追加されています。

## この境界が発見したbug

最初のend-to-end trainingでは、policy lossが約3なのにauxiliary lossが**約150**になりました。

原因は `time_to_terminal` を数百placementに達するraw countのままMSEへ入れていたことです。auxiliary lossがtotal lossを支配し、modelがほぼ他のtaskを学べない状態でした。

そのためcount系aux targetをengineから出す前にbounded rangeへsquashし、loss weightが意味を持つscaleへ揃えました。`aux_targets_are_normalised` で固定しています。

## 当時の結果

初期chainは次でend-to-endに動きました。

```sh
./build/tetra_cli export train.tetradat 10 100 16
python trainer/train.py train.tetradat --steps 300
```

`trainer/tetraformer.py` はspec §9–10のnetworkを実装しました。

- pre-norm RMSNorm block
- SwiGLU
- variable-length policy head
- legal action queryからstate tokenへのcross-attention
- WDL value head
- auxiliary regression

当時の `tetraformer_s()` は約7.2M parameters、`tetraformer_dev()` は約0.13Mでした。

engine-generated dataでheld-out total lossが300 stepsで **4.86 → 2.91** へ低下し、padded actionのprobabilityがexactly 0であることも確認しました。

## 当時このADRが主張していなかったこと

この段階のmodelは「強い」とは主張していませんでした。少数のheuristic-guided samplesをCPUで学習し、C++↔Python handoverがcorrectでlossが減ることを確認しただけです。

原文では「Arena/gatingはM2にまだない」と記録していましたが、これは**当時の状態**です。現在はGPU Arena、Candidate/Champion gating、guarded iterationまで実装済みです。

## 後続の形式変更

ADR 0012で、再構成可能なsingle-board sampleにはcompact Replay+π version 2を追加しました。

さらに現行 `DatasetHeader::VERSION = 3` では、rectangular datasetへschema・termination metadataを追加しています。two-board self-playはopponent event streamをcompact v2 metadataだけで再構成できないためversion 3を標準に使います。

つまり本ADRの「1つのtensor contractを共有する」という判断は維持しつつ、**disk representation自体はversioned evolutionしている**と理解してください。

## 2026-08の追補

### アーキテクチャ比較実験

fixed tensor contractはTetraFormer以外にも有効でした。CNN / CNN+Transformer ablationも同じpublic input/output interfaceを再利用できるため、simulatorやdataset semanticsを変えずにarchitectureを比較できます。

[ADR 0013](0013-architecture-ablation-and-local-geometry.md) は、これらのmodelをheld-out policy loss単独ではなくsearch/Arena evidenceで選ぶ方針を記録します。

### Sample efficiency

ここで始めたauxiliary regressionは、後にmulti-horizon sample-efficiency programへ発展しました。

[ADR 0014](0014-objectives-auxiliary-targets-and-vs-score.md) はWDLをobjective anchorに残しながら、valid mask付きmulti-horizon target、将来のaction-conditioned target、gradient interference monitoringを許容します。
