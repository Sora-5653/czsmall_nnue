# ADR 0013: Architecture変更はgame strengthで判断し、local geometry実験は段階的に進める

## 状態

採用。

## 背景

当初のnetwork pathはTetraFormerです。Transformerがengineのrow/column/global tokenを読み、positionごとに可変なlegal-action setをscoreします。

2026年8月までにself-play corpusが約47,693 samplesへ増え、明示的な2D board inductive biasを持つCNNがlocal geometryをよりsample-efficientに学ぶかを比較できるようになりました。

2026-08-08のablationでは、同一corpus上でparameter scaleを近づけた3 familyを比較しました。

- existing TetraFormer-S control
- residual CNN board encoder + global-token pooling
- CNN+Transformer hybrid

optimizer、update count、minibatch schedule、seedなども揃えました。

結果は、1つのmetricだけでは説明できませんでした。

- TransformerはTransformer-search-generated teacherに対するheld-out policy cross-entropyが一貫してわずかに良かった。
- corrected hashed splitではCNNがWDL/value targetを大幅に良く学習した。
- CNNの最も再現性の高い優位はnetworkを**search内部へ戻したとき**に現れた。
- 16-simulation Arenaでは独立seed blockを跨いでCNNがmatched Transformerより強かった。
- 一方raw policy-only playはparityに近く、初期の大勝だけを一般化できなかった。

follow-up hybridも重要でした。

- independent compact CNN value branchはstatic held-out value metricが良くてもlate overfitした。
- small CNN token / fusion branchはfull CNN baselineのsearch advantageを再現できなかった。
- inference時にpolicy/value outputを単純swapしても、どちらか1 headだけが原因とは切り分けられなかった。

つまり少なくとも次の3問いは別です。

1. networkがsearch teacherをどれだけ正確に模倣するか。
2. held-out sampleでWDL/valueをどれだけ正確に予測するか。
3. policy/value representationをsearchへ埋め戻したとき、実際にどれだけ役に立つか。

関連はありますが、ablationは**同値ではない**ことを示しました。

## 決定

### 1. TetraFormerをreference/controlとして残す

新architectureが短いArenaで勝っただけでは、既存Transformerをproduction referenceから外しません。

逆に、CNNがstrong local inductive biasを持つという理由だけでTransformerを捨てません。

alternativeを測る間、TetraFormerをstable comparison targetとして残します。

### 2. CNN resultをreal evidenceとして扱うが、即production replacementにはしない

full CNN baselineのsearch advantageはcorrected splitとmultiple Arena seed blocksでも残りました。

したがってlocal board encoderをさらに研究する十分な根拠があります。

しかし、これはChampion architectureをsilent replacementしてよいことを意味しません。normal Candidate/Champion gateを通します。

### 3. Held-out policy lossをarchitecture selectorにしない

policy cross-entropyは**teacher imitation diagnostic**です。

今回のcorpusではteacher自体がTransformer-driven searchから生成されています。したがって小さなCE差はarchitecture-neutralなgame strength evidenceではありません。

architecture evaluationには次を必須とします。

- fixed search budget
- paired Arena
- multiple seeds

raw-policy Arenaも診断には使いますが、search evaluatorとしてdeployするnetworkならsearched Arenaが主要testです。

### 4. 次のhybridはconcrete representation hypothesisを検証する

small weakly-coupled CNN headを足し、「local geometryが何となくtransferする」ことを期待する実験は繰り返しません。

次のhybrid候補は、ablationが残した具体hypothesisを1つずつtestします。

例:

- successful full CNN baselineに近いcapacityのlocal CNN encoder
- isolated value branchではなくpolicy/value双方が同じlocal featureを使うshared representation
- CNN-derived spatial featureをTransformerへ渡し、Transformerはlong-range/global/opponent interactionを主に担当する構成

exact implementationはまだexperimentです。採用済みなのは、**hypothesisを明示し、common I/O contractを保ったcontrolled comparisonにする**という規則です。

### 5. Ablation中もproduction Championを保護する

architecture experimentが生成するのはexperimental Candidate checkpointです。

configured Arena promotion criterionを満たすまでChampionを変更しません。

## 帰結

- single metricで即座にmodel familyを1つへ絞るより、複数architectureを長く保持するためexperiment code/checkpoint管理は複雑になる。
- search Arenaが必要になるためcomparison costは上がる。
- その代わり、evaluatorが実際に使われるdeployment regimeを直接測定できる。
- policy loss、value metric、game strengthが異なる方向を示しても、それぞれを別diagnosticとして報告できる。
- future sparse MoEは、まず有用性が確認されたdense shared trunk上で試す。未解決のCNN-vs-Transformer representation questionをMoEで迂回しない（ADR 0016）。

## 根拠

固定data experiment、split audit、independent seed、policy/value factorial diagnostic、失敗したhybrid variantの詳細は `../CNN_ABLATION_20260808.md` を参照してください。
