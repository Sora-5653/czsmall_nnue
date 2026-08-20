# ADR 0016: AI→人間学習構想を特定手法へ固定せず、Sparse MoEは延期する

## 状態

採用方針・実装延期。

## 背景

長期目標はversus engineを単に強くすることだけではありません。学習loopから十分に強いagentが得られた後、そのagentが獲得したstrategic knowledgeを**人間が理解・検証・学習できる形へ変換する**ことも研究対象にします。

この目標には、関連するが異なる2 trackがあります。

### 1. 「強く指すagent」と「教えるagent」は同一目的ではない

最大限強く指すことだけで最適化されたmodelが、人間に説明しやすい形でknowledgeをorganizeする保証はありません。

そこで次を分けます。

- **playing agent:** overwhelming game strengthを追求する。
- **teaching/analysis agent:** useful conceptの抽出、counterfactual analysis、explanation、curriculum生成を追求する。

teaching agentはplaying agentからdistillしたり、trace/activation/search statisticsを読んだり、probeを投げたりできます。人間は必ずしもstrongest modelの内部を直接decodeする必要はありません。

### 2. Natural languageをgame worldへの双方向interfaceにしたい

将来のlanguage-capable mediatorには2方向の役割を持たせます。

- **human → engine/research problem:** 人間のstrategic questionをposition、slice、search、intervention、comparison、hypothesisへ変換する。
- **engine/model → human:** state、counterfactual result、learned feature、routing pattern、discovered regularityをexplanation / lessonへ変換する。

Tetrisは最初の具体domainですが、これはTetris専用UIというより、**自然言語をformal problem spaceとのinterfaceにする原則**として設計します。

Sparse Mixture-of-Experts（MoE）やSparse Autoencoder（SAE）は、このagendaで有望な技術です。

- MoE routingは、modelがpositionをどのregimeとして扱うか粗いobservable signalを与える可能性がある。
- SAEは、internal activation中のrecurring featureを抽出できる可能性がある。

しかしどちらもgoalそのものではありません。

さらにnear-term engineering上、Tetris versusにはopening、midgame、finisher、high-stack survival、garbage interaction、timingなど異質なregimeがあるためsparse specializationは魅力的ですが、expertを早く増やすとdata fragmentation、router learning、rare expert starvationが発生します。

## 決定

### 1. Playing strengthとteaching qualityを別objectiveとして扱う

Champion/Arena loopは引き続きplaying strengthを最適化・測定します。

将来のteaching/analysis agentは別criteriaで評価します。

例:

- useful strategic distinctionを抽出できるか
- counterfactual questionへ答えられるか
- conceptをrepresentative positionへgroundできるか
- uncertaintyを表現できるか
- explanationをengine experimentでcheckできるか
- 人間のreasoning / play improvementへ実際に寄与するか

teaching agentはstronger playing agentのtrace、activation、search statistic、learned feature、demonstrationを利用できます。architectureが同一である必要はありません。

### 2. Natural languageはgame truthではなくbidirectional mediation layerとする

将来language layerは次をsupportします。

#### Human → engine / model

strategic questionをconcrete probeへ変換します。

- position generation
- labelled slice retrieval
- search comparison
- intervention
- ablation
- hypothesis test

#### Engine / model → human

次をexplanationへ変換します。

- discovered regularity
- feature activation
- routing pattern
- counterfactual result
- representative position
- failure case

ただしgame factsのauthorityはC++ simulator、search measurement、recorded provenanceに残します。

language-model outputはinterfaceとhypothesis generatorであり、exact rule/search evidenceの代替ではありません。

### 3. Interpretability mechanismをimplementation-agnosticに保つ

AI→human learning pipelineをMoE、SAE、特定probe methodへ固定しません。より良いmethodが現れた場合は置き換えられるようにします。

stable requirementは次のpipelineです。

1. strongかつwell-characterizedなagentを得る。
2. provenance-taggedなinternal / behavioural evidenceを集める。
3. candidate strategic conceptを抽出する。
4. counterexample、intervention、ablation、controlled engine experimentで検証する。
5. validated conceptをhuman-readable explanation / training materialへ媒介する。
6. teaching layerが実際にhuman reasoning / playへ寄与するか評価する。

### 4. Sparse MoEを次のscaling stepにしない

near-term priorityはdense self-play loopを強化し、CNN/Transformer representation questionを解き、stable Arena competenceを得ることです。

Sparse MoE experimentは次が揃うまで延期します。

- dense baselineがdemonstrably strong
- self-play generationとCandidate/Champion gatingがstable
- specialistを十分trainできるdiverse data
- dense modelとcomparable computeで追加complexityを測定できる

MoEをweak target、insufficient data、unresolved trunk architectureのshortcutとして使いません。

### 5. MoEを試す場合はsmall・shared・observableから始める

first MoE ablationではproven shared representationを残し、small number of sparse expert blockを追加します。

opening/midgame/finisher、high-stack survival、garbage pressureなどはuseful hypothesisですが、最初からnamed tacticのCartesian productごとにexpertを作りません。

最低限logするrouting information:

- selected expert / top-k probability
- router entropy
- load balance
- game record上のphase/situation metadata
- labelled slice別expert usage
- practicalな範囲でexpert/router ablation performance

human labelはspecialization hypothesisでありsemantic guaranteeではありません。

### 6. SAEはstrong agent後のcandidate feature-extraction methodとして扱う

十分強いcheckpointができた後、SAE-based workでは次を試せます。

1. shared trunk / expert blockからactivationをprovenance付きcorpus上で収集する。
2. selected activation streamへSparse Autoencoderを学習する。
3. learned featureが強くactivateするposition/actionをretrieveする。
4. board structure、attack timing、garbage pressure、phase、tactical outcomeなどengine-derived conceptとcorrelateする。
5. counterexample / interventionでinterpretationをchallengeする。
6. validated featureだけをlanguage/teaching layerへ渡す。

「それらしいexampleが見つかった」だけではcausal explanationとみなしません。

### 7. Explanationへuncertaintyを残す

router telemetry、SAE feature、linear probe、intervention、language explanationは、それぞれevidence strengthが違います。

plausible interpretationをすべて確定strategyとして提示せず、validation levelに応じてconfidenceを分けます。

最終consumerがhuman learnerであるため、explanationの確信度をevidenceの強さへ合わせることを設計要件とします。

## 帰結

- near-term modelはpremature MoEよりsimpleかつsample-efficientに保てる。
- 将来specializationする余地を残しつつ、現在の段階でbrittle tactical taxonomyをhard-codeしない。
- interpretabilityを「routerを見る」「SAEをtrainする」だけへ縮小せず、AI→human learning system全体として設計できる。
- teaching agentがplaying strengthとは別のfirst-class research objectになる。
- natural-language mediationをexact engine evidenceへgroundし、fluentだが実態と離れたexplanationを減らせる。

## 関連文書

- ADR 0013 — dense CNN/Transformer architecture experiment
- ADR 0014 — auxiliary target / evaluation hierarchy
- ADR 0015 — staged self-play / timing curriculum
- `../ARCHITECTURE.md` — current architectureとdeferred research placement
