# ADR 0015: 自己対局provenanceを固定し、timing・相殺外しは基本戦術の後に段階導入する

## 状態

採用。

## 背景

training loopはlocal GPU generation、Colab worker、replay mixing、resumable training、paired Candidate/Champion Arenaまで広がりました。

これによりsample volumeを増やしやすくなった一方、2つのriskが大きくなりました。

1. commit、checkpoint、ruleset、schema、seed、search budgetが違うdatasetを混ぜ、apparent gainの原因がtraining methodではなくdata provenance changeになる。
2. delay binがaction spaceに存在するだけで、「timingを学習した」と誤認する。

後者は実際に観測されました。初期trained policyはほぼ常にfastest placementを選んでいました。`WAIT_FOR_EVENT` や他のdelay choiceは存在していても、garbage timing・cancellation avoidance（相殺外し）を学ぶだけのexploration/data coverageがあるとは言えませんでした。

同時に、uniformly deep searchで全self-playを生成するのは高価です。広いcoverageにはshallow searchを多く使い、より高品質なteacher signalを少量のdeep searchで注入するmixtureが有力です。

## 決定

### 1. Generated shardをimmutable・provenance-tagged artifactとして扱う

本格self-play shardは具体的generation configurationへ追跡可能にします。

最低限:

- repository commit
- checkpoint identity/hash
- ruleset hash
- dataset/tokenizer/action/aux schema version
- model version
- search algorithm / simulation budget
- determinization / root-noise setting
- seed interval / shard identity
- sample count / termination metadata

shardをbyte-concatenateせず、incompatible schemaをsilent mergeしません。

Colab、Drive、Google Apps Scriptはartifact transport/orchestrationには使えますが、seed allocation、label、dataset merge semanticsの権威にはしません。

### 2. Data-generation experimentからChampionを保護する

experimentが作るのはCandidateです。Championはnormal paired Arena gateだけで更新します。

次のもの単独ではChampionをoverwriteしません。

- training runが成功した
- validation lossが下がった
- APM/APP/VS Scoreが高い
- 1回の短いArenaで勝った

self-play distribution自体を変えている最中もcomparison targetをstableに保つためです。

### 3. Uniform search budgetではなくsearch-strength mixtureを試す

self-play loopが安定した後は、次を基本候補とします。

- **mostly shallow-search games/positions:** broad coverageを安価に得る。
- **smaller deep-search component:** より強いpolicy/value targetを注入する。
- multiple search strengths / deliberately imperfect or recovery positions: current best policyのnarrow on-policy manifoldだけへ閉じない。

position-start、stratified、recovery curriculumを導入した場合は、manifest上でdistinct provenance classにします。

sample-efficiency claimでは、total game countまたはgeneration computeを固定します。

### 4. Timingをstaged capabilityとして扱う

Delay actionはtraining全期間で表現可能にしておきますが、explicit timing curriculumをbasic board tacticsの後段へ置きます。

readinessは複数signalで見ます。

- stable stacking
- Quad
- T-spin
- basic attack construction
- Arena strengthがelementary board failureだけで決まっていない
- APP / 将来のVS Score / qualitative play

平積みQuad baselineのAPP約0.5は**rough diagnostic**として使えますが、hard gateでもreward targetでもありません。

### 5. Timing導入後はaction dimensionの実利用を測る

Delay binがgeneratorに存在することからtiming skillを推論しません。

最低限:

- `FASTEST` vs delayed action frequency
- `WAIT_FOR_EVENT` usage
- timing decision周辺のattack sent/received/cancelled
- definableな範囲でcancellation avoidance / off-cancel event
- same search budgetでのpaired win rate
- VS Score（実装後）

「delay actionを選ぶ頻度が増えた」だけでは成功としません。delayが有効な局面でgame outcomeを改善している必要があります。

### 6. Waitingへgeneric positive rewardを与えない

networkには**いつdelayが有用か**をsearch、game result、trajectory-derived targetから学習させます。

waitingやcancellation avoidance自体へgeneric rewardを与えると、別局面では誤ったtactical preferenceをhard-codeする危険があります。

exploration不足なら、terminal objectiveを変えるのではなく次を調整します。

- search exploration
- data sampling
- curriculum coverage
- search-strength mixture

## 帰結

- self-play artifact/manifestはややverboseになるが、runを再現・診断できる。
- deep-search computeを全positionへ均等に使わず、高品質教師として部分的に配分できる。
- board policyの基礎能力とtiming/cancellation能力を分けて評価できる。
- dense basic policyが弱い間は高度timing dataを意図的に少なくする可能性がある。これは、rare/difficult action dimensionが早期にtraining capacityを消費するのを避けるためのtrade-offである。

## 関連文書

- ADR 0009 — determinizationとterminal reward contract
- ADR 0014 — objective/metric hierarchy、VS Score
- `../COLAB_MANUAL.md` — shard generation workflow
- `../TRAINING_AND_EVALUATION.md` — fixed-budget ablation / Arena protocol
