# ADR 0014: WDLを目的の基準に残し、dense auxiliary targetとVS Scoreを診断へ使う

## 状態

採用。

## 背景

ADR 0009ではterminal outcomeとattack/garbage statisticsを意図的に分離しました。

- self-play reward: actual game result
- combat quantity: auxiliary data

この分離は現在も重要です。hand-shaped rewardを加えると、局所的には魅力的でも「勝つ」という本来の目的とずれたbehaviorを強く学習する危険があるためです。

一方、初期trainingからWDL-only supervisionにも限界が見えました。

1. 1 complete gameから得られるindependent terminal outcomeは少ないが、中間positionには大量の学習可能情報がある。
2. policy/value/auxiliary headはshared trunkを通じて干渉するため、「教師信号を増やせば自動的に良くなる」とは限らない。

そのためsample-efficiency workでは、trajectory-derived targetを複数horizonへ拡張し、gradient diagnosticsを追加しました。

Arena report側ではAPM、APP、PPSを使っていましたが、これらはattack volumeやspeedを直接表す一方、versus combat effectiveness全体を1つで捉える指標ではありません。

そこで、TETR.IO文脈のVS Scoreをadditional combat diagnosticとして導入する方針を決めました。

ただしどのcombat statisticもproxyです。actual paired win/lossをpromotion criterionから置き換えません。

## 決定

### 1. Terminal game resultをreward/value anchorとして維持する

outcome targetはfixed player perspectiveからのwin/draw/lossです。

次の量で置換しません。

- attack
- stack shape
- cancellation
- survival time
- VS Score
- その他combat heuristic

これらをhand-shaped terminal rewardとして直接加えません。

piece limitによるtruncationもsynthetic winへ変換しません。`truncated` trajectoryで未知のfuture horizonは0 labelにせずmaskします。

### 2. Dense trajectory-derived supervisionでrepresentationを改善する

training objectiveは概念的に次です。

\[
L = L_{\pi} + \lambda_v L_v + \sum_i \lambda_i L_{\mathrm{aux},i}.
\]

現行・計画中のauxiliary family:

- 複数time/placement horizonのfuture attack
- future garbage received
- self/opponent top-out / survival horizon
- time-to-terminal / discrete time-to-event
- engineからexactに計算できるaction-conditioned immediate consequence
- ablationを経たVS Scoreなどのcombat summary prediction

Dense targetは1 trajectoryから得られるsupervisionを増やしますが、independent gameを増やすわけではありません。

したがってsample-efficiency claimは、fixed game countまたはfixed generation computeで比較します。

### 3. VS ScoreをArena/reportingの標準診断へ追加する

**実装とruleset interpretationをtest/documentで固定した後**、Arena/match reportへVS ScoreをAPM/APP/PPSと並べて追加します。

評価hierarchy:

1. specified search budgetでのpaired Arena win rate + confidence interval
2. VS Score — combat-oriented explanatory metric
3. APM / APP / PPS / survival / cancellation / timing statistics
4. held-out policy/value/auxiliary metrics

VS Scoreをwin rateより上位には置きません。

VS Scoreは「なぜstrongerなのか」を説明したり、APM/APPだけではblurするstyle差を見たりするためのproxyです。proxy自体をterminal objectiveにしません。

### 4. VS Scoreをauxiliary targetにする場合は別ablationとする

VS Scoreをreportへ出すことと、training targetに使うことは別判断です。

VS auxiliary headはexperimental targetとして、otherwise identical baselineと比較します。

target生成に使うexact formula/version/ruleset interpretationはschemaとともに記録します。

意図したmetricを正確に再現できない場合、undocumented approximationをlabelにせず、limitationを明記します。

### 5. Scalar lossだけでなくgradient interactionを監視する

auxiliary weightをraw loss magnitudeだけで選びません。

training logではshared trunk上の次を観測可能にします。

- policy gradient norm
- value gradient norm
- auxiliary gradient norm
- policy/value gradient cosine
- policy/auxiliary gradient cosine

small auxiliary lossでもgradientがpolicyを支配・oppositionする場合があります。逆にnumerically large lossでもweight後のgradientが小さいことがあります。

valid mask=0のtargetはlossだけでなくgradientにも寄与させません。

### 6. Auxiliary targetの有効性にはgameplay evidenceを要求する

auxiliary prediction自身のheld-out lossが改善しても、policy/searchに有用なrepresentationを得たとは限りません。

auxiliary targetを「strategically useful」と判断するには、controlled experimentで少なくとも次のどちらかを確認します。

- policy metricの改善
- Arena/gameplayの改善

かつ追加computeが許容範囲であること。

representation qualityとsearch interactionを切り分ける必要がある場合は、raw-policy / searched-policy双方を残します。

## 帰結

- trainerとdataset schemaはよりexplicitかつwideになる。
- Arena reportへcombat-specific diagnosticを増やしても、Champion promotion ruleは弱めない。
- auxiliary objectiveはaggressiveに試せるが、不要なら取り外せる。
- strengthとcorrelateするという理由だけで、そのmetricをprojectにおける「winningの定義」へ格上げしない。
- APM/APPは引き続き有用。特にAPPはtiming curriculum前にbasic attack constructionが存在するかを見るrough diagnosticになるが、VS Scoreやwin rateの代替ではない（ADR 0015）。

## 関連文書

- ADR 0009 — terminal rewardとdeterminized self-play
- `../SAMPLE_EFFICIENCY_PLAN.md` — target schema、mask、split、gradient diagnosticの詳細
- `../TRAINING_AND_EVALUATION.md` — experiment / Arena protocol
