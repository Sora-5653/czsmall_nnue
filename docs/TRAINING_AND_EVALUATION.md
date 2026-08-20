# 学習・評価プロトコル

この文書は、学習実験を比較可能に保つための短い運用契約です。補助目標生成の細部は `SAMPLE_EFFICIENCY_PLAN.md`、判断理由はADR 0014・0015を参照してください。

## 1. 何を「改善」とみなすか

本プロジェクトの目的は、単一のoffline lossを最小化することではなく、**管理された計算budgetのもとで実際に強いplayerを作ること**です。証拠の優先順位は次のようにします。

1. **固定search budget・paired seedのCandidate vs Champion Arena**。勝率と信頼区間を報告する。
2. **VS Scoreと戦闘診断**。同じpaired条件で比較する。
3. **APM、APP、PPS、生存時間、相殺、timing統計**。なぜ差が出たかを説明する。
4. **検証用policy/value/auxiliary metrics**。表現学習とtraining healthを診断する。

検証用policy lossが低いことは有用ですが、強いplayerであることを直接意味しません。CNN ablationが典型例です。TransformerはTransformer由来teacherをより良く模倣した一方、CNNはsearch内でより強い挙動を示しました。

## 2. 目的関数の契約

終端game resultを価値・rewardの基準として維持します。追加の密な教師信号を学んでも、reward shapingへ置き換えません。

概念的には次の形です。

\[
L = L_{\pi} + \lambda_v L_v + \sum_i \lambda_i L_{\mathrm{aux},i}.
\]

補助目標には次の制約を課します。

- recorded trajectoryと定義済みplayer perspectiveから導出できること。
- `terminated` と `truncated` を混同しないこと。
- 観測できないfuture horizonを0として学習せず、valid maskで除外すること。
- policy inferenceで見えないhidden stateを補助label生成に使って表現へ漏洩させないこと。
- loss weightはscalar lossだけでなくshared-trunk gradient norm / cosineも見て判断すること。
- 補助目標は、管理されたablationで方策・Arenaへ中立以上である場合だけ残すこと。

## 3. 密な補助目標

同じtrajectoryから複数の教師信号を抽出し、1局面あたりの学習情報量を増やします。ただし、label数が増えても独立したgame数が増えるわけではありません。

現行・計画中のtarget family:

- 複数time/placement horizonのfuture attack
- future garbage received
- self/opponent survival・top-out horizon
- time-to-terminal / 離散time-to-event
- engineから厳密に算出できるaction-conditioned immediate consequence
- ablationを経たVS Scoreなどのcombat summary prediction

VS Scoreはまず**評価指標**として導入します。VS Scoreをauxiliary targetとして使うことは別実験であり、WDL rewardやpromotion ruleを変更しません。

## 4. Dataset provenance

本格的なrunについて、「このsampleはどのcode・checkpoint・ruleset・schema・search設定・seedから生成されたか」を後から答えられる状態にします。

最低限、次を記録・検証します。

- repository commit
- checkpoint identity/hash
- ruleset hash
- dataset/tokenizer/action/aux schema version
- search algorithm、simulation数、determinizations、root noise
- game/seed interval、shard identity
- termination reason
- sample count、source generation

生成済みshardはimmutable inputとして扱います。datasetをbyte concatenationせず、schemaが異なるshardを暗黙に混ぜません。Colab / Drive / GASはartifact transportを補助できますが、seed、label、merge semanticsの権威にはしません。

## 5. Train / validation split

隣接局面を無作為に分けず、game、seed、またはshard単位でsplitします。

特に、source shardが連続したnumeric seed rangeを持つ場合、seedをsortして上位80% / 下位20%に切るだけでは、validationが「特定shardだけ」になる危険があります。単純な既定法として、**game seedのstable hashでsplitする方法**を優先します。

architecture比較では、同じsplitを共有し、可能ならsampled minibatch scheduleも共有します。

## 6. アーキテクチャ比較実験の契約

Transformer、CNN、hybrid、将来のMoEを比較するとき、次を固定または記録します。

- training sample set
- train/validation split
- optimizer・learning-rate schedule
- update数・batch size
- model parameter scale
- training seed
- search budget・Arena seed
- replay mixture・opponent/Champion checkpoint
- wall-clock time・accelerator time

1 seed・1 metricだけでproduction/reference modelを置き換えません。

architecture変更は、まずexperiment、次にCandidate checkpoint、最後に通常のChampion promotion gateという順に進めます。

## 7. Arena report

有用なArena reportには、最低限次を含めます。

| 分類 | 指標 |
|---|---|
| 昇格判定 | wins/losses/draws、win rate、95% confidence interval、paired seed protocol |
| 戦闘 | VS Score、APM、APP |
| 速度 | PPS |
| 生存 | mean survival time、placements to top out |
| Garbage interaction | sent/received/cancelled attack、cancellation efficiency |
| Timing | FASTEST / delayed action比率、`WAIT_FOR_EVENT` 使用率、timing関連cancellation |
| Search依存性 | 必要に応じてraw-policy / searched-policyの両方 |

VS Scoreはwin rateより上位の目標ではありません。APM/APPだけでは区別しにくいcombat styleを説明するproxyとして使います。

**現時点ではVS Scoreは未実装です。** 計算式とruleset interpretationを固定してから標準reportへ追加します。

## 8. Timing / 相殺外しカリキュラム

move generatorは `WAIT_FOR_EVENT` を含むdelay binをすでに出せますが、networkがそれらを使いこなすまではtimingを「学習済み」とみなしません。

段階は次のとおりです。

1. まずstable stacking、Quad、T-spinなど基本的なboard tacticsを獲得する。
2. 単一数値ではなく、play inspectionとcombat metricsを併用する。
3. 平積みQuadの理論baselineであるAPP約0.5は、attack constructionが成立し始めたかを見る**粗いreadiness signal**としてのみ使う。hard thresholdにはしない。
4. その後、garbage timing・相殺外しを含むdelay actionの探索/data coverageを強める。
5. action frequency、cancellation interaction、VS Score、paired Arenaで能力が実在するか検証する。

「待つこと」自体へ正のrewardを付けません。待つべき局面かどうかはsearchと実際のgame resultから学ばせます。

## 9. 自己対局のsearch mixture

自己対局loopが安定した後は、すべてのpositionをuniformに深く探索するより、次のmixtureを基本候補とします。

- **大部分を浅いsearch:** 安価に広いstate coverageを得る。
- **少量を深いsearch:** より高品質なpolicy/value targetを注入する。
- 複数search strengthや、意図的に不完全・recoveryが必要なpositionを含め、current policyの狭いon-policy manifoldだけに閉じない。

position-startやstratified curriculumを導入する場合は、dataset manifest上で別sourceとして記録し、通常self-playへ暗黙に混ぜません。

sample-efficiencyを比較するときは、game数またはgeneration computeを揃えます。

## 10. Champion promotion

Championは保護artifactです。training・ablationから得られるものはCandidateです。

CandidateがChampionを置き換えられるのは、設定済みpaired Arena gateを満たした場合だけです。

次のもの単独ではpromotion理由になりません。

- validation lossが下がった
- VS Score/APM/APPが高い
- 1つの短いArenaで勝った
- auxiliary lossが良くなった

この分離により、aggressiveな実験を行っても比較対象のChampionが途中でdriftしません。
