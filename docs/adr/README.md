# 設計判断記録（Architecture Decision Records / ADR）

ADRは、**なぜその設計を選んだか**を保存するための文書です。`SPEC.md` は当初設計のbaselineとして残し、後から生じた判断を原仕様へ逆挿入しません。

## Statusの意味

- **採用** — 現在も設計上の制約として有効。実装済みかどうかはcode/testで別途判断する。
- **採用・実装方式は更新済み** — 根本の不変条件は有効だが、当時の具体実装は後の方式へ置換されている。
- **採用方針・実装延期** — 方向性は決めたが、現在のproduction pathへはまだ入れない。
- **提案** — 検討中でありproject constraintではない。
- **置換済み** — 歴史的記録として残し、現在は別ADR/実装が置き換えている。

既存ADRを現在形へ全面改稿することは避けます。後の判断でscopeが変わった場合は、古い理由を残したまま「追補」または新ADRへのlinkを追加します。

## 一覧

| ADR | 判断 | 状態 |
|---|---|---|
| [0001](0001-cpp-instead-of-rust.md) | rule coreはRustではなくC++で実装する | 採用。言語選択は維持、現行standardはC++23 |
| [0002](0002-movegen-state-includes-rotation-provenance.md) | 合法手生成でrotation provenanceを失わない | 採用・旧BFS実装はCobraへ置換済み |
| [0003](0003-movegen-performance.md) | movegenをMCTS inner loopとして最適化する | 旧project-side generatorはpure Cobraにより置換済み |
| [0004](0004-action-duration-and-dijkstra.md) | actionに実時間costとbounded delay binを持たせる | timing contractは採用、旧Dijkstra route enumerationはCobraへ置換済み |
| [0005](0005-replay-format.md) | replayはderived stateではなくinputを記録する | 採用 |
| [0006](0006-gravity-reachability.md) | gravityをreachability constraintとして扱う | 採用 |
| [0007](0007-evaluator-interface-first.md) | searchより先にbatched Evaluator interfaceを固定する | 採用 |
| [0008](0008-search-gumbel-calibration.md) | low-budget searchではGumbel sequential halvingを既定とする | 採用 |
| [0009](0009-determinization-and-selfplay.md) | hidden futureをdeterminizeし、terminal rewardはgame resultだけにする | 採用 |
| [0010](0010-cpp-python-handover.md) | C++↔Pythonで1つのpadded/masked tensor contractを共有する | 採用 |
| [0011](0011-cpp-inference-without-onnx.md) | ONNX RuntimeなしでC++ trained-weight inferenceを持つ | 採用 |
| [0012](0012-compact-dataset-replay-pi.md) | 再構成可能なsampleにはcompact Replay+π形式を使う | 採用。現行二盤面self-playはrectangular v3 |
| [0013](0013-architecture-ablation-and-local-geometry.md) | CNN/Transformer変更はpolicy loss単独でなくArena/search evidenceで判断する | 採用 |
| [0014](0014-objectives-auxiliary-targets-and-vs-score.md) | WDLを目的の基準に残し、dense auxとVS Scoreを補助/診断へ使う | 採用 |
| [0015](0015-selfplay-provenance-search-mixture-and-timing-curriculum.md) | 自己対局provenanceを固定し、timing/相殺外しを段階導入する | 採用 |
| [0016](0016-defer-sparse-moe-and-build-for-interpretability.md) | strong-playing agentとteaching agentを分け、自然言語媒介をimplementation-agnosticに設計する | 採用方針・実装延期 |

## 当初仕様後の判断の流れ

1. **探索・data correctnessを先に固定（0007–0012）。** Evaluator boundary、low-budget search、hidden information determinization、C++/Python contract、forward parity、dataset形式を、学習scale-upより先に安定させた。
2. **architecture評価を単一metricから分離（0013）。** 2026-08-08 CNN ablationで、teacher imitation、WDL learning、search strengthが一致しないことが判明したため、model selectionにArena evidenceを必須とした。
3. **reward shapingを避けたままdense supervisionへ拡張（0014）。** WDLをterminal objectiveに残し、multi-horizon / action-conditioned targetを許容する。VS Scoreは実装後にcombat diagnosticへ追加する。
4. **自己対局を管理されたcurriculumとして扱う（0015）。** provenance、search-budget mixture、Champion保護、timing/相殺外しの段階導入を明文化した。
5. **人間学習を目的にし、MoE/SAEを手段へ降ろす（0016）。** maximally strong playとteaching/analysis qualityを分け、natural languageをhuman questionとreproducible engine/model evidenceの双方向媒介層として位置づける。Sparse MoE / SAEは将来の候補技術であり固定要件ではない。

## ADRを追加するとき

最低限、次を記録します。

1. **状態**
2. **背景** — concrete failure mode、measurement、trade-off
3. **決定** — 何がproject ruleになるか
4. **帰結** — benefitだけでなくcost・risk
5. 必要に応じてexperiment report / implementation planへのlink

単なる実験観測を、projectとして方針を決めていない段階で「採用済み判断」に格上げしないでください。
