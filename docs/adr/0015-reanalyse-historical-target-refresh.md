# ADR 0015: Reanalyse は厳密な履歴再生と非破壊ターゲット更新で行う

- Status: Accepted and implemented (minimum viable path)
- Date: 2026-08-16

## Context

同じ自己対局ターゲットを反復学習するだけでは、古い浅い探索の誤差も一緒に模倣する。Gen38/Gen39 ではオフライン方策一致が改善しても固定 seed の探索挙動が崩れたため、次の試行では学習回数を増やす前に教師ターゲットそのものを改善する必要がある。

矩形 v4 `.tetradat` はトークン、合法手埋め込み、方策、終局値、補助ラベルに加え、`game_seed`、時系列の `move_number`、`player_perspective`、実際に選ばれた `chosen_action` を持つ。ただし完全な `Player` 状態や、学習列としての `search_value` は持たない。トークンから hidden state を推測する実装は、queue、hold、clock、garbage、B2B/combo、乱数状態を捏造する危険がある。

## Decision

Reanalyse の最小実装は次の契約にする。

1. C++ の権威あるルール実装で、各ゲームを seed から開始し、記録された `chosen_action` を時系列に再生する。
2. 各履歴 root で再構成したトークンと全合法手埋め込みを保存済み矩形データと照合する。全行一致しないデータは再探索しない。
3. 現行教師の生方策を安く評価し、履歴探索方策との KL divergence が大きい順に staleness/surprise 候補を選ぶ。
4. 選ばれた一部だけを、元探索より大きい固定 simulation budget、root noise 0 で再探索する。全 replay buffer の一様な深探索を既定にしない。
5. 元ファイルは変更しない。更新方策だけを含む別 `.tetradat` shard と、旧方策・現行生方策・再探索方策・現行生 value・再探索 search value・選択理由を含む監査 JSONL、双方の hash と探索条件を含む manifest を出す。
6. `chosen_action`、terminal W/D/L、trajectory-derived auxiliary labels は元のまま保持する。反実仮想の最善手で履歴行動を上書きしない。
7. trainer には既存の source-aware sampling で別 source として渡し、Reanalyse の比率を通常 replay と独立に制御する。

実装入口は `trainer/reanalyze.py`、C++ の再構成・選別補助は `include/tetra/reanalyse.hpp`、GPU protocol は `tetra_cli gpu-reanalyse-protocol` とする。

## Binary-format boundary

矩形 v4 には独立した search-value training column がない。今回は binary schema を同時変更せず、terminal `value_target` を不正に置換しないことを優先する。再探索 search value は監査 JSONL に lossless に保存し、方策ターゲットだけを学習可能 shard へ反映する。

将来 search value を直接損失へ使う場合は、append-only の新しい dataset version と明示的な valid mask を別 ADR で導入する。監査 sidecar を暗黙に terminal value として読む実装は禁止する。

## Initial acceptance evidence

2026-08-16 に以下を実行した。

- Gen24 clean canonical seed `19000000..19000009`: 5,366/5,366 rows の token/action parity が一致。上位 270 rows（約 5%）を Gen24 teacher で 64 から 128 simulations に更新。
- Gen15 seed `15082005` の historical 0.424 APP replay: 255/255 rows が一致。競技用 Gen14 champion で上位 13 rows を 100 から 200 simulations に更新。
- 通常テスト 289 tests / 1,055,122 assertions / 0 failed。Reanalyse 専用テスト 3 tests / 17 assertions / 0 failed。
- 生成 shard は Python loader の sanity check、10-shard concatenate、chosen-action と terminal-value 保持確認に合格。

この結果は plumbing と historical target refresh の受け入れであり、学習後モデルの強さ改善を意味しない。固定 clean seed と frozen Arena gate は従来どおり必要である。

## Consequences

### Positive

- hidden state をトークンから逆算せず、実シミュレータ状態を再現できる。
- 元教師と新教師を監査でき、元データをいつでも再利用できる。
- 計算を stale/surprising roots に集中できる。
- clean と competitive の教師・attack-delivery 条件を分離できる。

### Costs and risks

- v4 provenance がない旧 dataset、`chosen_action == -1` の行、未知 ruleset、timing/attack 条件が一致しないデータは再構成 gate で拒否される。
- KL 上位選別は高 APP や Quad を直接最適化しない。これは意図的な分布保護だが、最適な selector である保証はない。
- 深探索ターゲット改善が rollout 改善へ直結する保証はない。検索の非線形性は残る。
- search value は現段階では監査対象で、学習列ではない。

## Rejected alternatives

- トークンから simulator state を推測する: hidden state を正確に復元できないため却下。
- `.tetradat` を in-place rewrite する: provenance と比較可能性を失うため却下。
- `chosen_action` や terminal outcome を再探索結果で上書きする: historical trajectory label と counterfactual target を混同するため却下。
- 全局面を同じ大予算で再探索する: 計算効率と surprise selection の検証可能性を損なうため却下。
- Reanalyse と dataset schema、timing、MoE、reward shaping を同時変更する: 初回 ablation の因果帰属ができなくなるため却下。
