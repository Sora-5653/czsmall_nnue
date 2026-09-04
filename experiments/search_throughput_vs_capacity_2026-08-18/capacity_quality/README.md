# Phase 0: capacity × quality confound check

## 結論と停止判定

既存 checkpoint の A と S400について、primary seed=42、50 pair、各 pair 4局の
200局を no-search と equal-node 64 の両方で完了した。S400 は A に対して両条件で
大敗したため、handoff の Stop A を成立と判定する。

この判定は、旧XSの3%結果だけから容量不足を断定するものではない。同じ規模の
S400も、探索を無効にした場合と同じnode budgetを与えた場合の双方で弱かったため、
旧400-step・Gen4系列の学習世代差または学習条件差が主要な交絡候補である。

Stop A成立後は、次の処理を実行していない。

- M400/XS400の追加同条件測定
- 新規学習・公平な再学習
- policy/value head-swap
- throughput測定、real forward ratio測定、wall-clock Main
- checkpointのpromotionまたは置換

## 固定 checkpoint

| label | path | role |
|---|---|---|
| A | `models/gen14_rank100_100_20260814.best.pt` | frozen current reference |
| S400 | `models/size_search_ablation_20260816/seed42/transformer_s.final.pt` | primary same-size / old-training control |
| M400 | `models/size_search_ablation_20260816/seed42/transformer_m.final.pt` | Stop Aにより未測定のsecondary control |
| XS400 | `models/size_search_ablation_20260816/seed42/transformer_xs.final.pt` | Stop Aにより未測定のsecondary control |

全 checkpoint の絶対 path、SHA-256、architecture、parameter数、埋め込みmetadata、
commitとunknown項目は `manifests/checkpoints.json` に保存した。A/S400/M400/XS400は
実験中に変更していない。

## primary seed と4局展開

primaryは `base_seed=42` 固定である。実行は `--pairs 50 --seed 42` とし、C++
`Arena::evaluate` が各 `pair_index=i` に対して次を計算する。

```text
pair_seed = uint64(42 + i * 0x9E3779B97F4A7C15)
```

各 pair は同じ `pair_seed` で次の4局をこの順に実行する。

1. `mirror=false, roles_swapped=false`
2. `mirror=false, roles_swapped=true`
3. `mirror=true, roles_swapped=false`
4. `mirror=true, roles_swapped=true`

したがって、50 pair × 4局 = 200局である。全pair seedとC++ソース上の展開根拠は
`manifests/seed_design.json` に保存した。既存の `2026081800` 始まりのseed listは
primary結果には使っていない。

## runner条件

既存 `experiments/search_throughput_vs_capacity_2026-08-18/scripts/run_condition.py`
を再利用した。C++側がルール、movegen、探索、paired game、集計を担当し、Python側は
checkpointをGPU bridgeへ供給した。共通条件は `configs/phase0.json` に保存している。

- engine: `build/tetra_cli.exe`
- ruleset/environment: E3 / `garbage_style=1`, `garbage_period=8`, `garbage_lines=2`
- device: `cuda:1`、AMD Radeon RX 9070 XT、ROCm PyTorch
- precision: fp16、batch 16、determinizations 1、Gumbel有効、timing actions無効
- `pieces=300`, protocolの基準 `sims=100000`
- `fixed_token_count=0`, `fixed_action_count=0`
- stderrは`drain`、batch prewarmあり、watchdog 300秒
- candidate=A、champion=S400。従って結果JSONの`wins`はAの勝ち、`losses`はS400の勝ち

既存runnerは `--require-gpu` を公開していないため、`scripts/run_phase0.py` が
実行前に `torch.cuda.is_available()`, HIP version, device index/name, gfx architecture
を検査し、`--require-gpu` が渡されない場合は停止する。CPU fallbackは許可していない。
GPU情報、childの正確なコマンド、commit、checkpoint hash、所要時間は
`notes/run_records/` と結果JSONへ記録した。

## 測定結果

結果JSONにある `games`, `wins`, `losses`, `draws`, `ci95_low`, `ci95_high` を集計の
権威とした。CIは勝率に対するWilson 95%区間である。S400側のscore rateは
`(losses + 0.5 * draws) / games` とし、勝敗が二値の本測定ではS400の区間はAの区間を
反転したものになる。

| condition | games | A wins | S400 wins | draws | A score / Wilson 95% CI | S400 score / Wilson 95% CI | result |
|---|---:|---:|---:|---:|---:|---:|---|
| no-search policy control | 200 | 179 | 21 | 0 | 89.5% / 84.48–93.03% | 10.5% / 6.97–15.52% | complete |
| equal-node 64 control | 200 | 189 | 11 | 0 | 94.5% / 90.42–96.90% | 5.5% / 3.10–9.58% | complete |

両結果とも `seed=42`, `pairs=50`, `games=200`, `device=AMD Radeon RX 9070 XT`,
`git_commit=04d49ffee95f1c937c07ab60e88a868b6eb61ef6` である。

### no-search の適用確認

結果の diagnostics で A/S400とも `simulations=0`、`raw_policy_matches=decisions`、
`searched_action_changes=0` を確認した。C++ child commandの有効部分は次のとおりで、
両側のsimulationを0にしている。

```text
gpu-arena-protocol 50 100000 300 16 1 1 42 0 0 -1 -1 0.01 0.05 -1 -1 -1 -1 -1 -1 -1 -1 1 8 2
```

### equal-node 64 の適用確認

`candidate_sims=-1`, `champion_sims=-1` とし、`candidate_node_budget=64`,
`champion_node_budget=64` を設定した。同じC++ search algorithm、Gumbel設定、seed展開で
node budgetだけを64に固定した。C++ `tools/tetra_cli.cpp` の引数位置と
`include/tetra/arena.hpp` のSearchConfigへの代入をソースで確認し、結果JSONのchild
commandにも `... -1 -1 64 64 1 8 2` が残っている。

診断の要点は次のとおりである。

| side | decisions | simulations | nodes | node budget argument | searched action changes |
|---|---:|---:|---:|---:|---:|
| A / candidate | 19,615 | 1,635,474 | 1,255,199 | 64 | 1,408 |
| S400 / champion | 18,883 | 9,406,462 | 1,204,798 | 64 | 5,719 |

`node_budget_cutoffs=0` は、探索ループが次のsimulationへ進む前にnode budget到達を
判定する実装であるため、budget未適用の根拠にはしない。適用根拠はchild commandの
64/64、C++引数解析、ArenaからSearchConfigへの代入、および両側の1局あたりnode数が
概ね64に制約されていることの組合せである。

## 接続確認の扱い

本測定前に、`pairs=1`, `pieces=5` の別smokeを実行した。これは4局すべてdrawで、
`games=4`, `wins=0`, `losses=0`, `draws=4` だった。局面が短すぎる接続確認であり、
no-search 200局またはequal-node 64 200局の集計には混ぜていない。

smoke結果は `results/smoke/A-vs-S400-no-search-seed42-pairs1.json` と対応する
state/trace/logに分離保存した。

## 実験上の最終表

これはhandoffの「容量cliff」と「training cliff」を分ける表を、Stop A時点で埋めた
ものである。`equal-node WR vs A` は各モデル側から見た勝率、`10ms WR vs A` と
`real forward ratio` はこのPhase 0では測定していないため未測定とした。

| model | params | training data | steps/samples | converged? | equal-node WR vs A | 10ms WR vs A | real forward ratio |
|---|---:|---|---:|---|---:|---:|---:|
| A | 7,176,624 | unknown in checkpoint | step 3312 / samples unknown | unknown | 基準 50% | 未測定 | 未測定 |
| S400 | 7,175,592 | Gen4 / 47,693 samples | 400 / 47,693 | no/unknown | 5.5% (200局) | 未測定 | 未測定 |
| M400 | 946,920 | Gen4 / 47,693 samples | 400 / 47,693 | no/unknown | 未測定（Stop A） | 未測定 | 未測定 |
| XS400 | 132,744 | Gen4 / 47,693 samples | 400 / 47,693 | no/unknown | 未測定（Stop A） | 3% existing pilot* | 未測定 |
| S-new | unknown | 未実施 | 未実施 | 未実施 | 未測定 | 未測定 | 未測定 |
| M-new | unknown | 未実施 | 未実施 | 未実施 | 未測定 | 未測定 | 未測定 |
| XS-new | unknown | 未実施 | 未実施 | 未実施 | 未測定 | 未測定 | 未測定 |

`*` XS400の3%は旧別実験の値であり、Phase 0の200局結果ではない。この値だけから
容量不足とは結論しない。

## 観測・推論・未確認

### 観測

- no-search 200局で A=179勝、S400=21勝、引き分け0。
- equal-node 64 200局で A=189勝、S400=11勝、引き分け0。
- いずれもprimary seed=42、50 pair、4局展開、GPU gate通過、exit code 0。
- A/S400のcheckpoint hashはmanifestとrun recordで一致し、測定中にモデルファイルを変更していない。
- preflightのHEAD、dirty status、GPU、engine、テスト結果は `notes/preflight.md` にある。

### 推論

- S400が同規模の比較対象で no-search と equal-node 64 の双方に大敗したため、
  旧A/XS差を容量差だけで説明することはできない。
- 旧400-step・Gen4学習条件、データ、checkpoint世代差が、少なくとも一次の交絡候補である。
- この結果は「現行データで収束まで公平に学習した小型モデルも弱い」という意味ではない。

### 未確認

- held-out quality、policy/value swap、公平な再学習、収束後のM/XS品質は未確認。
- throughput Pareto、real forward ratio、同一wall-clockの強さは未測定。
- S400のarchitectureはAとparameter数がほぼ同じだが、Aの`aux_targets=44`に対して
  S400は`aux_targets=36`であり、完全に同一architectureではない。この差はmanifestに残した。
- A checkpointのdataset、training seed、validation metadataは埋め込まれておらずunknownである。

## Stop A後の再開手順

Stop A後に続行する場合は、Solから新しい明示的な判断を受け、次を順に再確認する。

1. 同じlocal working treeで、`git status --short --branch` とHEADを保存する。
2. A/S400/M400/XS400のSHA-256を `manifests/checkpoints.json` と照合する。
3. ROCm GPU gate、engine path、runner、commit整合を再確認する。GPUが見えなければ実行しない。
4. 本Phase 0を再現する場合は、次のコマンドを使う。seed listを差し替えず、primary seed=42を維持する。

```powershell
.venv-rocm714\Scripts\python.exe experiments\search_throughput_vs_capacity_2026-08-18\capacity_quality\scripts\run_phase0.py --mode no_search --device cuda:1 --require-gpu --seed 42 --pairs 50 --pieces 300 --watchdog-seconds 300
.venv-rocm714\Scripts\python.exe experiments\search_throughput_vs_capacity_2026-08-18\capacity_quality\scripts\run_phase0.py --mode equal_node64 --device cuda:1 --require-gpu --seed 42 --pairs 50 --pieces 300 --watchdog-seconds 300
```

5. M400/XS400、再学習、head-swap、throughput、wall-clockへ進む場合は、Stop Aを覆す
   新しい仮説、条件、局数、資源承認を先に記録する。本成果物だけを根拠に自動続行しない。

## 成果物の対応

- `configs/phase0.json`: 条件とStop A scope
- `manifests/checkpoints.json`: A/S400/M400/XS400 metadata、hash、unknown
- `manifests/seed_design.json`: seed 42を起点とする50 pairの決定論的展開
- `scripts/build_manifests.py`: manifest生成スクリプト
- `scripts/run_phase0.py`: GPU gate付き既存runner再利用スクリプト
- `results/phase0/`: 200局の機械可読結果JSON
- `results/phase0/phase0_summary.json`: A/S400双方のscore、Wilson CI、budget、runner条件、ログ参照、Stop A判定をまとめた機械可読JSON
- `results/smoke/`: 4局の接続確認。本結果と分離
- `results/logs/`: stdout/stderr、request、trace、state、hang記録
- `notes/preflight.md`: 非破壊preflightとソース確認
- `notes/phase0_result.md`: 実行コマンド、所要時間、失敗・再試行、検証記録
- `notes/run_records/`: GPU gate、child command、hash、commit、結果の実行記録
