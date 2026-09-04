# Phase 0 実行記録

## 実行順

1. AGENTS.md と CAPACITY_QUALITY_HANDOFF.md をUTF-8全文で読み、dirty working tree、HEAD、runner、engine、GPU、checkpoint metadataを非破壊確認。
2. `scripts/build_manifests.py` で A/S400/M400/XS400 manifest と seed設計を生成。
3. 1 pair / 5 pieces の接続確認を別smokeとして実行。結果は本測定へ混入していない。
4. A vs S400 no-search を `--pairs 50 --seed 42` で実行。
5. A vs S400 equal-node64 を `--pairs 50 --seed 42` で実行。
6. equal-node64完了後、Stop A成立を判定。M400/XS400、再学習、head-swap、throughput、wall-clockは実行していない。

## GPU gate

使用したPythonは `C:\Users\eddyf\czsmall_nnue\.venv-rocm714\Scripts\python.exe`。

- `torch=2.12.0+rocm7.14.0`
- `torch.version.hip=7.14.60850`
- `torch.cuda.is_available()=True`
- device index 1: `AMD Radeon RX 9070 XT`
- architecture: `gfx1201`
- device count: 2（index 0の内蔵GPUは使用していない）
- runner wrapper: `--device cuda:1 --require-gpu`
- CPU fallback: 禁止

engineは `C:\Users\eddyf\czsmall_nnue\build\tetra_cli.exe`、実行commitは
`04d49ffee95f1c937c07ab60e88a868b6eb61ef6` である。

## smoke（本結果と分離）

目的はGPU bridge、checkpoint読込、protocol、4局展開の接続確認だけである。`pairs=1`,
`pieces=5` のため、4局すべてdrawとなった。

結果:

- result: `results/smoke/A-vs-S400-no-search-seed42-pairs1.json`
- games=4, A wins=0, S400 wins=0, draws=4
- device=AMD Radeon RX 9070 XT
- wall_seconds=0.471587700012606
- request_count=36
- child command:

```text
C:\Users\eddyf\czsmall_nnue\build\tetra_cli.exe gpu-arena-protocol 1 100000 5 16 1 1 42 0 0 -1 -1 0.01 0.05 -1 -1 -1 -1 -1 -1 -1 -1 1 8 2
```

state/trace/hangは同じ `results/smoke/` に保存し、200局結果へ混ぜていない。

## no-search policy control

実行した正確なwrapper command:

```powershell
.venv-rocm714\Scripts\python.exe experiments\search_throughput_vs_capacity_2026-08-18\capacity_quality\scripts\run_phase0.py --mode no_search --device cuda:1 --require-gpu --seed 42 --pairs 50 --pieces 300 --watchdog-seconds 300
```

記録:

- status=complete, exit_code=0
- wrapper elapsed_seconds=577.239778200048
- runner wall_seconds=569.044977399986
- result: `results/phase0/A-vs-S400-no-search-seed42-pairs50.json`
- record: `notes/run_records/A-vs-S400-no-search-seed42-pairs50.record.json`
- A wins=179, S400 wins=21, draws=0, games=200
- A score=0.895, Wilson 95% CI=[0.8448196800887696, 0.9302925120731899]
- S400 score=0.105, Wilson 95% CI=[0.0697074879268101, 0.1551803199112304]
- child command:

```text
C:\Users\eddyf\czsmall_nnue\build\tetra_cli.exe gpu-arena-protocol 50 100000 300 16 1 1 42 0 0 -1 -1 0.01 0.05 -1 -1 -1 -1 -1 -1 -1 -1 1 8 2
```

diagnosticsはA/S400とも simulations=0、raw_policy_matches=decisions、
searched_action_changes=0で、no-searchの意味を満たす。

## equal-node 64 control

実行した正確なwrapper command:

```powershell
.venv-rocm714\Scripts\python.exe experiments\search_throughput_vs_capacity_2026-08-18\capacity_quality\scripts\run_phase0.py --mode equal_node64 --device cuda:1 --require-gpu --seed 42 --pairs 50 --pieces 300 --watchdog-seconds 300
```

記録:

- status=complete, exit_code=0
- wrapper elapsed_seconds=4673.981166099955
- runner wall_seconds=4665.91753839998
- GPU inference seconds=3546.76050369861
- result: `results/phase0/A-vs-S400-equal-node64-seed42-pairs50.json`
- record: `notes/run_records/A-vs-S400-equal-node64-seed42-pairs50.record.json`
- A wins=189, S400 wins=11, draws=0, games=200
- A score=0.945, Wilson 95% CI=[0.9042130012276914, 0.9690146582965411]
- S400 score=0.055, Wilson 95% CI=[0.0309853417034590, 0.0957869987723090]
- child command:

```text
C:\Users\eddyf\czsmall_nnue\build\tetra_cli.exe gpu-arena-protocol 50 100000 300 16 1 1 42 -1 -1 -1 -1 0.01 0.05 -1 -1 -1 -1 -1 -1 64 64 1 8 2
```

`candidate_node_budget=64` と `champion_node_budget=64` は上記child command、
`tools/tetra_cli.cpp` の引数解析、`include/tetra/arena.hpp` のSearchConfig代入で
確認した。diagnosticsの主な値はA candidateが decisions=19,615 / nodes=1,255,199、
S400 championが decisions=18,883 / nodes=1,204,798で、両側のnode budget引数は64である。

## 失敗・再試行

- GPU gate failure、engine failure、child非ゼロ終了、watchdog timeoutはなかった。
- 200局本測定の再試行はない。各本測定は一回の完了runである。
- equal-node64の監視中、runnerが状態JSONを非原子的に書き換える瞬間に空内容または
  JSON parse不能として読めることがあった。再読込すると有効な状態へ戻り、request_count
  が増加していたため、これは監視読取競合であり、runの失敗・再試行ではない。
- smokeの短さによるdrawは接続確認の性質であり、本結果に混ぜていない。

## Stop A 判定

no-searchでS400は10.5%、equal-node64で5.5%だった。いずれも200局要件を満たし、
同規模S400がAに大敗したため、training-generation / 400-step Gen4 confoundを測る
Stop Aを成立と判定した。この後のM400/XS400測定、再学習、head-swap、throughput、
wall-clockは開始していない。

## 生ログ参照

各本測定について、同じstemの次のファイルを `results/logs/` に保存した。

- `.stdout.log`, `.stderr.log`
- `.requests.jsonl`, `.trace.jsonl`
- `.state.json`, `.hang.json`（hangなしでも出力先を記録）

機械可読な集計は `results/phase0/`、GPU gate・child command・hash・commitを含む
実行記録は `notes/run_records/` にある。
両条件のcandidate/reference双方のscore rate、Wilson CI、budget、runner条件、ログ参照、
Stop A判定をまとめた機械可読サマリーは `results/phase0/phase0_summary.json` にある。

## 最終検証

- `.venv-rocm714\Scripts\python.exe` で `capacity_quality/**/*.json` のJSON parseを実行し、12ファイルが成功。
- phase0結果の `wins + losses + draws == games == pairs * 4 == 200` を両条件で確認。
- 結果JSONのWilson CIを同じ式で再計算し、保存値と一致。
- no-search diagnosticsの両側で `simulations=0`、`raw_policy_matches=decisions`、`searched_action_changes=0` を確認。
- equal-node result command末尾が `64 64 1 8 2` であること、両側のnode diagnosticsが正であることを確認。
- seed manifest 50件を `uint64(42 + i * 0x9E3779B97F4A7C15)` で再計算し、全件一致。
- A/S400/M400/XS400のsha256をmanifestと再照合し、全件一致。
- 新規Python script 2本を `ast.parse` で構文確認。
- `git status --short --branch` と `git diff --ignore-space-at-eol --name-only` を再確認。既存tracked変更は不変で、今回の生成物は指定された `capacity_quality/` 配下に限定した。追加のstage/commit/pushは行っていない。
