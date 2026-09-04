# 追加診断コードマップ

## 実行経路

- `experiments/search_throughput_vs_capacity_2026-08-18/scripts/run_condition.py`
  は `gpu-arena-protocol` の子プロセスを起動し、要求フレームを
  `trainer/gpu_match.py` の既存デコーダーで受け、候補BとChampion Aを同じ
  Pythonプロセス内で推論する。診断時は要求ごとの形状、実バッチ、テンソル構築、
  推論、同期、応答書き込み、総時間をJSONLへ保存する。
- `trainer/gpu_match.py::_serve_request_batch` は通常のモデル推論に加えて、
  `diagnostic_evaluator=protocol-dummy` のときは同じ要求・応答プロトコルを通し、
  モデル前向き計算だけを省略する。
- `trainer/gpu_arena.py` は通常のArena呼び出し経路である。子プロセスの標準エラー
  出力を専用スレッドで常時排出し、結果フレーム受信後の終了待ちでパイプが詰まらない
  ようにした。

## C++側の計測点

- `include/tetra/search.hpp` は決定ごとのルート準備、探索木の収集、評価器、バックアップ、
  最終化の時間を記録する。`gather` は合法手生成・状態遷移などを含む広い区間であり、
  `node_allocation_us`、`legal_action_generation_us`、`state_transition_us`、
  `selection_us` はその内側の診断値で、足し上げてはいけない。
- `include/tetra/arena.hpp` は候補側・Champion側に集約し、
  `tools/tetra_cli.cpp` はArenaのバイナリ結果を標準出力へ、診断JSONを標準エラー
  出力へ出す。この標準エラー出力が長尺実行で数KBに達する。
- `tools/tetra_cli.cpp::arena-diagnostic` は同じC++ Arena/SearchをUniform/Heuristic
  評価器で動かす局所ダミー診断であり、強さ比較には使わない。

## 診断スクリプト

- `scripts/diagnose_decision_timing.py`: Arena診断JSONとPython JSONLを突合し、
  per-decision内訳、per-request橋渡し時間、実バッチ分布をCSV/JSONへ集約する。
- `scripts/diagnose_hang.py`: 別プロセスの状態ファイル更新を監視し、停止時に最後の状態、
  直近トレース、子プロセスの終了結果を保存する。Windows版Pythonでは
  `faulthandler.register` が利用できないため、SIGBREAKによるPythonスタック取得は
  補助的な試行に留まる。
- `scripts/diagnose_local_dummy.py`: E3 10/40/160 msとE1 40 msの局所ダミー行を生成する。
- `scripts/summarize_diagnosis.py`: 全診断結果を一つの機械可読JSONと人間向けMarkdownへまとめる。
