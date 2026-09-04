# Search Throughput 実験・性能異常の原因究明手順書

作成日: 2026-08-18
対象: `experiments/search_throughput_vs_capacity_2026-08-18/`

## 0. 目的

このタスクの目的は、前段の A/B 実験で観測された次の2つの異常を、推測ではなく計測で切り分けることである。

1. **評価器単体では B が A より大幅に高速なのに、40 ms / 160 ms の実ゲーム探索では A/B の nodes/decision がほぼ同じになる。**
2. **長尺 GPU Arena が、数十〜数百 request 後に結果フレームを返さず停止する。**

本タスクでは強さ比較をやり直さない。200局 Pilot、2,000局 Main、モデル再学習、モデル追加縮小は、原因究明と安定化が完了するまで行わない。

最終目的は、次の問いに証拠付きで答えることである。

> A/B の evaluator throughput 差が、40 ms / 160 ms の探索量差へ変換されないのはなぜか。

および

> 長尺 Arena は、GPU forward、GPU request/response bridge、C++ search、Arena protocol、batch scheduler のどこで停止しているのか。

---

## 1. 現在までに確定している観測事実

以下を前提として扱う。再解釈して書き換えない。

### 1.1 evaluator microbenchmark

- A: `models/gen14_rank100_100_20260814.best.pt`
- B: `models/size_search_ablation_20260816/seed42/transformer_xs.final.pt`
- A parameters: 7,176,624
- B parameters: 132,744
- device: RX 9070 XT, `cuda:1`
- precision: fp16

測定済み throughput:

| batch | A states/s | B states/s | B/A |
|---:|---:|---:|---:|
| 1 | 119.2 | 297.3 | 2.494x |
| 16 | 1,786.6 | 4,764.6 | 2.667x |
| 64 | 6,302.5 | 17,850.4 | 2.832x |
| 128 | 7,039.7 | 32,632.2 | 4.635x |
| 256 | 5,981.9 | 44,564.3 | 7.450x |

したがって、**「B が十分高速でない」ことを、40/160 ms で nodes 差が消える説明として採用してはならない。**

### 1.2 完走した短尺 Arena

| environment | budget | max pieces | B nodes/decision | A nodes/decision | B/A nodes |
|---|---:|---:|---:|---:|---:|
| E3 | 10 ms | 5 | 17.0 | 1.9 | 8.95x |
| E3 | 40 ms | 20 | 39.5 | 39.9 | 0.99x |
| E3 | 160 ms | 5 | 141.2 | 139.8 | 1.01x |
| E1 | 40 ms | 20 | 78.5 | 43.1 | 1.82x |
| E2 | 40 ms | 20 | 41.1 | 39.4 | 1.04x |

特に、40 ms E3 と 160 ms E3 がほぼ同一の `nodes/ms` に収束している点を重要な異常として扱う。

### 1.3 長尺停止

- 300 pieces / 4 games / 10 ms / batch 128 が約13分後も Arena result frame を返さなかった。
- batch 32 / batch 16 の長尺でも同様の no-result が発生。
- request trace では token width 約84〜103、action count 約9〜75。
- 未知 shape の direct probe は初回約4.6〜4.7秒、その後約9〜11 ms。
- fixed padding 128 token / 128 action の5-piece smokeは完走。
- しかし fixed shape の長尺でも後段の batch 2 request で停止。

したがって、**可変 shape による ROCm 初回 setup は実在するが、長尺停止の唯一の原因だと決め打ちしてはならない。**

---

## 2. このタスクの非目標

以下は行わない。

- A/B checkpoint の再学習。
- B の追加小型化。
- MoE 等の新 architecture 導入。
- Gumbel/PUCT/Search family の変更。
- exploration constant の tuning。
- 200局 Pilot / 2,000局 Main の再開。
- Arena 勝率から A/B 強度を判定すること。
- GPU utilization を上げること自体を目的にした最適化。
- 症状を隠すためだけの timeout / retry の追加。

このタスクでは、**原因を観測し、最小修正で再現性ある安定経路を作るところまで**を扱う。

---

## 3. 最重要原則

### 3.1 修正より先に観測

原因を推測して先に大規模修正してはならない。

まず現行経路へ計測を追加し、A/B、10/40/160 ms、E1/E2/E3 で時間がどこへ消えているかを取る。

### 3.2 「GPU inference time」1項では不十分

現在の `run_condition.py` は `_serve_request_batch(...)` 全体を `gpu_inference_seconds` としている。この値だけでは、以下を分離できない。

- request decode
- NumPy -> Tensor conversion
- host-to-device
- queue / batch wait
- model forward
- synchronize
- device-to-host
- output decode
- protocol writeback

これらを可能な限り分ける。

### 3.3 Hang は必ず最後の状態を記録してから停止

単に「何分待っても返らなかった」で終えない。

watchdog timeout 時に、少なくとも最後に完了した request / 現在処理中 request / child process 状態 / Python thread stack / request shape / batch size を保存する。

### 3.4 A/B で instrumentation を変えない

比較可能性を保つ。A/B とも同一コードパス、同一ログ項目を使う。

---

## 4. 仮説一覧

以下を候補として扱い、どれかを先に正解とみなさない。

### H1: evaluator 以外の CPU search overhead が支配している

候補:

- tree selection
- sequential halving
- node allocation
- backup
- legal action generation
- state cloning
- move application
- garbage/event simulation

予測:

- 40/160 ms では evaluator time fraction が小さい。
- A/B の forward 差が total decision time にほぼ影響しない。
- E1 では simulator/opponent interaction が軽いため B の差が比較的残る可能性。

### H2: GPU request batch が実際にはほぼ 1〜2 で、microbenchmark の大 batch が利用されていない

予測:

- actual request batch histogram が1〜2へ集中。
- B/A forward ratio は実ゲーム中だと batch1 相当の約2.5x程度。
- 大 batch 用の理論 throughput は探索へ伝播しない。

ただし、これだけでは 40/160 ms で nodes 差がほぼ1.0xになることを完全には説明しない可能性がある。

### H3: batch fill / synchronization barrier が budget を消費する

予測:

- evaluator forward より queue/batch wait が大きい。
- A/B 共通の待ち時間が存在する。

### H4: wall-clock budget の実装に、ほぼ 1 node/ms の別上限がある

候補:

- 1 ms sleep / poll
- timer granularity
- per-node throttling
- request-response roundtrip serialization
- sequential protocol
- implicit synchronization

予測:

- 40 ms で約40 nodes、160 ms で約140〜160 nodes というほぼ線形上限。
- evaluator を dummy / instant evaluator に置換しても nodes/ms が大きく増えない。

### H5: Python <-> C++ protocol roundtrip が律速

予測:

- request read/write、serialization、pipe wait が大きい。
- GPU forward は短いが、1 request あたり protocol overhead が一定。
- batch 1〜2で特に悪化。

### H6: ROCm shape setup / kernel compilation が長尺停止へ寄与

予測:

- 未知 shape 初回のみ長い。
- fixed shape / full prewarm で shape由来 stall は消える。

ただし fixed shape でも停止しているため、H6単独説は否定寄り。

### H7: GPU worker / bridge が特定 request 後に deadlock または応答不能になる

予測:

- request N は受信済みだが response complete がない。
- Python stack が model forward / synchronize / pipe write の特定箇所で止まる。

### H8: C++ child 側が request を出さず search/Arena 内部で停止する

予測:

- Python側は次の frame read 待ち。
- child process は alive。
- 最後の request は正常完了済み。
- C++ stack / progress heartbeat が search や game loop の特定箇所を指す。

### H9: Arena result protocol / stderr pipe / stdout pipe が詰まる

候補:

- stderr 未読による pipe saturation
- result frame write blockage
- Python/C++ frame read mismatch

これは必ず検証すること。現在 `stderr=PIPE` を Arena 終了まで読み出していないため、**stderr pipe saturation は優先度の高い候補**として明示的に除外する。

---

## 5. Phase 1: 1 decision の完全な時間分解

### 5.1 追加する診断ログ

各 decision ごとに、可能な限り以下を nanosecond/µs 精度の monotonic clock で記録する。

```text
decision_total_us
search_selection_us
state_transition_us
legal_action_generation_us
node_allocation_us
inference_submit_us
inference_queue_wait_us
request_protocol_write_us
request_protocol_read_us
host_tensor_build_us
host_to_device_us
model_forward_us
device_sync_us
device_to_host_us
output_decode_us
backup_us
other_us
```

既存構造上、一部の区間を厳密に分離できない場合は無理に値を捏造しない。

その場合は、より粗い区間へまとめて `notes/DIAGNOSIS_DEVIATIONS.md` に理由を書く。

### 5.2 decision ごとの count

必ず次も記録する。

```text
model_id
request_count
requested_positions_total
actual_batch_size histogram
leaf_evals
expanded_nodes
selected_nodes
max_depth
mean_depth
legal_actions_root
root_visits
budget_ms
budget_overshoot_ms
```

### 5.3 shape 情報

各 inference request に:

```text
request_id
game_id
decision_id
model_id
batch_size
token_count
action_count
shape_seen_before
shape_first_seen_index
request_received_at
forward_started_at
forward_finished_at
response_written_at
```

を記録する。

---

## 6. Phase 2: Python GPU bridge の内訳計測

`trainer/gpu_match.py` の `_serve_request_batch` を読むこと。

大規模に書き換える前に、診断モードで以下を分離計測する。

1. NumPy request -> torch tensor 生成。
2. `.to(device)` / device transfer。
3. autocast setup。
4. actual model forward。
5. `torch.cuda.synchronize` を使う場合その時間。
6. logits/value の device -> host。
7. protocol response encode。
8. stdin write + flush。

GPU forward 自体の実時間を計る箇所では、非同期実行を誤って短く測らないこと。

必要なら CUDA event 相当または同期前後の wall clock を使う。

通常経路の性能を大きく変えないよう、詳細 timing は `--diagnostic-trace` 等の明示フラグ時だけ有効にする。

---

## 7. Phase 3: batch の実態を確認

microbenchmark の batch128/256 は、実ゲームでその batch が形成されて初めて意味を持つ。

したがって、実ゲーム中の **actual batch size histogram** を最優先で出す。

A/B × 10/40/160 ms × E1/E2/E3 について最低50 decisions取得し、次を表にする。

```text
batch size
request count
positions count
fraction of requests
fraction of positions
mean forward us
p95 forward us
```

特に batch 1, 2, 4, 8, 16 の比率を明示する。

もし batch 1〜2 が大部分なら、「batch128で7.45x」を探索 throughput の期待値として使わない。

---

## 8. Phase 4: dummy evaluator による探索上限測定

これは重要な診断である。

モデル A/B とは別に、**GPU model forward を実質ゼロにした診断専用 evaluator** を作る。

条件:

- search/action semantics を壊さない。
- policy/value は固定値または deterministic な簡易値でよい。
- 強さ評価には絶対に使わない。
- inference protocol は可能なら同じ経路を通す版と、C++内で即時応答する版の2つを比較する。

目的は「評価器を完全に無料にしても nodes/ms が約1のままか」を測ること。

### D1: protocol dummy

Python bridge / pipe / request-response は維持し、model forward だけを即時固定出力にする。

### D2: local dummy

可能なら C++ 内で評価を即時返し、Python/GPU bridge 自体を除外する。

比較:

```text
real A
real B
protocol dummy
local dummy
```

10/40/160 ms で nodes/decision と nodes/ms を取る。

解釈:

- protocol dummy でも約1 node/ms -> model forward は主因ではない。
- local dummy で大幅改善、protocol dummyで改善しない -> protocol/bridge が律速。
- protocol dummyもlocal dummyも大幅改善 -> evaluator path に主因。
- local dummyでも改善しない -> C++ search/simulator/budget loop 側が律速。

この診断は勝率を一切解釈しない。

---

## 9. Phase 5: Hang watchdog

長尺 run は watchdog なしで実行してはならない。

### 9.1 watchdog の条件

最低2種類用意する。

- request timeout: 1 request が N 秒以上完了しない。
- progress timeout: request completion / decision completion のどちらも N 秒以上進まない。

初期値は 10秒程度でよい。未知 shape 初回4.7秒を誤検知しない値にする。

### 9.2 timeout 時に保存するもの

プロセス kill より先に次を保存する。

```text
wall time
last completed request id
current request id
last completed decision id
current game id
model id
batch size
token/action shape
shape seen before
last forward start/end
last response write
child pid
child alive/dead
child return code if any
Python thread stacks
outstanding requests
queue length
last 100 trace events
```

可能なら child process の native stack trace も取得する。

### 9.3 Python stack

`faulthandler.dump_traceback()` 等を利用し、全Python thread stackをファイルへ保存する。

### 9.4 C++ progress heartbeat

C++側に診断モード限定で軽量 heartbeat を追加し、最低限:

```text
game_id
decision_id
phase
search_nodes
last_inference_request_id
```

を一定間隔または主要 phase transition ごとに stderr 以外の安全な診断先へ出す。

---

## 10. Phase 6: stderr pipe saturation を明示的に除外

現行 `run_condition.py` は:

```python
stderr=subprocess.PIPE
```

で child を起動し、Arena 終了まで `proc.stderr.read()` しない。

child が長尺中に十分な stderr を出す場合、OS pipe buffer が埋まり child が stderr write で停止する可能性がある。

この仮説は必ず最初期に検証する。

### 比較条件

同一短〜中尺条件で以下を比較する。

1. 現行: stderr PIPE を最後まで未読。
2. stderr を専用 reader thread で常時 drain。
3. stderr を診断 log file へ直接流す。
4. stderr を `DEVNULL` にする診断条件。

結果:

- 2〜4で長尺停止が消えるなら、pipe saturation を主要原因と判定できる。
- 消えなくても、この候補を除外できる。

**ログを消すことを恒久対策にしない。恒久対策は常時 drain か適切な logging path にする。**

---

## 11. Phase 7: Shape 仮説の再検証

可変 shape の初回4.6秒遅延は再現済みなので、次を行う。

### S1: variable shape, no prewarm

現行に近い。

### S2: variable shape, observed shapes prewarm

既存 request log から観測済み token/action shape を列挙し、開始前に prewarm。

### S3: fixed 128/128 padding

既に試した経路を再利用。

### S4: fixed 128/128 + batch sizes 1,2,4,8,16 prewarm

fixed shape でも batch 次元変化が別 compilation/setup を起こす可能性があるため、batch dimension も明示的に warmup する。

比較対象:

- first request latency
- later request latency
- hang occurrence
- hang request shape/batch
- total nodes/ms

fixed shape で後段 batch2停止が再現するなら、shape-only 仮説は棄却方向とする。

---

## 12. Phase 8: A/B の40/160 ms異常を最小再現する

長尺 Arena を使う必要はない。

1 decision または5 decisionsだけを固定 state から繰り返す診断 harness を優先して作る。

同じ state、同じ legal actions、同じ seed で:

```text
A 10 ms
B 10 ms
A 40 ms
B 40 ms
A 160 ms
B 160 ms
```

を最低50 repetitions。

記録:

- nodes
- requests
- actual batches
- forward time
- search CPU time
- protocol time
- overshoot

ゲーム進行による state 差を消すことで、モデル差が nodes差へ伝播しない原因を直接見る。

---

## 13. Phase 9: E1 / E2 / E3 差の説明

E1 40 ms では B/A nodes が約1.82x残っている一方、E2/E3ではほぼ1.0xである。

これは重要なので、E1/E2/E3 の per-decision timing breakdown を比較する。

特に:

- opponent state update
- garbage scheduling
- cancellation
- event queue
- additional legal action generation
- state copy size

の時間差を見る。

E2/E3で追加されるCPU/simulator処理が40 msを支配しているなら、H1を支持する。

---

## 14. 実行マトリクス

### Stage A: 最小診断

各条件50 decisions程度。

```text
Evaluator: A, B, protocol-dummy, local-dummy
Budget: 10, 40, 160 ms
Environment: E1, E3
```

まずこれで十分。

### Stage B: interaction diagnostics

Stage Aで探索側律速が見えた場合:

```text
A/B
40 ms
E1/E2/E3
100 decisions
```

### Stage C: Hang reproduction

長尺は強さ比較ではなく停止再現用。

```text
4 games
300 pieces
10 ms
batch 16
fixed seed
```

以下を1つずつ変える。

```text
stderr drain on/off
fixed shape on/off
batch prewarm on/off
GPU worker restart on/off
```

一度に複数要因を変更しない。

---

## 15. GPU worker restart の扱い

一定 Arena block ごとに GPU worker を再起動する方式は、診断候補としてはよいが、最初から恒久対策にしない。

もし restart で hang が消える場合、

- GPU runtime state leak
- request queue leak
- Python object accumulation
- ROCm cache/state accumulation

等の可能性がある。

restart interval sweep:

```text
1 game
2 games
4 games
never
```

を比較し、停止までの request 数と相関を見る。

---

## 16. メモリリーク確認

長尺停止の前に CPU RAM / VRAM / Python object / tensor 数が単調増加していないかを見る。

最低限10〜30秒間隔で:

```text
process RSS
GPU allocated memory
GPU reserved memory
request count
game count
```

をログする。

可能なら `torch.cuda.memory_allocated()` / `memory_reserved()` を記録する。

VRAM/RAM が request 数に比例して増える場合、別途リークとして扱う。

---

## 17. 診断結果の保存先

同じ実験ディレクトリ内に以下を追加する。

```text
notes/DIAGNOSIS_CODEMAP.md
notes/DIAGNOSIS_DEVIATIONS.md
results/diagnosis/
results/diagnosis/traces/
results/diagnosis/hangs/
results/diagnosis/timing_summary.csv
results/diagnosis/timing_summary.json
scripts/diagnose_decision_timing.py
scripts/diagnose_hang.py
scripts/summarize_diagnosis.py
```

既存成果物を上書きしない。

---

## 18. Trace format

JSONLを推奨する。

例:

```json
{
  "event": "inference_complete",
  "run_id": "...",
  "game_id": 0,
  "decision_id": 17,
  "request_id": 431,
  "model_id": 1,
  "batch": 2,
  "tokens": 128,
  "actions": 128,
  "tensor_build_us": 91.2,
  "h2d_us": 73.0,
  "forward_us": 2130.4,
  "sync_us": 31.7,
  "d2h_decode_us": 66.5,
  "response_write_us": 14.8,
  "timestamp_monotonic_ns": 1234567890
}
```

C++側 trace と Python側 trace は `request_id` / `decision_id` で突合できるようにする。

---

## 19. 成功判定

### 19.1 探索 throughput 異常

以下のいずれかを証拠付きで特定できれば成功。

例:

- 40/160 msでは search CPU処理が80%以上を占め、evaluator差が隠れる。
- actual batchがほぼ1であり、さらにprotocol overheadが主要時間を占める。
- C++側のbudget loopに約1 node/msの上限が存在する。
- Python/C++ roundtripが1 nodeごとに直列化されている。

「多分CPU側」という結論だけでは不十分。

どの関数/区間が何%を使うかまで示す。

### 19.2 Hang

次のどこで停止したかを特定できれば成功。

```text
A. Python stdout frame read待ち
B. Python request decode
C. tensor build / transfer
D. model forward / GPU synchronize
E. response write
F. C++ inference response read
G. C++ search
H. C++ game loop
I. stderr/stdout pipe write
J. result frame write/read
```

少なくとも大分類を1つに絞る。

### 19.3 安定化

原因修正後、最低条件:

- 300 pieces × 4 games を同じseedで3回連続完走。
- watchdog timeout 0。
- result frameを毎回取得。
- traceに欠損requestがない。
- `make test` 成功。

この時点で初めて200局 Pilot再開を推奨する。

---

## 20. 修正の優先順位

原因が分かった後のみ修正する。

優先度は:

1. deadlock / pipe saturation / protocol bug
2. request lifecycle bug
3. GPU runtime / shape setup instability
4. needless synchronization
5. protocol roundtrip overhead
6. batch formation inefficiency
7. C++ search CPU bottleneck

モデル architecture は最後まで触らない。

---

## 21. 重要な判定例

### ケースA

- B forward = Aの2.5x高速
- total decisionでは forward が10%
- search/simulator が80%
- nodesは同程度

結論:

> 評価器の高速化余地は存在するが、現在の40/160 ms探索では evaluator が律速ではない。search/simulator側を最適化しない限り、Bの速度差は探索量へ変換されない。

### ケースB

- actual batch = 1〜2中心
- pipe roundtrip = 0.7 ms/request
- forward = 0.2 ms/request
- 約1 request/node

結論:

> protocol serialization が nodes/ms の上限を形成している。大batch microbenchmarkの優位は現行探索では利用できていない。

### ケースC

- hang時Pythonは `read_exact(proc.stdout, 4)` 待ち
- child stackは `fprintf(stderr, ...)` / stderr write
- drain thread有効で完走

結論:

> stderr pipe saturation が長尺停止の原因。

### ケースD

- hang時 request受信済み
- forward_started あり、forward_finished なし
- Python stackは synchronize/model kernel 内
- fixed shape / prewarmでも再現

結論:

> GPU runtime/model execution path 側の停止。shape-onlyではない。

---

## 22. Codex 作業順序

この順番を守る。

1. `git status` と既存 dirty worktree を記録。
2. `AGENTS.md` を再確認。
3. 既存 `README.md`, `PLAN.md`, `notes/CODEMAP.md`, `notes/DEVIATIONS.md` を読む。
4. `run_condition.py`, `trainer/gpu_match.py`, `trainer/gpu_arena.py`, C++ `gpu-arena-protocol` 実装を読む。
5. stderr pipe saturation の再現/除外を最初に行う。
6. Python request lifecycle timing を追加。
7. C++ decision/search timing を追加。
8. 1 decision固定state診断を作る。
9. A/B 10/40/160 ms を計測。
10. actual batch histogram を生成。
11. protocol dummy / local dummy を比較。
12. E1/E2/E3 40 msを分解。
13. watchdogを実装。
14. 長尺 hang を再現し、stack/stateを保存。
15. shape/prewarm/restartを1要因ずつ比較。
16. 原因を1つ以上特定。
17. 最小修正。
18. 300 pieces × 4 games × 3連続完走を確認。
19. `make test` / script syntax / `git diff --check`。
20. `results/diagnosis/` と `README.md` または `notes/` に結論を保存。

---

## 23. 既存 dirty worktree の扱い

前回と同じ。

- reset禁止。
- clean禁止。
- stash禁止。
- unrelated file削除禁止。
- commit/stage禁止。

今回の診断変更だけを追跡可能にする。

必要な既存ファイル変更は最小限にする。

---

## 24. 最終報告で必ず答える質問

最終報告は次の順で答える。

1. **40 ms E3で、1 decisionの時間は何に何%使われていたか。**
2. **160 ms E3でも同じか。**
3. **A/B forward時間差は実ゲーム中に何倍だったか。**
4. **actual batch size distribution はどうだったか。**
5. **なぜ40/160 msでA/B nodesがほぼ一致したのか。**
6. **E1だけBのnodes優位が残った理由を説明できるか。**
7. **dummy evaluatorでnodes/msはいくつまで上がったか。**
8. **protocolを除外するとnodes/msは上がったか。**
9. **長尺Arenaはどこで停止していたか。**
10. **stderr pipe saturationは原因だったか。**
11. **可変shapeは主因、寄与因、無関係のどれか。**
12. **fixed shape + batch prewarmでも停止したか。**
13. **修正後に300 pieces × 4 gamesを3回連続完走したか。**
14. **200局Pilotを再開してよい状態か。**

---

## 25. 最終的な受入基準

この診断タスクは、単にArenaが一度完走しただけでは完了としない。

最低限:

- [ ] stderr pipe saturationを検証済み。
- [ ] per-decision timing breakdown取得済み。
- [ ] per-request GPU bridge timing取得済み。
- [ ] actual batch histogram取得済み。
- [ ] A/B 10/40/160 ms固定state診断済み。
- [ ] protocol dummy比較済み。
- [ ] local dummy比較済み、または実装不能理由明記。
- [ ] E1/E2/E3の40 ms timing差を比較済み。
- [ ] watchdogがhang時stack/stateを保存できる。
- [ ] hangの停止区間を大分類1つまで絞った。
- [ ] shape-only仮説を再検証済み。
- [ ] 最小修正後300 pieces × 4 gamesを3回連続完走。
- [ ] 既存test成功。
- [ ] 既存A/B checkpoint未変更。
- [ ] 200局Pilot再開可否を明記。

---

## 26. 最後の注意

今回最も避けるべき誤りは、

> Bはmicrobenchmarkで速いがArenaでは速くないので、軽量評価器は意味がなかった

と早合点することである。

逆に、

> Bはmicrobenchmarkで7.45倍速いので、本来Arenaでも7.45倍探索できるはず

とみなすのも誤りである。

必要なのは、**microbenchmark throughput → request batching → GPU bridge → search loop → nodes/decision** の各段階で、速度差がどこまで伝播し、どこで失われるかを定量化することである。

本診断が完了するまで、Search Throughput vs Evaluator Capacity の本来の仮説検定は保留する。
