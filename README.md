# TetraZero / TetraFormer

厳密かつ決定論的なTetrisルールコア、Leela系探索、自己対局による方策・価値学習基盤を一体化した研究用プロジェクトです。現在の基準モデルはTetraFormerで、CNNおよびCNN+Transformer系は、同じシミュレータ・同じ入出力契約の上で比較する実験モデルとして実装されています。

まず [`docs/README.md`](docs/README.md) のドキュメント案内を参照してください。`docs/SPEC.md` は**当初の設計仕様を保存する歴史的基準**であり、後から現在の実装に合わせて書き換えません。その後の設計変更は [`docs/adr/`](docs/adr/README.md) にADRとして記録します。

**本リポジトリはローカルシミュレータ専用です。** TETR.IOへ接続するネットワークコードは含みません。最初に [`docs/POLICY.md`](docs/POLICY.md) を確認してください。

**現在地:** ルールコア、合法手生成・トークナイズ、探索、自己対局、GPU学習・推論、Candidate/Champion Arenaまでの反復ループは実装済みです。現在の主要課題は、サンプル効率、モデル構造の比較、再現可能な自己対局カリキュラムであり、その後にタイミング学習、MoE、解釈可能性・人間学習支援を扱います。詳細は [`docs/ROADMAP.md`](docs/ROADMAP.md) を参照してください。

## クイックスタート

エンジン本体に必要なのは**C++23対応コンパイラだけ**です。外部依存、GPU、Pythonは不要です。合法配置の列挙には、Apache-2.0で同梱しているCobra movegenバックエンドを使用します。

```sh
make test         # 依存なしのテストスイート
make tools        # 開発用CLI
make test-asan    # AddressSanitizer + UBSanでも実行
```

学習パイプライン全体を一度に確認する場合:

```sh
./scripts/bootstrap.sh    # build -> test -> self-play -> train -> export -> play
```

個別に実行する場合:

```sh
./build/tetra_cli ruleset league                    # ルールセットとhash
./build/tetra_cli moves T 7                         # 合法配置を列挙
./build/tetra_cli timing T                          # 操作コストとdelay bin
./build/tetra_cli search 64 1 42                    # 単一局面の探索を確認
./build/tetra_cli record g.tetrarep 42 300          # 対局を記録
./build/tetra_cli verify g.tetrarep                 # 再シミュレーションして差分検証
./build/tetra_cli export train.tetradat 10 100 16   # 学習データを出力
./build/tetra_cli play models/gen1.tetrawts         # 学習済み重みで対局
./build/tetra_cli determinism 42                    # 再現性を検証
./build/tetra_cli bench 20000                       # movegen throughput

python trainer/train.py train.tetradat --steps 300 --save models/gen1.pt
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts

# optimizerとsampling RNGも含めてcheckpointから継続学習。
python trainer/train.py train.tetradat --resume models/gen1.pt --steps 5000 \
    --device cuda --require-gpu --checkpoint-every 1000 \
    --best-save models/gen1.best.pt --save models/gen1.cont.pt

# C++探索 + PyTorch/ROCm GPU推論。APM/APP/PPSを出力。
python trainer/gpu_match.py models/gen1.cont.pt --device cuda --games 4 \
    --pieces 200 --sims 32 --precision fp16 --workers 4

# Candidate / Championのpaired GPU Arena。
python trainer/gpu_arena.py models/candidate.pt models/champion.pt --device cuda \
    --pairs 10 --pieces 300 --sims 32 --precision fp16

# 現在のcheckpointで次世代の自己対局データを生成。
python trainer/gpu_selfplay.py models/gen1.cont.pt data/gen2.tetradat \
    --device cuda --games 32 --pieces 300 --sims 64 --model-version 2

# self-play -> replay mix -> train -> Arena -> 条件付きpromotionを1世代実行。
python trainer/iterate.py --champion models/champion.pt \
    --replay data/gen1.tetradat --generation 2 \
    --champion-output models/champion --device cuda
```

AMD GPUでの学習は、プロジェクト標準としてROCm 7.2以降を前提にしています。RX 9070 XTは `gfx1201` として扱います。詳細は [`docs/SETUP.md`](docs/SETUP.md) を参照してください。

## 実装済み機能

### M0 — ルールコア

| 領域 | 実装 |
|---|---|
| 盤面 | 行bitboard。占有セルとgarbage由来セルを別平面で保持 |
| ミノ | 7種、4回転状態、衝突maskを事前計算 |
| 回転 | Guideline SRS、**TETR.IO SRS+**（I-kickの左右対称化）、TETR.IO 180表 |
| Spin | T-spin 3-corner、All-Mini、**All-Mini+**、4方向immobile判定 |
| 攻撃 | `floor((base + b2b) × combo_mult)` + 固定bonus。公開TETR.IO表に準拠 |
| Combo | multiplier方式。base 0 clearには `floor(ln(1 + 1.25c))` |
| B2B | Charging（Surge時の固定+1）とChaining（oskのstep table） |
| Surge | streak 4から蓄積し、余りを先頭へ持ち越して3分割放出 |
| Garbage | travel time、activation delay、FIFO相殺、cap、messiness |
| Opener | 最初の14 placementsは相殺量を2倍 |
| Ruleset | 安定hash付き `RulesetConfig`。league / quickplay / guideline |
| 時刻 | 浮動小数点clockを使わず整数tickのみ |
| Handling | DAS / ARR / SDF / ARE / lock delayを行動時間へ反映 |
| Gravity | 到達可能性制約として整数・有理数で厳密に処理 |
| Replay | version付きbinary、checkpoint差分検証、約19 B/placement |

### M1 — movegenとモデル入力

- **Cobraを唯一の合法配置バックエンドとして使用。** `MoveList` と全target `PathFinder` により、kick、spin、tuck、wall climbを別の旧move generatorなしで列挙します。Cobraの固定10x40盤面が対応geometryであり、custom geometryではactionを返しません。
- **実時間付きaction。** ルールセットのhandling設定から最短経路のtick数を求め、総当たり検証で最適性を固定しています。
- **delay bin。** `FASTEST / +1F / +2F / +4F / +8F / WAIT_FOR_EVENT`。最後のものもgarbage activation、相手の次lock、lock delayで必ず上限を持ちます。
- 各actionに**canonical input sequence**を保持し、その入力列を再生して宣言した配置へ厳密に到達することをテストします。
- **観測mask**により、不可視状態が方策へ流れないようにします。
- **Row/Column tokenizer**（spec §9.2）。盤面を約400 cell tokenにせず、各盤面を `H` row + `W` column + summary tokenで表現します。
- **可変長action embedding**（spec §10.1）。
- bag残量・相手counterを含む現行token意味論はschema version/hashで識別します。単なるfeature width一致だけでは互換とみなしません。

### M2 — 探索・自己対局・学習

- **batch前提の `Evaluator` interface。** 実測でspec規模TetraFormer-Sが1局面約8.6 ms、movegenが約0.09 msだったため、batchingは後付け最適化ではなく構造上の前提です（[ADR 0007](docs/adr/0007-evaluator-interface-first.md)）。
- **PUCTとGumbel sequential halving。** virtual lossとtransposition tableを実装。分岐数に対してsimulation数が少ない領域ではPUCTが不安定なため、Gumbelを既定にしています（[ADR 0008](docs/adr/0008-search-gumbel-calibration.md)）。
- **root determinization**（spec §11.3）。preview以後のpieceを探索が読めてしまう情報漏洩を閉じています（[ADR 0009](docs/adr/0009-determinization-and-selfplay.md)）。
- **自己対局worker、replay buffer、`.tetradat` export。** 現行通常datasetはversion 3でschema・termination metadataを持ちます。単盤面で再構成可能なデータにはcompact Replay+π version 2も残しています（[ADR 0012](docs/adr/0012-compact-dataset-replay-pi.md)）。
- **TetraFormer** (`trainer/`)。pre-norm RMSNorm + SwiGLU、各合法actionがstate tokenへcross-attentionする可変長方策head、WDL価値head、補助回帰headを持ちます（[ADR 0010](docs/adr/0010-cpp-python-handover.md)）。
- **依存なしC++推論。** `.tetrawts`を直接読み、PyTorchとの最大差約 `3e-08` まで検証しているため、「学習したnetwork」と「C++が実際に指すnetwork」の乖離をテストで検出できます（[ADR 0011](docs/adr/0011-cpp-inference-without-onnx.md)）。
- **GPU match bridge** (`trainer/gpu_match.py`)。ゲーム・探索はC++、network評価だけPyTorch/ROCmでbatch処理し、APM/APP/PPSを報告します。
- **GPU self-play** (`trainer/gpu_selfplay.py`)。C++側をルール・探索・dataset生成の権威に保ったままGPU評価を使います。二盤面自己対局では両player視点と相手event streamが必要なため、compact v2へ無理に落とさずrectangular v3として保持します。
- **GPU Arena** (`trainer/gpu_arena.py`)。paired game、mirror seed、二盤面attack delivery、promotion計算はC++が所有し、Candidate/Champion checkpointの評価のみGPUで行います。
- **再開可能checkpoint。** modelだけでなくoptimizerとsampling RNGも保存します。WDL価値headは既定で学習し、accuracyとscalar MSEも記録します。
- **replay mixとguard付きiteration。** `train.py` は複数generationと `--new-data-repeat` を受け取り、`trainer/iterate.py` がGPU自己対局・学習・Arena・条件付きpromotionを接続します。
- **sample-efficiency contract。** 現行schema v2の補助目標は36次元で、未知未来をmaskし、方策・価値・補助loss間のshared-trunk gradient norm/cosineも観測できます。

過去の校正実験では、garbage stream下でGumbel-32が250/250 placementsへ到達し、policy-onlyは平均228.7でした。engine生成データ上では初期のheld-out total lossが4.86 → 2.91へ低下しました。これらは実装・学習経路の成立を示す過去の測定であり、現在のChampion強度そのものを表す数値ではありません。

モデルはまだ最終到達点ではありません。強さは自己対局データの量・多様性、探索教師、学習条件に依存します。現在の比較方針は、検証用lossだけでモデル構造を選ばず、paired Arenaと探索下の実戦性能を重視することです（[ADR 0013](docs/adr/0013-architecture-ablation-and-local-geometry.md)）。

## 検証

2026-08-10時点の通常 `make test` では **282 tests / 1,045,724 assertions / 0 failed** を確認しています。重要な不変条件は次のとおりです。

- **決定性** — 同一seedのreplayはgarbage込みでbit-identical。RNG状態もsnapshot/restore可能。
- **鏡映関係** — 回転結果・合法配置集合をrandom board上でcell単位に検査。ただしclassic SRS IおよびTETR.IO 180の既知の非対称性は別途固定。
- **情報漏洩防止** — hidden garbage holeを変えてもtokenが同一であり、探索はpreview以後のpieceへ依存しない。
- **attack保存則** — 送信attackは相殺されるか着弾するかのどちらかで、消失しない。
- **PyTorch parity** — C++ forwardとPython forwardの一致をfixtureで検証。
- **dataset contract** — schema、termination reason、aux mask、二盤面tokenのround-tripを検証。

`tests/data/` のbinary fixtureは再生成可能です。

```sh
python scripts/make_fixtures.py
```

fixtureがない環境では該当するparity testがskipされますが、残りのテストは実行されます。

### ルール実装から得られた重要な知見

これらはproperty testで発見され、学習時のaugmentationにも影響するため専用テストで固定しています。

1. **SRS+のI-kickはentry単位で完全に左右対称**です。これがguideline SRSとの差の1つです。
2. **classic SRSは完全対称ではありません。** I pieceのkick順序が左右で異なるため、guideline SRSではI配置の左右反転を無条件augmentationにできません。他pieceは対称です。
3. **immobile spin判定には上方向checkも必要です。** left/right/downだけでは平坦なnotch上のpieceを大量に誤spin判定し、過去の300-piece gameではspin数が16から198へ膨らみました。
4. **同じcell集合でも座標表現が異なる場合があります。** I pieceのrotation N at `y` とrotation 2 at `y+1` が同一cellを埋めることがあり、merge時はinput列だけでなく実行全体を置き換えないと自己矛盾します。
5. **TETR.IO 180 kick tableは意図的に非対称**です。180有効時の一部位置はmirror-equivalentではありません。

## 性能

movegenは探索の内側で繰り返し呼ばれます。標準10x40盤面・20,000 callsの同一CLI benchmarkでは次の結果でした。

| 実装 | µs / call |
|---|---:|
| pure Cobra移行前のhybrid adapter | 132.4 |
| pure Cobra (`MoveList` + all-target `PathFinder`) | **99.0** |

同じbenchmarkでは約**25.2%高速化**しています。プロジェクト側のtiming layerはCobraが返したpathを価格付けし、canonical inputとdelay-bin contractを維持します。

## 継続的インテグレーション

CI workflowのひな形は [`docs/ci.yml`](docs/ci.yml) にあります。g++ / clang++ build、sanitizer、決定性検証、およびwarning-as-error確認を想定しています。

ただし**2026-08-10時点では、そのまま有効化して全jobがgreenになる状態ではありません。** 通常 `make test` は282 testsでpassしますが、`-Werror` jobはvendored Cobra内部の `#pragma unroll` とshadow warningをerrorへ昇格させて失敗します。project codeのwarning policyとthird-party codeのwarningを分離する方法を決めてからworkflowを有効化してください。

現在 `.github/workflows/` には置かれていません。有効化するときは、上記warning scopeを修正したうえで移動します。

```sh
mkdir -p .github/workflows && git mv docs/ci.yml .github/workflows/ci.yml
```

## 構成

```text
include/tetra/
  types.hpp          piece、rotation、clear descriptor、tick型
  bitboard.hpp       occupancy + garbage planeを持つ盤面
  pieces.hpp         shapeとSRS / SRS+ / 180 kick table
  ruleset.hpp        version付きRulesetConfig
  rng.hpp            xoshiro256** + 7-bag randomizer
  piece_state.hpp    collision、kick付きrotation、spin判定
  attack.hpp         attack table、combo、B2B、Surge
  garbage.hpp        pending queue、相殺、activation
  events.hpp         bounded event log
  player.hpp         1 player分のsimulation
  timing.hpp         handling、action cost、gravity、delay bin
  movegen.hpp        Cobra-backed合法配置生成
  observation.hpp    observation mask
  tokenizer.hpp      Row/Column token + action embedding
  replay.hpp         replay format / recorder / verifier
  evaluator.hpp      batch policy/value interface + baseline
  gpu_evaluator.hpp  C++/PyTorch GPU evaluator bridge
  search.hpp         PUCT / Gumbel + virtual loss + TT
  replay_buffer.hpp  training sample + replay buffer
  selfplay.hpp       自己対局worker
  batch.hpp          padded tensor + mask
  dataset.hpp        .tetradat serialization
  schema.hpp         tokenizer / observation / action / aux schema
  nnue.hpp           C++ trained-weight inference
src/                 ruleset hashとCobra adapter実装
tests/               dependency-free test harness
tools/               tetra_cli
trainer/
  tetraformer.py     TetraFormer
  ablation_models.py CNN / hybrid実験モデル
  tetra_dataset.py   .tetradat reader + validation
  train.py           再開可能trainer
  gpu_match.py       GPU match runner
  gpu_selfplay.py    GPU自己対局dataset生成
  colab_generate.py  再現可能Colab shard launcher + validator
  gpu_arena.py       GPU Candidate-vs-Champion Arena
  iterate.py         generation/replay/Arena driver
  export_weights.py  PyTorch checkpoint -> .tetrawts
scripts/
  bootstrap.sh       end-to-end確認
  make_fixtures.py   binary fixture再生成
docs/                文書案内、設計、学習protocol、ADR、原仕様
```
