# GPU学習・推論を実行する

GPU学習、自己対局、Arenaを実行するときの手順です。コマンド例は、リポジトリのルートからLinux/WSL2のシェルで実行します。パスと実験条件は、使用するcheckpointとdatasetに合わせて指定してください。

## 実行前に確認する

1. 環境の導入とGPU検出が済んでいることを確認します。未導入の場合は[セットアップ](SETUP.md)を参照します。
2. `git status` で実行するcommitを確認します。engine、checkpoint、dataset、manifestのcommitとschemaが互換であることを確認します。
3. `make test` が成功することを確認します。失敗している状態で学習を開始しません。
4. 比較実験では、[学習・評価プロトコル](TRAINING_AND_EVALUATION.md)と対象実験のplan/handoffに従い、games、pieces、sims、seed、precision、splitを固定します。

GPU作業の前提は次のとおりです。

- AMD GPUでも、PyTorchのデバイス指定は `cuda` です。`trainer/train.py` はGPUが見えないとCPUへフォールバックするため、GPU学習では `--device cuda --require-gpu` を指定します。
- PyTorch側の学習、GPU自己対局、GPU Arenaは `.pt` を読みます。C++のCPU推論は `export_weights.py` で作る `.tetrawts` を読みます。逆方向の変換はありません。
- ルール、合法手生成、探索、dataset serializationはC++ childが担当します。Python processはPyTorch/ROCmでbatch推論だけを返します。
- GPU bridgeの `--engine` には、Linuxでは `build/tetra_cli`、Windowsでは `build/tetra_cli.exe` を指定します。

## 初回checkpointを作る

学習済みcheckpointがない場合は、C++の自己対局で初期datasetを作り、GPUでbootstrap学習します。

```sh
mkdir -p data models
./build/tetra_cli export data/bootstrap.tetradat 50 200 32

python trainer/train.py data/bootstrap.tetradat \
    --steps 2000 --model s --batch 256 \
    --device cuda --require-gpu \
    --save models/gen1.pt
```

接続確認だけなら、先に `--model dev --batch 32` で実行してから `--model s` へ移ります。学習ログにGPU名が表示されることを確認してください。

### モデルサイズの目安

| preset | パラメータ数 | 用途 |
|---|---:|---|
| `dev` / `xs` | 約0.13 M | 同じ構成（width 64、2層）。`dev` は接続確認、`xs` は高速探索の比較基準と蒸留先（[ADR 0014](adr/0014-model-size-vs-search-budget.md)） |
| `teacher1m` | 約0.95 M | XSへ蒸留する教師モデル |
| `s` | 約7.2 M | TetraFormer-Sの基準モデル |

batchは256程度から始め、VRAMとstep時間を見ながら調整します。

## checkpointをC++形式へ変換する

```sh
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts
./build/tetra_cli play models/gen1.tetrawts 200 64
```

tokenizerやモデルのfeature widthを変更した場合は、古いcheckpointを流用せず、同じcommitのdatasetから再学習します。

## GPU自己対局で次のdatasetを生成する

```sh
python trainer/gpu_selfplay.py models/gen1.pt data/gen2.tetradat \
    --engine build/tetra_cli \
    --device cuda \
    --games 32 --pieces 300 --sims 64 --batch 16 \
    --determinizations 2 --precision fp16 --model-version 2
```

`gpu_selfplay.py` には `--require-gpu` がありません。実行前に[セットアップ](SETUP.md#rocm版pytorchを導入する)の確認コマンドでGPUが見えることを確認してください。

出力にはdatasetのsample数とGPU推論の局面数が表示されます。二盤面の自己対局datasetは両プレイヤー視点を含むため、Compact Replay形式へ変換せずrectangular形式のまま扱います。

timing action（`WAIT_FOR_EVENT` などのdelay）は既定で無効です。`--timing-actions` は、[timingの再開条件](TRAINING_AND_EVALUATION.md#8-timingと相殺外しのカリキュラム)を満たした実験でだけ指定します。

## GPUで継続学習する

過去generationをreplayとして混ぜ、最後のdatasetを新データとして扱います。`--resume` はモデルだけでなくoptimizerとsampling RNGも復元します。

```sh
python trainer/train.py \
    data/gen1.tetradat data/gen2.tetradat \
    --resume models/gen1.pt \
    --new-data-repeat 1 \
    --steps 5000 --batch 256 --model s \
    --device cuda --require-gpu --value-weight 1.0 \
    --checkpoint-every 1000 \
    --best-save models/gen2.best.pt \
    --save models/gen2.pt
```

- 比較実験の基準は `--new-data-repeat 1` です。`4` などへ上げる場合は、新generationを意図的にoversampleする実験として記録します。
- loss weightの初期値は、方策 `1.0`、価値 `1.0`、補助目標 `0.1` です。checkpointにも保存されます。
- `--value-weight 0` は方策だけのablationで明示的に使います。通常の学習では `1.0` を維持します。
- ログには価値の正解率、価値のMSE、補助目標のvalid率、shared-trunkの勾配診断が出ます。方策損失だけで学習の健全性を判断しません。

## 対局統計を確認する

`gpu_match.py` はC++のゲームと探索を起動し、評価だけをGPUで処理します。APM、APP、PPSを出力します。

```sh
python trainer/gpu_match.py models/gen2.pt \
    --engine build/tetra_cli \
    --device cuda --games 4 --pieces 200 --sims 32 \
    --batch 16 --precision fp16 --workers 4
```

APM/APPだけで強さを判断しません。比較ではArenaの勝率と95%信頼区間を主な証拠にし、VS Score、PPS、平均生存時間、top outまでの手数を診断として記録します。

## GPU ArenaでCandidateを評価する

```sh
python trainer/gpu_arena.py models/gen2.pt models/gen1.pt \
    --engine build/tetra_cli \
    --device cuda --pairs 20 --pieces 300 --sims 32 \
    --batch 16 --determinizations 1 --precision fp16 --seed 42
```

Arenaはpaired seedで対局し、勝敗、勝率、VS Scoreなどを報告します。CandidateがChampionを上回っても、設定済みのpromotion thresholdを満たすまではChampionを置き換えません。

## 1 generationを自動実行する

`iterate.py` は、自己対局、replay mix、GPU学習、weight export、GPU Arena、条件付きpromotionを1つのdriverで実行します。

```sh
python trainer/iterate.py \
    --champion models/champion.pt \
    --replay data/gen1.tetradat \
    --generation 2 \
    --champion-output models/champion \
    --engine build/tetra_cli \
    --device cuda \
    --games 16 --pieces 300 --sims 64 --inference-batch 16 \
    --determinizations 2 --train-steps 5000 --train-batch 256 \
    --new-data-repeat 4 --arena-pairs 10 --arena-sims 32 --arena-pieces 300
```

`--champion-output models/champion` を指定すると、Arenaを通過した場合だけ `models/champion.pt` と `models/champion.tetrawts` が更新されます。通過しない場合はCandidateを保存し、Championを保持します。CPU Arenaを使う場合だけ `--cpu-arena` を指定します。

Reanalyseを含む連続実行は `trainer/auto_improve.py` を使います。人間リプレイから始める場合の例は[人間リプレイによる事前学習](HUMAN_REPLAY_PRETRAINING.md#自己改善を続ける)を参照してください。

## 失敗箇所を切り分ける

| 症状 | 最初に確認すること |
|---|---|
| 学習が異常に遅い | CPUへフォールバックしていないか。`--device cuda --require-gpu` を指定したか |
| weight load時の `feature width mismatch` | engineとcheckpointのcommit、schemaが一致しているか |
| validatorがschema mismatchを報告する | widthだけでなく、tokenizer、observation、action、補助目標のschemaが一致しているか |
| `cpp_matches_pytorch_exactly` が失敗する | C++とPyTorchのforwardがずれています。解消するまで学習を進めません |
| `.pt` をC++ CLIで読めない | `.pt` と `.tetrawts` を取り違えていないか |
| OOM | 学習の `--batch`、次に推論batchと並列度を下げます。比較実験では変更後の条件を記録します |

GPU検出やビルドの問題は[セットアップのトラブルシューティング](SETUP.md#環境のトラブルシューティング)を参照してください。
