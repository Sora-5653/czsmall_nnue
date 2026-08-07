# AGENTS.md
このリポジトリでGPU学習・GPU推論を行うときの標準手順。別セッションで
学習を再開する場合も、まずこの手順を確認すること。

## 重要な前提

- RX 9070 XTはAMD GPUなので、PyTorchはROCm版を使う。ROCm版PyTorchでも
  GPU指定は `cuda` と書く。
- `python trainer/train.py` は、GPUが見えないとデフォルトではCPUへ
  フォールバックする。GPU学習では必ず `--device cuda --require-gpu` を付ける。
- 学習・GPU self-play・GPU Arenaが読むのはPyTorch checkpointの `.pt`。
  C++のCPUエンジンが読むのは `export_weights.py` で作る `.tetrawts`。
- ルール、Cobra movegen、探索、dataset serializationの権威はC++側。
  Python/ROCm側はモデル評価と学習を担当する。
- 実行するcommitを、ローカル・Colab・checkpoint・datasetで揃える。

## 1. GPU環境を確認する

ROCmはLinuxまたはWSL2側で実行する。リポジトリのルートで次を行う。

```sh
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install -r trainer/requirements.txt

rocminfo | grep gfx
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

期待値は `gfx1201`、`True`、`AMD Radeon RX 9070 XT`。GPUが見えないまま
学習を開始してはいけない。

RDNA 4のarchitecture検出で失敗する場合だけ、次を試す。

```sh
export PYTORCH_ROCM_ARCH=gfx1201
export HSA_OVERRIDE_GFX_VERSION=12.0.1
```

`HSA_OVERRIDE_GFX_VERSION` は常用設定ではなく、通常の検出が失敗した場合
だけ使う。ROCm 7.2以上、`render` / `video` group、ROCm用wheelを先に確認する。

## 2. C++エンジンをビルドして確認する

```sh
make test
make tools
```

`make test` が失敗したら学習を開始しない。特に
`cpp_matches_pytorch_exactly`、feature width mismatch、Tokenizerのテストが
失敗している場合は、checkpointやfixtureと現在のcommitが不一致の可能性がある。

Linuxでは `build/tetra_cli`、Windowsでは `build/tetra_cli.exe` が生成される。
GPU bridgeへ渡す場合は、必要に応じて `--engine` でその絶対パスを指定する。

## 3. 初回checkpointを作る

まだ学習済みcheckpointがない場合は、C++ self-playで初期datasetを作り、GPUで
bootstrap学習する。

```sh
mkdir -p data models
./build/tetra_cli export data/bootstrap.tetradat 50 200 32

python trainer/train.py data/bootstrap.tetradat \
    --steps 2000 --model s --batch 256 \
    --device cuda --require-gpu \
    --save models/gen1.pt
```

必要なら最初は `--model dev --batch 32` で接続確認を行い、その後 `--model s`
へ移る。学習ログにGPU名が表示されることを確認する。

## 4. checkpointをC++形式へ変換する

```sh
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts
./build/tetra_cli play models/gen1.tetrawts 200 64
```

`.pt` を `.tetrawts` に変換するだけであり、逆方向の変換はない。Tokenizerや
モデルのfeature widthを変更した場合は、古いcheckpointを無理に使わず、同じ
commitのdatasetから再学習する。

## 5. GPU self-playで次のdatasetを生成する

GPU self-playではC++ childがルール・Cobra movegen・探索・dataset出力を担当し、
Python processがPyTorch/ROCmでbatched inferenceを返す。

```sh
python trainer/gpu_selfplay.py models/gen1.pt data/gen2.tetradat \
    --engine build/tetra_cli \
    --device cuda --require-gpu \
    --games 32 --pieces 300 --sims 64 --batch 16 \
    --determinizations 2 --precision fp16 --model-version 2
```

Windowsの場合は `--engine build/tetra_cli.exe` とする。出力にはdatasetの
sample数とGPU inference位置数が出る。self-playのdatasetは二盤面・両プレイヤー
視点を含むため、Compact Replay形式へ変換せずrectangular datasetとして扱う。

## 6. GPUで継続学習する

過去generationをreplay mixし、最後のdatasetを新データとして扱う。`--resume`
はモデルだけでなくoptimizerとsampling RNGも復元する。

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

新データを意図的に重くする実験では `--new-data-repeat 4` などを使うが、
sample-efficiencyの比較では条件を固定し、まず `1` を基準にする。WDL value head
は標準で学習されるので、通常は `--value-weight 1.0` を維持する。

## 7. GPU推論とTetr.io風スタッツを確認する

`gpu_match.py` はC++のゲーム・探索を起動し、評価だけをPyTorch/ROCm GPUで処理する。
APM、APP、PPSを出力する。

```sh
python trainer/gpu_match.py models/gen2.pt \
    --engine build/tetra_cli \
    --device cuda --games 4 --pieces 200 --sims 32 \
    --batch 16 --precision fp16 --workers 4
```

比較実験では `--seed` 相当の条件、games、pieces、sims、precision、checkpointを
固定する。APM/APPだけで強さを判断せず、Arenaの勝率と95% CI、PPS、平均生存時間、
top outまでの手数も記録する。

## 8. GPU ArenaでCandidateを評価する

```sh
python trainer/export_weights.py models/gen2.pt models/gen2.tetrawts
python trainer/gpu_arena.py models/gen2.pt models/gen1.pt \
    --engine build/tetra_cli \
    --device cuda --pairs 20 --pieces 300 --sims 32 \
    --batch 16 --determinizations 1 --precision fp16 --seed 42
```

Candidate checkpointがChampionを上回っても、Arenaのpromotion thresholdを
満たすまではChampionを置き換えない。CPU Arenaを使う場合だけ
`trainer/iterate.py --cpu-arena` を指定する。

## 9. 1 generationを自動実行する

通常の継続学習は、self-play、replay mix、GPU train、weight export、GPU Arena、
条件付きpromotionを一つのdriverで行う。

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

`--champion-output models/champion` を指定した場合、Arenaが通ったときだけ
`models/champion.pt` と `models/champion.tetrawts` が更新される。Arenaが通らない
場合はcandidateを保存したままChampionを保持する。

## 10. Colabを局面生成に使う

Colabは追加self-play局面の生成に使い、checkpointのpromotionと最終学習条件の
管理はローカル側で行う。全instanceで同じcommit、同じcheckpoint、同じruleset、
同じsearch設定を使う。

Colab上でROCmではなくCUDA GPUが見えることを確認した後、直接shardを生成する。

```sh
python trainer/colab_generate.py generate models/champion.pt \
    data/colab/shard-0.tetradat \
    --repo-root . --build-engine --device cuda \
    --base-seed 100000 --shard-id 0 --shard-count 4 \
    --games 32 --pieces 300 --sims 64 --model-version 4
```

shard `i` では `--shard-id i` だけを変える。同じbase seed、games、shard countを
使うことでseed intervalが重ならない。生成された `.tetradat` と
`.tetradat.manifest.json` をローカルへ戻す。

```sh
python trainer/colab_generate.py validate \
    data/colab/shard-0.tetradat.manifest.json \
    data/colab/shard-1.tetradat.manifest.json \
    data/colab/shard-2.tetradat.manifest.json \
    data/colab/shard-3.tetradat.manifest.json \
    --checkpoint models/champion.pt --require-complete

python trainer/train.py \
    data/colab/shard-0.tetradat data/colab/shard-1.tetradat \
    data/local.tetradat --resume models/champion.pt \
    --device cuda --require-gpu --value-weight 1.0 \
    --steps 5000 --save models/candidate.pt
```

datasetをバイト連結しない。各shardを個別の入力として渡し、validatorでcommit、
checkpoint hash、ruleset/model version、search設定、sample数、seed重複を検査する。
GAS/Driveはファイル移動の補助であり、seed・label・mergeの権威にはしない。

## 11. 失敗時の確認順

1. `python -c "import torch; print(torch.cuda.is_available())"` が `True` か。
2. PyTorchがROCm wheelか。AMDでもAPI名は `torch.cuda` で正しい。
3. 学習コマンドに `--device cuda --require-gpu` があるか。
4. `make tools`を実行済みか。engine pathが `build/tetra_cli(.exe)` と一致するか。
5. `.pt` と `.tetrawts` を取り違えていないか。
6. checkpoint、dataset、engineが同じcommit由来か。
7. `make test`、特にPyTorch parityとfeature width mismatchを確認する。
8. OOMなら、まず `--batch`、次に `--inference-batch`、self-playの`--games`を下げる。

GPUが見えない状態でCPUへ黙って切り替わる実行は、GPU学習の成功とはみなさない。
