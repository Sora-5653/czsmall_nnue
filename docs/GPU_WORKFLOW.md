# GPU学習・推論を実行する

GPU学習、自己対局、Arenaを実行するときに使う手順です。コマンド例はリポジトリルートからLinux/WSL2のシェルで実行します。既存checkpointやdatasetに合わせてパスと実験条件を指定してください。

## 実行前に確認する

- 環境の導入とGPU検出は[セットアップ](SETUP.md)を参照します。既存環境を確認してから、必要な導入だけを行います。
- AMD GPUでもPyTorchのデバイス指定は `cuda` です。GPU学習では `--device cuda --require-gpu` を指定し、CPUへのフォールバックを成功として扱いません。
- PyTorch側は `.pt`、C++のCPU推論は `.tetrawts` を読みます。ルール、合法手、探索、dataset serializationはC++が担当します。
- [学習・評価プロトコル](TRAINING_AND_EVALUATION.md)に従い、engine、checkpoint、datasetのcommitとschemaの互換性を確認します。実験計画のseed、予算、停止条件を維持します。
- Colabの環境とshard運用は[Colab手順](COLAB_MANUAL.md)を参照します。生成済みdatasetをバイト連結せず、manifestで由来とseed重複を検証します。

## 1. C++エンジンをビルドして確認する

```sh
make test
make tools
```

`make test` が失敗したら学習を開始しない。特に
`cpp_matches_pytorch_exactly`、feature width mismatch、Tokenizerのテストが
失敗している場合は、checkpointやfixtureと現在のcommitが不一致の可能性がある。

Linuxでは `build/tetra_cli`、Windowsでは `build/tetra_cli.exe` が生成される。
GPU bridgeへ渡す場合は、必要に応じて `--engine` でその絶対パスを指定する。

## 2. 初回checkpointを作る

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

## 3. checkpointをC++形式へ変換する

```sh
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts
./build/tetra_cli play models/gen1.tetrawts 200 64
```

`.pt` を `.tetrawts` に変換するだけであり、逆方向の変換はない。Tokenizerや
モデルのfeature widthを変更した場合は、古いcheckpointを無理に使わず、同じ
commitのdatasetから再学習する。

## 4. GPU self-playで次のdatasetを生成する

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

## 5. GPUで継続学習する

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

## 6. GPU推論とTetr.io風スタッツを確認する

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

## 7. GPU ArenaでCandidateを評価する

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

## 8. 1 generationを自動実行する

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

## 9. 失敗箇所を切り分ける

1. GPU検出とROCm版PyTorchを確認します。architecture overrideは通常検出が失敗した場合だけ、セットアップ文書に従って検討します。
2. engine path、checkpoint形式、commitとschemaの互換性を確認します。
3. C++/PyTorch parity、feature width、Tokenizerの失敗を解消してから学習を再開します。
4. OOMの場合は、実行中コマンドのbatch設定を下げて再検証します。比較実験では変更後の条件を記録します。
