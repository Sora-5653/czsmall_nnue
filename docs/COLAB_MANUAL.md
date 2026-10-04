# Colabで自己対局shardを生成する

Google Colabを追加の自己対局workerとして使う手順です。次の2つの入口があります。

- `trainer/colab_manual.py`: Google Driveで成果物を受け渡す半自動runner。Colabのcellから1段階ずつ実行します。
- `trainer/colab_generate.py`: shardの生成とmanifestの検証を行うCLI。ローカルでもColabでも使えます。

Colabは追加workerです。ルール、合法手生成、探索、label生成はC++エンジンが担当し、Champion promotion、seed割り当て、datasetの統合方法はローカル側で決めます。Drive、GASは成果物の転送に使えますが、seed、label、統合方法の権威にはしません。

## Driveを使って手動で実行する

### Driveにファイルを置く

同じDriveフォルダに次の2ファイルを置きます。

- `colab_bundle_sample_eff_manual.zip`
- `baseline_gpu_gen_20260805_v2.best.pt`

既定のフォルダ名は `czsmall_nnue_colab_20260806` です。別名にした場合は、各コマンドに `--drive-folder フォルダ名` を追加します。

Windowsで作成したZIPも使えます。スクリプトがメンバー名のバックスラッシュを変換します。

### セットアップする

ランタイムをGPUに設定し、最初のcellでDrive上のスクリプトを起動します。

```python
from google.colab import drive
drive.mount('/content/drive')
!python "/content/drive/MyDrive/czsmall_nnue_colab_20260806/colab_manual.py" setup
```

`setup` は次の処理を行います。

1. Driveをマウントし、ZIPを `/content/czsmall_nnue` に展開してcheckpointを配置します。
2. C++23の機能プローブを通るコンパイラを選びます。必要な場合はColab内でg++を追加インストールします。
3. `make tools` を実行し、GPUと `tetra_cli` を確認します。

### shardを生成して確認する

setup後は、展開されたスクリプトを使います。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate
!python /content/czsmall_nnue/trainer/colab_manual.py inspect
```

既定値は、1 shard、32ゲーム、1ゲーム最大200 pieces、探索32 sims、base seed `2026080600`、FP16です。生成後、次の2ファイルがDriveへ戻されます。

- `colab_shard_2026080600.tetradat`
- `colab_shard_2026080600.tetradat.manifest.json`

条件を変える場合の例です。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate --games 64 --pieces 300 --sims 64 --overwrite
```

shardを分ける場合は、全shardで `--base-seed`、`--games`、`--shard-count` を固定し、`--shard-id` だけを変えます。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate --shard-id 0 --shard-count 4 --output-name shard-0.tetradat
!python /content/czsmall_nnue/trainer/colab_manual.py generate --shard-id 1 --shard-count 4 --output-name shard-1.tetradat
```

### 補助目標の有無を比較する

同じseedで `aux` と `noaux` を実行します。両方とも同じデータ分割と初期化seedを使うため、paired ablationになります。補助目標のweightは `aux` が0.1、`noaux` が0.0です。WDLの目的は変更しません。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition aux --seed 0
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition noaux --seed 0
```

複数seedで比較する場合は、`--seed 1`、`--seed 2` と同じ順序で繰り返します。checkpointは `aux_seed0.pt` / `aux_seed0.best.pt` のような名前でDriveへ保存されます。既存の出力を置き換える場合だけ `--overwrite` を付けます。

### エラーを報告する

各コマンドは失敗した時点で停止し、次の段階へ進みません。診断を依頼する場合は、`error:` 行だけでなく、直前のGPU情報、実行した `$ ...` 行、Python/C++のtracebackを含むcellの出力をそのまま渡してください。

## colab_generate.pyでshardを生成・検証する

### shardを生成する

```sh
python trainer/colab_generate.py generate models/champion.pt \
    data/colab/shard-0.tetradat \
    --base-seed 100000 \
    --shard-id 0 --shard-count 4 \
    --games 32 --pieces 300 --sims 64 \
    --model-version 4 --device cuda --build-engine
```

shardごとに変えるのは `--shard-id` だけです。同じ `base_seed` と `games` のとき、shard `i` のseed区間は次のとおりです。

```text
[base_seed + i * games, base_seed + (i + 1) * games)
```

manifestには、repository commit、checkpoint hash、ruleset、model、探索設定、seed区間、sample数、datasetとschemaの情報が記録されます。

### manifestを検証する

```sh
python trainer/colab_generate.py validate \
    data/colab/shard-0.tetradat.manifest.json \
    data/colab/shard-1.tetradat.manifest.json \
    data/colab/shard-2.tetradat.manifest.json \
    data/colab/shard-3.tetradat.manifest.json \
    --checkpoint models/champion.pt --require-complete
```

validatorは、seed区間の重複や、互換でないruleset、checkpoint、探索設定、schemaを拒否します。

### 検証済みshardを学習へ渡す

shardをバイト連結せず、各パスを個別の入力として渡します。

```sh
python trainer/train.py \
    data/colab/shard-0.tetradat \
    data/colab/shard-1.tetradat \
    data/local.tetradat \
    --resume models/champion.pt \
    --device cuda --require-gpu \
    --value-weight 1.0 \
    --steps 5000 \
    --save models/candidate.pt
```

ローカルだけで生成したgenerationと、ローカル+Colabで生成したgenerationは、同じArena条件で比較してからpromotionします。詳細は[学習・評価プロトコル](TRAINING_AND_EVALUATION.md#4-datasetの由来)を参照してください。
