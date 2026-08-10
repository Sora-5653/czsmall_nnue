# Colab手動実行手順

`trainer/colab_manual.py` は、Google Driveをartifact受け渡しに使う半自動runnerです。Colabのcellから一段ずつ実行し、失敗した段階のoutputをそのまま診断へ返せるようにしています。

この手順でも、ルール・Cobra合法手生成・探索・label生成の権威はC++ engineです。Colabは追加workerであり、Champion promotion、seed allocation、dataset merge semanticsの第二の権威にはしません。生成datasetは現行schema付きrectangular version 3です。

## 1. Driveに置くファイル

同じDriveフォルダに次の2ファイルを置きます。

- `colab_bundle_sample_eff_manual.zip`
- `baseline_gpu_gen_20260805_v2.best.pt`

既定のフォルダ名は `czsmall_nnue_colab_20260806` です。別名にした場合は、以下の各コマンドに `--drive-folder フォルダ名` を追加してください。

ZIPはWindowsで作成したものでも構いません。スクリプト側でメンバー名のバックスラッシュをColab向けに変換します。

## 2. Colabセル

ランタイムはGPUに設定します。最初のセルでは、Drive上のスクリプトを直接起動します。

```python
from google.colab import drive
drive.mount('/content/drive')
!python "/content/drive/MyDrive/czsmall_nnue_colab_20260806/colab_manual.py" setup
```

`setup` はDriveをマウントし、ZIPを `/content/czsmall_nnue` に展開し、チェックポイントを配置します。その後、C++23機能プローブを通ったコンパイラを選び、必要ならColab内でg++を追加インストールして `make tools` を実行します。GPUと `tetra_cli` が確認できたら生成へ進みます。

setup後は、展開されたスクリプトを使えます。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate
```

既定値は、1 shard・32ゲーム・1ゲーム最大200 pieces・search 32 sims・固定base seed `2026080600`・FP16です。生成後、次の2ファイルがDriveへ戻されます。

- `colab_shard_2026080600.tetradat`
- `colab_shard_2026080600.tetradat.manifest.json`

生成物を確認します。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py inspect
```

## 3. 補助目標の比較学習

同じseedで `aux` と `noaux` を別々に実行します。両方とも同じデータ分割・初期化seedを使うため、paired ablationになります。`aux` 条件の既定weightは0.1、`noaux` は0.0です。これは補助目標の有無を切り分ける実験であり、WDL reward自体は変更しません。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition aux --seed 0
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition noaux --seed 0
```

複数seedで比較する場合は `0, 1, 2` などを同じ順序で繰り返します。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition aux --seed 1
!python /content/czsmall_nnue/trainer/colab_manual.py train --condition noaux --seed 1
```

チェックポイントは `aux_seed0.pt` / `aux_seed0.best.pt` のような名前でDriveへ保存されます。既存の出力を置き換える場合だけ `--overwrite` を付けます。

## 4. Errorを診断するとき

自動で次の段階へ進めず、各コマンドが失敗した時点で停止します。`error:` 行だけでなく、直前のGPU情報、実行した `$ ...` 行、Python/C++のtracebackを含むセル出力をそのまま渡してください。

設定を変える例:

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate --games 64 --pieces 300 --sims 64 --overwrite
```

shardを分ける場合は、全shardで `--base-seed`、`--games`、`--shard-count` を固定し、`--shard-id` だけを変えます。

```python
!python /content/czsmall_nnue/trainer/colab_manual.py generate --shard-id 0 --shard-count 4 --output-name shard-0.tetradat
!python /content/czsmall_nnue/trainer/colab_manual.py generate --shard-id 1 --shard-count 4 --output-name shard-1.tetradat
```
