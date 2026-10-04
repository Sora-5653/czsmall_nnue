# 開発環境をセットアップする

この文書では、C++エンジンのビルドと、RX 9070 XTでGPU学習を行うためのROCm/PyTorch環境の導入を説明します。

- 導入後のGPU学習、自己対局、Arenaの実行手順は[GPU手順](GPU_WORKFLOW.md)を参照してください。
- Colabを追加の生成workerとして使う手順は[Colab手順](COLAB_MANUAL.md)を参照してください。
- 実装済みの機能と現在の優先順位は[ロードマップ](ROADMAP.md)で管理します。

エンジン本体は外部依存を持たず、ビルドにはC++23コンパイラだけが必要です。PythonとGPUは、学習とGPU推論を行う場合だけ必要です。

## 必要な環境

RX 9070 XTは、本プロジェクトではRDNA 4（LLVM target `gfx1201`）として扱います。

| 項目 | 前提 |
|---|---|
| ROCm | 7.2以降 |
| OS | Linux、またはWSL2上のLinux |
| Python | 3.10–3.12 |
| コンパイラ | C++23に対応したg++またはclang++ |
| ディスク | ROCmとPyTorchでおよそ10 GB |

再現可能な標準手順はLinux/WSL2側に置きます。Windows nativeのROCm環境（例: `.venv-rocm714`）を使う場合も、この文書の確認項目は同じです。

## ROCmを導入する

使用しているディストリビューション向けのAMD公式インストーラーでROCmを導入し、GPU targetが見えることを確認します。

```sh
rocminfo | grep gfx
```

出力に `gfx1201` が含まれれば成功です。表示されない場合は、`render` / `video` グループへの所属を確認します。

```sh
sudo usermod -aG render,video $USER
```

グループの変更は再ログイン後に反映されます。

## ROCm版PyTorchを導入する

リポジトリのルートで仮想環境を作り、ROCm用のwheelを導入します。

```sh
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install -r trainer/requirements.txt
```

ROCm版PyTorchでも、APIとデバイス指定は `torch.cuda` / `cuda` です。次のコマンドでGPUが見えることを確認します。

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

`True` と `AMD Radeon RX 9070 XT` が表示されれば成功です。GPUが見えない状態で学習を開始しないでください。

### architecture検出が失敗する場合

通常の検出で `gfx1201` を認識できない場合だけ、次の環境変数を試します。

```sh
export PYTORCH_ROCM_ARCH=gfx1201
export HSA_OVERRIDE_GFX_VERSION=12.0.1
```

`HSA_OVERRIDE_GFX_VERSION` は常用設定ではありません。先にROCmのバージョン、グループ権限、ROCm用wheelを確認してください。

## エンジンをビルドして検証する

```sh
make test
make tools
```

Linuxでは `build/tetra_cli`、Windowsでは `build/tetra_cli.exe` が生成されます。

別のプラットフォームで作った `build/` が残っている場合は、既存の成果物を削除せず、別のビルドディレクトリを使います。古いオブジェクトが混ざると、テストが無関係な箇所で異常終了することがあります。

```sh
make BUILD=build-local test
```

### parity fixtureを再生成する

`tests/data/` のC++/PyTorch parity fixtureは、seedから再生成できます。tokenizerやモデルのfeature widthを変更した場合は再生成してから `make test` を実行します。

```sh
python scripts/make_fixtures.py
```

fixtureがない場合、対応するparity testはskipされ、残りのテストは実行されます。

## 環境のトラブルシューティング

| 症状 | 最初に確認すること |
|---|---|
| `torch.cuda.is_available()` が `False` | ROCm用wheel、Linux/WSL2からのGPU可視性、`render` / `video` 権限 |
| `HIP error: invalid device function` | architecture検出。必要なら `PYTORCH_ROCM_ARCH=gfx1201` |
| `make test` が途中で異常終了する | 別プラットフォームの古い `build/` が混ざっていないか。別の `BUILD=` で再ビルドする |
| `-Werror` ビルドがvendored Cobra内で失敗する | 既知のwarning範囲の問題です。[ロードマップ](ROADMAP.md#toolingとciの既知の課題)を参照 |

学習・推論の実行時に起きる問題は、[GPU手順の失敗の切り分け](GPU_WORKFLOW.md#失敗箇所を切り分ける)を参照してください。
