# セットアップ: RX 9070 XTで学習する

この文書は、**何を自分で導入する必要があるか**と、**リポジトリ側ですでに用意されているもの**を説明します。

エンジン本体は外部dependencyを持ちません。現在の `Makefile` とvendored Cobra backendは**C++23**を要求します。PythonやGPUが必要なのは学習・GPU推論を行う場合だけです。

---

## 1. 必要な環境

### Hardware / OS

RX 9070 XTは本プロジェクトではRDNA 4、LLVM target `gfx1201` として扱います。プロジェクト標準の学習環境は次のとおりです。

| 項目 | 前提 |
|---|---|
| ROCm | **7.2以降**を標準とする |
| OS | Linux、またはWSL2上のLinux |
| Python | 3.10–3.12を想定 |
| Compiler | C++23を扱えるg++ / clang++（現在の開発環境ではg++ 12を使用） |
| Disk | ROCm + PyTorch用におおむね10 GB程度 |

> Windows native側のPyTorch/ROCm実験環境を使うこともありますが、再現可能な標準手順はLinux/WSL2側に置きます。

### ROCmを導入する

利用しているdistributionに合わせてAMD公式のROCm installerを使用します。導入後、GPU targetが見えることを確認します。

```sh
rocminfo | grep gfx
# gfx1201 が表示されること
```

permissionで見えない場合は、必要に応じて `render` / `video` groupを確認します。

```sh
sudo usermod -aG render,video $USER
```

変更後は再loginが必要です。

### ROCm版PyTorchを導入する

GPU学習にはROCm用wheelを使います。

```sh
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install -r trainer/requirements.txt
```

ROCm版PyTorchもAPI上は `torch.cuda` を使います。これは正常です。

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True  AMD Radeon RX 9070 XT
```

`False` の場合、まず次を確認します。

- ROCm版wheelを入れているか
- `render` / `video` permissionがあるか
- WSL/Linux側からGPUが見えているか
- ROCm環境とPyTorch wheelの組み合わせが正しいか

### RDNA 4のarchitecture検出が失敗する場合

通常の検出で `gfx1201` を認識できない場合だけ、次を試します。

```sh
export PYTORCH_ROCM_ARCH=gfx1201
export HSA_OVERRIDE_GFX_VERSION=12.0.1   # 通常検出が失敗する場合だけ
```

`HSA_OVERRIDE_GFX_VERSION` は常用設定ではありません。まず通常のGPU認識、permission、ROCm wheelを確認してください。

---

## 2. リポジトリ側ですでに実装されているもの

2026-08-10時点では次が利用できます。

- C++23のdependency-free engine build
- Cobra-backed legal placement generation
- PUCT / Gumbel search
- hidden-future determinization
- 二盤面self-play
- replay bufferとdataset serialization
- PyTorch trainer
- C++ `.tetrawts` inference
- C++ / PyTorch parity test
- GPU match bridge
- GPU self-play generation
- GPU Candidate/Champion Arena
- resumable checkpoint
- replay mixingとguarded iteration
- tokenizer / observation / action / aux schema contract
- 36-dimensional auxiliary target schema + valid mask
- Colab shard generation / manifest validation
- CNN / hybrid architecture ablation tooling

通常 `make test` の直近検証は **282 tests / 1,045,724 assertions / 0 failed** です。

---

## 3. 初回実行

```sh
git clone <this repo>
cd czsmall_nnue

# 1. C++ engineをbuildして検証する。
make test
make tools
```

### Parity fixtureを再生成する場合

`tests/data/` のbinary fixtureはseedから再生成できます。

```sh
python scripts/make_fixtures.py
```

fixtureがない場合は対応するPyTorch parity testがskipされますが、残りのsuiteは実行されます。

### 初期データを作る

```sh
mkdir -p data models
./build/tetra_cli export data/train.tetradat 50 200 32
```

### GPUで学習する

GPUを使うrunでは、CPUへのsilent fallbackを避けるため `--device cuda --require-gpu` を付けます。

```sh
python trainer/train.py data/train.tetradat \
    --steps 2000 --model s \
    --device cuda --require-gpu \
    --save models/gen1.pt
```

### C++用weightへexportする

```sh
python trainer/export_weights.py models/gen1.pt models/gen1.tetrawts
./build/tetra_cli play models/gen1.tetrawts 200 64
```

`.pt` はPyTorch checkpoint、`.tetrawts` はdependency-free C++ evaluator用weightです。両者を取り違えないでください。

### GPU matchを実行する

```sh
python trainer/gpu_match.py models/gen1.pt \
    --device cuda --games 4 --pieces 200 --sims 32 \
    --precision fp16 --workers 4
```

現在はAPM / APP / PPSなどを報告します。VS Scoreは設計上の次期標準診断ですが、計算式とruleset interpretationを固定して実装するまでは「利用可能」とみなしません。

### GPU Arenaを実行する

```sh
python trainer/gpu_arena.py models/candidate.pt models/champion.pt \
    --device cuda --pairs 20 --pieces 300 --sims 32 \
    --precision fp16 --seed 42
```

### GPU自己対局を生成する

```sh
python trainer/gpu_selfplay.py models/gen1.pt data/gen2.tetradat \
    --device cuda --games 32 --pieces 300 --sims 64 \
    --model-version 2
```

### 1 generationをguard付きで回す

```sh
python trainer/iterate.py --champion models/champion.pt \
    --replay data/gen1.tetradat --generation 2 \
    --champion-output models/champion --device cuda
```

CandidateはArena gateを通った場合だけChampionへ昇格します。

全体の接続確認だけなら次でも実行できます。

```sh
./scripts/bootstrap.sh
```

---

## 4. 学習loop

AlphaZero系の反復では、概念的に次を繰り返します。

1. 現在のChampionで自己対局を生成する。
2. 過去generationをreplayとして混ぜる。
3. Candidateを学習する。
4. CandidateとChampionをpaired Arenaで比較する。
5. promotion thresholdを満たした場合だけChampionを更新する。

### C++ weightで自己対局datasetを出力する

`tetra_cli export` は `.tetrawts` を `--weights` で受け取れます。

```sh
./build/tetra_cli export data/gen$N.tetradat 200 300 64 \
    --weights models/champion.tetrawts
```

### Resume training

`--resume` はmodel weightだけでなくoptimizerとsampling RNGも復元します。

```sh
python trainer/train.py \
    data/gen1.tetradat data/gen2.tetradat \
    --resume models/champion.pt \
    --new-data-repeat 1 \
    --steps 5000 --model s --batch 256 \
    --device cuda --require-gpu \
    --value-weight 1.0 \
    --checkpoint-every 1000 \
    --best-save models/gen2.best.pt \
    --save models/gen2.pt
```

`--new-data-repeat 1` を比較実験の基準とします。`4` などへ上げる場合は、新generationを意図的にoversampleする実験として記録してください。

### Loss weight

通常の初期値は次です。

- policy: `1.0`
- value: `1.0`
- aux: `0.1`

checkpointにはloss weightも保存されます。

`--value-weight 0` はpolicy-only ablationのために明示的に使うoptionです。通常runでvalueを意図せず無効にしないでください。

`trainer/train.py` はvalue accuracy、scalar value MSE、aux valid rate、shared-trunk gradient diagnosticsも出力できます。policy lossだけを見て「健康なrun」と判断しないでください。

---

## 5. Dataset formatとschema

現行の `DatasetHeader::VERSION` は**3**です。

Version 3は、schemaとtermination metadataを持つrectangular datasetで、二盤面自己対局の標準形式です。

別に `DatasetHeader::VERSION_COMPACT = 2` が存在します。これはReplay+πを保存し、再生可能なtrajectoryからload時にtoken/actionを再生成するcompact形式です。

二盤面self-playでは、相手のattack/event streamまで含む観測をcompact v2 metadataだけから再構成できないため、version 3を使います。

また、feature widthだけで互換性を判定しません。次のようなschema metadataを照合します。

- tokenizer schema version/hash
- observation schema hash
- action schema version
- auxiliary target schema version/count
- ruleset hash
- model version
- termination reason

異なるschemaを持つshardを暗黙に同じbatchへ混ぜないでください。

---

## 6. Colabを追加生成workerとして使う

Driveを使う手動workflowは [`COLAB_MANUAL.md`](COLAB_MANUAL.md) を参照してください。

Colabは**追加の自己対局worker**です。ローカルmachine側をChampion promotionと最終比較の権威に保ちます。

### shardを生成する

```sh
python trainer/colab_generate.py generate models/champion.pt \
    data/colab/shard-0.tetradat \
    --base-seed 100000 \
    --shard-id 0 --shard-count 4 \
    --games 32 --pieces 300 --sims 64 \
    --model-version 4 --device cuda --build-engine
```

shard `i` ごとに変えるのは基本的に `--shard-id i` だけです。

同じ `base_seed` と `games` なら、shard `i` のseed intervalは

```text
[base_seed + i * games, base_seed + (i + 1) * games)
```

になります。

manifestには少なくとも次が記録されます。

- repository commit
- checkpoint hash
- ruleset/model information
- search setting
- seed interval
- sample count
- dataset/schema information

### manifestを検証する

```sh
python trainer/colab_generate.py validate \
    data/colab/shard-0.tetradat.manifest.json \
    data/colab/shard-1.tetradat.manifest.json \
    data/colab/shard-2.tetradat.manifest.json \
    data/colab/shard-3.tetradat.manifest.json \
    --checkpoint models/champion.pt --require-complete
```

validatorはoverlapするseed intervalや互換でないruleset/checkpoint/search/schemaを拒否します。

### validated shardを学習へ渡す

byte単位で結合せず、各pathを個別inputとして渡します。

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

Google DriveやGASはfile transfer/orchestrationの補助には使えますが、seed・label・merge semanticsの権威にはしません。

---

## 7. Model sizeの目安

| model | parameters | 過去のCPU forward測定 | 用途 |
|---|---:|---:|---|
| `--model dev` | 約0.13 M | 約1 ms | 接続確認・sanity check |
| `--model s` | 約7.2 M | 約8.6 ms | TetraFormer-S基準model |

これらのCPU値は初期の2-core環境での測定であり、現在のGPU throughputを表しません。

RX 9070 XTでは、まずbatch 256程度から開始し、VRAM・step timeを観測しながら調整するのが安全です。OOM時は `--batch` を最初に下げます。

---

## 8. Troubleshooting

| 症状 | 最初に確認すること |
|---|---|
| `torch.cuda.is_available()` が `False` | ROCm wheel、WSL/LinuxからのGPU可視性、`render`/`video` permission |
| `HIP error: invalid device function` | architecture detection。必要なら `PYTORCH_ROCM_ARCH=gfx1201` |
| weight load時の `feature width mismatch` | engineとcheckpointのcommit/schemaが一致しているか |
| dataset validatorがschema mismatch | widthだけでなくtokenizer/observation/action/aux schemaが一致しているか |
| `cpp_matches_pytorch_exactly` が失敗 | C++とPyTorch forwardがdriftしている。解決まで学習を進めない |
| `-Werror` buildがvendored Cobra内で失敗 | 既知のwarning-scope問題。通常 `make test` の成否と分け、third-party warning policyを整理してからCIを有効化する |
| OOM | まずtraining `--batch`、次にinference batch/self-play並列度を下げる |
| GPU学習が異常に遅い | CPUへfallbackしていないか。`--device cuda --require-gpu` を確認 |
| `.pt` をC++ CLIで読めない | `.pt` と `.tetrawts` の用途を取り違えていないか |

## 9. 実行前の最小checklist

1. `git status` と実行commitを確認する。
2. `make test` が通ることを確認する。
3. GPU runなら `torch.cuda.is_available()` が `True` であることを確認する。
4. checkpoint / dataset / engine / manifestが同じschema・ruleset・commit系列か確認する。
5. comparison runではgames、pieces、sims、seed、precision、splitを固定する。
6. CandidateをChampionへ直接copyせずArena gateを通す。

より詳細なGPU運用手順はリポジトリ直下の `AGENTS.md`、学習比較の規約は `TRAINING_AND_EVALUATION.md` を参照してください。
