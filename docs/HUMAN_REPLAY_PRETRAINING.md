# 人間リプレイによる事前学習

同意を得たTETR.IOのマルチプレイリプレイから方策/価値networkを初期化し、その後は通常の自己対局とReanalyseのloopへ移る手順です。

実装は、MochBot/fusionが公開している学習pipelineの大まかな流れ（rankで絞った集団 → リプレイcorpus → 前処理 → 方策/価値データ → 学習）を参考にしています。MochBotのsource codeは取り込んでいません。MochBot/fusionが公開している収集・前処理のsourceにはlicenseの宣言がないため、このリポジトリでは処理の流れだけを参考にし、公開されているリプレイのデータ形式とTetraの既存のエンジンinterfaceに対して独自に実装しました。

ネットワークアクセスの範囲は[利用ポリシー](POLICY.md#tetra-channel-apiと学習データ収集)を参照してください。

## データの流れ

```text
TETRA CHANNEL X+ cohort or consented .ttrm files
  -> trainer/collect_xplus_replays.py (optional collection)
  -> trainer/ttrm_exact_replay.py + trainer/ttrm_ingest.py
  -> content-addressed normalized replay cache
  -> tetra_cli import-human-replay
  -> C++ legal-action validation + tokenization
  -> human_*.tetradat + manifest.json
  -> teacher1m supervised training
  -> trainer/distill.py
  -> XS checkpoint
  -> trainer/auto_improve.py
  -> self-play -> Reanalyze -> train -> Arena gate
```

責務は次のように分かれます。

- **Python:** JSONを正規化し、TETR.IO v19のframeとsubframeの状態（seed付き7-bag、handling、lock、line clear、garbageの応酬を含む）を再構成します。厳密な取り込みでは、両プレイヤーの再構成後の終了状態を検証し、不一致または警告があればそのroundを除外します。
- **C++:** 合法配置の生成、model向けのtoken化、action埋め込み、dataset serializationを担当します。sampleを書き出す前に、実演された各配置がproductionのaction空間と一致するかを独立に照合します。

## X+ corpusを収集する

X+ collectorは中断から再開でき、rate limitを守ります。文書化されたTETRA CHANNELのleague leaderboardから現在のX+ cohortを取得し、各playerの最近のleague記録IDを取得します。replay IDを全体で重複排除し、`data/xplus_replays/_meta/` に追記専用のledgerを保持します。

```sh
python trainer/collect_xplus_replays.py \
  --output-dir data/xplus_replays \
  --ranks x+ \
  --request-interval 1.05
```

- `https://ch.tetr.io/api` は、playerと記録の検索に使います。ページ送りされるdatasetの間では `X-Session-ID` を引き継ぎます。
- 文書化されていない本体ゲームのAPIは使わず、proxyの切り替えも行いません。
- TETRA CHANNELはreplay IDを公開していますが、リプレイファイルのdownload endpointは文書化していません。そのためリプレイ本体は、`{replayid}` を含む設定可能なURL templateから取得します。既定値は、MochBotの収集コードが参照している公開ミラーです。同意済みまたは私的なsourceを使う場合は、`TETRA_REPLAY_URL_TEMPLATE` または `--replay-url-template` で指定します。

範囲を絞った動作確認の例です。

```sh
python trainer/collect_xplus_replays.py \
  --output-dir data/xplus_replays_smoke \
  --max-players 1 \
  --max-record-pages 1 \
  --max-replays 1 \
  --strict
```

`players.jsonl`、`records_status.jsonl`、`replay_index.jsonl`、`downloads.jsonl` により、中断したrunを再開できます。`--refresh-cohort` と `--refresh-records` は対応する検索ledgerを意図的に無効化します。これらのflagで、download済みのリプレイファイルが削除されることはありません。

## エンジンをビルドする

通常のビルドを使います。

```sh
make tools
```

`build/` に別プラットフォームのobjectが残っている場合は、既存の成果物を削除せず、別のディレクトリを使います。

```sh
make BUILD=build-human tools
```

その実行ファイルを `--engine` に渡します。

## リプレイを一括で取り込む

```sh
python trainer/import_human_replays.py /path/to/consented/replays \
  --engine build/tetra_cli \
  --output-dir data/human_replay_shards \
  --cache-dir data/human_replay_cache \
  --samples-per-shard 4096 \
  --workers 16 \
  --ruleset league \
  --strict-source \
  --exact
```

- 入力にはファイルとディレクトリを指定できます。ディレクトリは再帰的に探索します。
- 出力は、検証済みの `.tetradat` shard、content-addressedなcache、sourceごととshardごとの件数とhashを含む `manifest.json` です。
- 実演された配置は、C++の合法手生成が出すmacro actionと一致した場合だけ採用します。不正な状態、リプレイの実行失敗、一致する合法手がない場合は、推測でdatasetへ入れず、それぞれ別に集計します。
- `--exact` を指定すると、fail-closedなTETR.IO v19の再構成器を必須にします。X+ bootstrapは常にこれを指定します。`--exact` なしの汎用importerは、以前から対応しているリプレイexport向けの旧keydown adapterを使います。

## X+からXSまでを1コマンドで作る

サイズと探索のablationで使う小型の推論用モデルには、`xs` preset（width 64、Transformer 2層、4 heads、FFN 192）を使います。

```sh
python3 trainer/xplus_bootstrap.py \
  --engine build/tetra_cli \
  --train-python ./.venv-rocm714/Scripts/python.exe \
  --device auto \
  --require-gpu \
  --steps 5000 \
  --batch 256 \
  --save models/xplus_xs_bootstrap.pt
```

このコマンドは、収集、厳密な取り込みとshard化、`teacher1m` の教師あり学習、XSへの蒸留を順に実行します。

- 収集とC++の取り込みはwrapperを起動したPythonで実行し、学習は `--train-python` で別のinterpreterを指定できます。これにより、WSL上のELF形式の `tetra_cli` と、Windowsの `.venv-rocm714` のROCm環境を組み合わせられます。
- 処理はfail-closedです。すべてのsource roundが厳密な再構成を通過し、正規化したすべての配置がC++の検証を通過し、manifestが `exact-v19` normalizerを記録している必要があります。
- `--skip-collect` と `--skip-import` は完了済みのデータ段階を再利用しますが、上記の検証は緩めません。教師モデルの学習の再開には `--teacher-resume` を使います。
- 一部のcorpusでの診断やXSを直接学習する比較では、`import_human_replays.py --exact` または `trainer/train.py` を個別に実行します。production用のwrapperは、検証を弱めるmodeを持ちません。

学習は、shardディレクトリ内の `.tetradat` をすべて拾うのではなく、最新の `manifest.json` に記録されたshardの一覧を読みます。古いshardをcacheやdebug用に残しても、新しいshard化のrunへ暗黙に混ざりません。

同意済みのリプレイディレクトリが手元にある場合は、汎用の入口を使います。

```sh
python trainer/human_pretrain.py /path/to/consented/replays \
  --engine build/tetra_cli \
  --model xs \
  --device auto \
  --require-gpu \
  --steps 5000 \
  --batch 256 \
  --save models/human_pretrain.pt
```

- 汎用の人間データ経路の既定値は、方策weight `1.0`、価値weight `0.25`、局所補助目標とtiming目的のweight 0です。X+の蒸留経路は、教師の学習と生徒の蒸留の両方で方策だけを既定とします。人間のキー入力のtimingは、Tetraの戦略的なdelay actionとは別のtargetのため、timingと局所補助目標は自己対局とReanalyseが担当します。
- 既存のcheckpointから始める場合は `--resume` を使います。
- `--exact` で厳密なv19再構成を選びます。`--skip-import --exact` は、manifestにその由来が記録されたshardだけを受け付けます。

## 自己改善を続ける

```sh
python trainer/auto_improve.py \
  --champion models/human_pretrain.pt \
  --bootstrap-replay-dir data/human_replay_shards \
  --reanalyze \
  --device cuda
```

`--bootstrap-replay-dir` は、そのディレクトリ以下のすべての `.tetradat` を固定のreplay集合に加えます。`--bootstrap-replay` で指定したパスとの重複は除き、新しい自己対局のrolling windowとは別に保持します。

## リプレイ再構成の契約

normalizerは、正規形のtop-levelのreplay-setデータと、以前のcollector形式の `replay.rounds` データを受け付けます。

`--exact` を指定すると、再構成は次のように行います。

1. 共通のreplay seedを使い、見えている最初のbagの先頭部分を検証します。
2. TETR.IOがserializeしたsubframeの解像度で入力を処理し、v19のhandlingとgravityを適用します。
3. 2人のプレイヤーのstream間で、相互作用のeventを照合します。
4. 再構成した盤面、操作中とholdのpiece、queue、combo、B2B、garbageの状態、game overの状態を、serializeされた各終了snapshotと比較します。

再構成で警告または終了状態の不一致が出た場合、importerはそのround全体を除外します。配置を推測したり、ずれたroundの有効な前半だけを残したりしません。Pythonで検証した後、C++がleague rulesetで各配置前の状態を再構築して合法なmacro actionを生成し、配置とline clear後の状態が厳密なリプレイ記録と一致するactionだけを採用します。

厳密な再構成はデータの完全性を保証するgateであり、強さの結果ではありません。promotionの前に、得られたXS checkpointを人間データを使わないXSの基準とArenaで比較してください。

## 検証する

```sh
python -m unittest \
  trainer.test_ttrm_ingest \
  trainer.test_ttrm_exact_replay \
  trainer.test_import_human_replays \
  trainer.test_human_pretrain \
  trainer.test_xplus_replay_collector \
  trainer.test_xplus_bootstrap \
  trainer.test_distill
make test
```

`tests/data/human_replay_sample.ttrm` は、汎用parserとC++ importer用の小さな正規形のfixtureです。厳密な経路では、代表的なTETR.IO v19のファイルで検証し、`manifest.json` のsource error、`totals.import_fraction`、および `invalid`、`execution`、`unmatched` の件数を確認します。productionのX+ bootstrapでは、source errorがなく、C++の検証のskipがなく、import fractionが `1.0` である必要があります。
