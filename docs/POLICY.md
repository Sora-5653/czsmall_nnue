# 利用範囲と運用ポリシー

このリポジトリは、ローカル・オフラインで動作するTetrisシミュレータと研究用コードです。TETR.IOのclientではなく、ゲームサーバーへ接続して対局するコードを含みません。学習データの収集にはTETRA CHANNELの公開APIを利用します。

## このリポジトリに含まれるもの

- 厳密で決定論的なevent駆動のルールコア（`include/tetra`）
- ルールコア上の合法配置generator
- 機械学習実験用の観測とtokenizerの層
- ローカルmachine上で動作する開発ツールと学習コード

## 意図的に含めないもの

当初仕様の運用条件（[SPEC](SPEC.md) §1、§3.2、§22）に従い、次をエンジンと標準構成に含めません。

- **エンジン内のnetwork code:** ルールコア、合法手生成、探索、自己対局、Arena、学習は、HTTPやWebSocketのclientと、それらを提供する依存を持ちません。
- **ゲームサーバーとの接続:** TETR.IOのゲームサーバーへ接続して操作するadapterを持ちません。
- **非公開情報の取得:** DOM、memory、非公開endpointのscrapingを行いません。
- **人間playerが観測できない情報の利用:** RNGの状態、preview以降のhidden queue、未着弾garbageのhidden hole columnなどは、シミュレータ内部にだけ存在します。`include/tetra/observation.hpp` の `observe()` でmodel入力から除外し、`tests/test_observation.cpp` で回帰を検証しています。

## TETRA CHANNEL APIと学習データ収集

文書化されたTETRA CHANNELの公開API（`https://ch.tetr.io/api`）は、学習データの収集や分析に利用します。現在の入口は、人間リプレイによる事前学習用の `trainer/collect_xplus_replays.py` です。

- 収集ツールはエンジンと標準ビルドから分離しており、明示的に実行した場合だけ動作します。
- リクエスト間隔を空け、取得結果を再開可能なledgerに記録します。文書化されていない本体ゲームのAPIは使わず、proxyの切り替えも行いません。
- リプレイ本体は、設定可能なURL templateから取得します。同意済みまたは私的なsourceを使う場合は `--replay-url-template` または `TETRA_REPLAY_URL_TEMPLATE` で指定します。
- 取得したデータはローカルで正規化・検証してから学習に使います。対局中のゲームへ入力を送ることはありません。

詳細は[人間リプレイによる事前学習](HUMAN_REPLAY_PRETRAINING.md#x-corpusを収集する)を参照してください。

## 公開・競技環境での利用

このコードまたは派生物を、公開TETR.IOサーバーやranked/competitive playでの自動操作やsolver assistanceに使わないでください。

将来、公開環境でbotを動作させる必要が生じた場合は、対象サービスの運営者から明示的な事前許可を得たうえで、標準ビルドから分離したopt-inのadapterとして設計します。

## 将来connection adapterを追加する場合

[SPEC](SPEC.md)の方針どおり、connection adapterは標準ビルドで無効な独立したopt-in moduleとします。少なくとも次のcore componentへ直接linkしません。

- `core-rules`
- `movegen`
- 既定の開発ツール

ゲームルール、探索、学習のコードをnetwork integrationから分離し、ローカル・オフラインのシミュレータとしての再現性を保ちます。
