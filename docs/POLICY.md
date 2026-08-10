# 利用範囲と運用ポリシー

このリポジトリは、**ローカル・オフラインで動作するTetrisシミュレータおよび研究コード**です。TETR.IO clientではなく、TETR.IOへ接続するコードを含みません。

## このリポジトリに含まれるもの

- 厳密・決定論的・event-drivenなrule core（`/include/tetra`）
- rule core上の合法配置generator
- machine-learning実験用のobservation / tokenizer layer
- ローカルmachine上だけで動作するdeveloper tool

## 意図的に含めないもの

当初仕様の運用条件（spec §1, §3.2, §22）に従い、次を標準構成へ入れません。

- **Network code。** HTTP client、WebSocket client、およびそれらを提供するdependencyを持ちません。
- **TETR.IO API access。** 本プロジェクトはTETR.IO APIを利用せず、標準buildにadapterも含めません。
- **DOM / memory / private endpoint scraping。**
- **人間playerが観測できない情報の利用。** RNG state、preview以後のhidden queue、未着弾garbageのhidden hole columnなどはsimulator内部にのみ存在し、`include/tetra/observation.hpp` の `observe()` でmodel inputから除外します。`tests/test_observation.cpp` で回帰検証しています。

## 公開・競技環境での利用

このコードまたは派生物を、公開TETR.IO serverやranked/competitive playで自動操作・solver assistanceとして利用しないでください。

将来、公開環境でbotを動作させる必要が生じた場合は、対象サービスの運営者から明示的な事前許可を得たうえで、標準buildから分離されたopt-in adapterとして設計する必要があります。

## 将来connection adapterを追加する場合

`SPEC.md` の方針どおり、connection adapterは**標準buildで無効な独立opt-in module**とします。

少なくとも次のcore componentへ直接linkしません。

- `core-rules`
- `movegen`
- default developer tools

ゲームルール・探索・学習コードをnetwork integrationから分離し、ローカルoffline simulatorとしての再現可能性を保ちます。
