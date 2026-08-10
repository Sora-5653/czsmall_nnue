# ADR 0012: Compact dataset（Replay + π）とload時token再生成

## 状態

採用。

**追補:** `VERSION_COMPACT = 2` は現在もsupportedですが、現行通常datasetは `DatasetHeader::VERSION = 3` です。特にtwo-board self-playはopponent event streamをcompact v2 metadataだけで再構成できないため、schema/termination metadataを持つrectangular v3を使用します。

## 背景

ADR 0010では、C++ inference、on-disk `.tetradat`、PyTorch trainerで共通のpadded rectangular `TensorBatch` contractを導入しました。

初期 `DatasetHeader::VERSION = 1` は、各sampleのfully padded float32 tensorをそのままdiskへ保存していました。

- `tokens`: `[max_tokens, TOKEN_FEATURES]`
  - 当時 `80 * 24 * 4 = 7.68 KB / sample`
- `actions`: `[max_actions, ACTION_FEATURES]`
  - 当時 `40 * 24 * 4 = 3.84 KB / sample`
- mask・auxiliary targetを含め、約12.2 KB / sample

100,000+ samples級のself-playでは `.tetradat` が容易に1 GBを超え、GPU computeより先にdisk capacityとI/Oがbottleneckになる見込みでした。

## 決定

再構成可能なsample向けに `DatasetHeader::VERSION_COMPACT = 2` を導入します。

Version 2ではpre-generated token/action tensorを保存せず、**Replay + π** を保存します。

### 1. Provenance / replay information

- `ruleset_hash`
- `game_seed`
- `move_number`
- `chosen_action`
- garbage schedule parameter
  - `garbage_style`
  - `garbage_period`
  - `garbage_lines`

### 2. Search / training target

導入時には次を保存しました。

- `search_policy`
- `outcome`
- `search_value`
- auxiliary target
  - `time_to_terminal`
  - `future_attack_1s`
  - `future_garbage_received`
  - top-out horizonなど

後にaux schemaはversion 2 / 36 targetsへ拡張されています。compact compatibilityを判断するときは**当時のfield listだけでなくschema metadata**も確認します。

## Load時の再生成

### C++

`deserialize_dataset` がcompact v2を検出すると、`deserialize_compact_dataset` がrecordを `(ruleset_hash, game_seed)` ごとにgroup化します。

各gameをmove 0から `Player`、`MoveGenerator`、`Tokenizer` で再simulateし、recorded placement時点のtokenized observationとlegal action embeddingを再生成します。

目的は、保存前とbit-for-bit同じ `TensorBatch` を復元することです。

### Python

`trainer/tetra_dataset.py` はversion 2を検出すると、engine toolの

```sh
./build/tetra_cli decode-dataset <path>
```

をsubprocessで使い、C++側で再構成したunpadded tensor streamをnumpyへ読み込みます。中間rectangular fileをdiskへ書き出しません。

## 導入時の効果

- **disk/I/O:** 約12,200 bytes/sample → 約95 bytes/sampleで100倍以上縮小。
- 100,000 samples: 約1.2 GB → 約10 MB級。
- **CPU regeneration:** 当時約30,000 placements/sで、30,000 samplesのloadへ1秒未満の追加cost。
- **backward compatibility:** version 1とversion 2をreaderが扱える構成を維持。

これらは導入当時のsingle-board再構成pathのhistorical measurementです。

## なぜ現行two-board self-playはcompact v2ではないか

二盤面自己対局のobservationには、相手board/counterだけでなくattack deliveryへ影響する相手側event historyが必要です。

現在のcompact Replay metadataは、1 player側のplacement sequenceだけからそのopponent event streamを完全には復元できません。

不完全な再構成でstorageを小さくするより、**正しいtraining sampleを保持すること**を優先し、two-board GPU self-playはrectangular version 3へfallbackします。

Version 3はtoken/action tensorを保存しますが、次のmetadataを追加しています。

- tokenizer schema version/hash
- observation schema hash
- action schema version
- auxiliary target schema version/count
- termination reason
- player perspective / provenanceに必要なfield

したがって現在の設計は「compactが常に優れている」ではなく、**正確に再構成できる経路だけcompact化する**というものです。

## 帰結

- single-board等の再構成可能dataでは大幅なstorage reductionを得られる。
- two-board semanticsを壊してまでcompact化しない。
- dataset formatはversioned contractとして進化し、reader側はcompatibilityを明示的に検証する。
- storage optimizationより、observation correctness・schema provenanceを上位に置く。
