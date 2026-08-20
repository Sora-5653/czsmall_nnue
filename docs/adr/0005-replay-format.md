# ADR 0005: Replayにはderived stateではなくinputを記録する

## 状態

採用。

## 背景

Spec §18.1はrule consistencyを最重要要件とし、§22は微妙なrule driftへの対策として「replay diff testing」を挙げています。Spec §17は `/protocol` でProtobufを使う案を示していましたが、この時点ではreplayは単一test内のin-memory placement vectorとしてしか存在しませんでした。

## 決定

### Inputを記録し、derived stateを保存しない

Replayは次を保存します。

- seed
- ruleset hash
- placement sequence

board、attack、timingなどはplayback時に再計算します。

もしderived state自体を保存してそれを再生するだけなら、verificationはsimulatorの検証ではなく「cached resultを正しく読めたか」の確認になってしまいます。

### Spin provenanceだけは明示的に記録する

同じnotchへslideで入ったpieceとrotationで入ったpieceは、同じcellを占有してもspin scoreが異なります。final positionだけからspin classを一意に復元できないため、spin provenanceはreplayへ記録します。

verifierはrecorded spinをそのまま信頼するのではなく、可能な範囲でspinを再導出してrecorded valueと比較します。これによりspin-detection regressionを検出できます。

### Checkpointをinterleaveする

N placementsごとに次をcheckpointとして持ちます。

- board hash
- sent / received line count
- timestamp

これにより、最終stateだけでなく**最初にdivergeしたplacement**を報告できます。checkpoint intervalはfile sizeとlocalizationのtrade-offです。

### 現段階ではcustom binary chunkを使う

明示little-endian、versioned format、trailing FNV-1a checksumを使います。

この時点ではProtobuf toolchainをsandboxへ取得できず、cross-language wire compatibilityよりdependency-free buildを優先しました。field layoutは将来 `.proto` へ移しやすい形に保ちます。

## 帰結

- 約19–21 bytes / placement。300-piece gameで約5.5 KBだった。
- `verify_replay()` は `first_divergence` とexpected/actual board hashを報告できる。
- 次のfailure modeを区別して検出できる。
  - ruleset hash mismatch
  - piece-sequence divergence（randomizer change）
  - checkpoint mismatch（rule change）
  - checksum mismatch（file corruption）
- format実装直後、verifierが全replayed pieceを「rotated」と扱い、存在しないspinを生成するbugを発見した。53-placement gameのplacement 16でdivergeしたことで局所化できた。

## 検討した代替案

### 毎placement後のboardを保存する

再生結果を保存値と置き換えるだけになり、rule engineのdriftを検出できません。

### 最初からProtobufを導入する

当時のsandboxではtoolchainを取得できず、現在必要のないcross-language wire compatibilityのためにdependencyを増やすことになるため見送りました。
