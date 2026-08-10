# ADR 0011: ONNX Runtimeを使わずC++でtrained-weight inferenceを行う

## 状態

採用。

## 背景

trainingはPyTorch、実際のsearch/game loopはC++で動きます。したがって学習済みweightをengine内部で実行する方法が必要です。

Spec §17はinference format候補としてONNXを挙げています。

## 決定

`nnue.hpp` にforward passを直接C++実装し、`trainer/export_weights.py` が出力する `.tetrawts` を読み込みます。

標準C++ engineにthird-party neural runtimeを追加しません。

主な理由:

### 1. dependency-free buildを維持できる

この判断当時、`git clone && make` だけでengine/testを実行できました。ONNX Runtimeを追加するとbuildとCIがlarge binary dependencyへ依存します。

当時は263 testsがC++17 compilerだけで動作していました。これはhistorical countです。現在のbuildはCobra要件により**C++23**、2026-08-10時点の通常 `make test` は282 testsです。

### 2. GPU toolchain問題をengine全体へ波及させない

training GPU stackに問題が起きても、rule engine・CPU inference・replay verificationまで同時にbuild不能になる構造を避けます。

GPU large-scale training/evaluationはPython側に置き、C++ coreは独立にtest可能なまま保ちます。

### 3. small-batch self-playではdispatch overheadも無視できない

self-play searchはsmall batch・latency-sensitiveな場面があります。heavy runtimeの導入が常に有利とは限りません。

代償も明示的に受け入れます。手書きC++ pathはscalar CPU implementationであり、spec-sized modelのlarge-scale evaluationには遅い可能性があります。現在その用途はPython GPU bridgeが担います。

## 手書きforwardが生むrisk

最大のriskは、**C++が学習したnetworkと微妙に違うnetworkを実行してもcrashしない**ことです。

このfailureはvalidation lossには現れず、「なぜかengineだけ弱い」という形になります。

そのため `cpp_matches_pytorch_exactly` でC++ / PyTorch forwardをfixture上に固定します。

導入時の測定ではpolicy distributionのmaximum absolute differenceは **3e-08** でした。

loaderはさらに、weightに記録された `token_features` / `action_features` がengineと一致しない場合にrejectします。

後のschema workではfeature widthだけでなくtokenizer/observation/action/aux schema意味論も管理するようになっています。したがって新しいcheckpoint/dataset pipelineでは「幅が同じだから互換」と判断しないでください。

## 現行の役割分担

- `.pt`: PyTorch training / GPU self-play / GPU Arena向けcheckpoint
- `.tetrawts`: dependency-free C++ evaluator向けexport weight
- C++: rules、search、game mechanics、CPU reference inference
- Python/PyTorch: GPU batched inference、training、architecture ablation

## 将来bottleneckになった場合

ONNX Runtimeやlibtorch backendを、同じ `Evaluator` interfaceの後ろへ追加できます。search側を変更する必要はありません。

その場合もC++ scalar backendとPyTorch fixtureをreferenceとして残し、新backendのparityを同じcontractで検証します。
