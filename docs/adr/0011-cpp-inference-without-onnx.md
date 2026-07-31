# ADR 0011: C++ inference without ONNX Runtime

## Status
Accepted.

## Context
Training happens in PyTorch; play happens in the C++ search. Something has to
run the trained weights inside the engine. Spec §17 nominates ONNX as the
inference format.

## Decision
Implement the forward pass directly in C++ (`nnue.hpp`), loading weights from a
`.tetrawts` file written by `trainer/export_weights.py`. No third-party runtime.

Reasons, in order of weight:

1. **`git clone && make` keeps working.** The engine has no dependencies today
   and its 263 tests run anywhere with a C++17 compiler. Adding ONNX Runtime
   would make the build, and CI, contingent on a large binary dependency.
2. **The ROCm/RDNA 4 toolchain is still settling.** `gfx1201` only became
   officially supported in ROCm 7.2. Coupling the *engine* to that stack would
   make the whole project hostage to it; keeping inference dependency-free means
   a toolchain problem blocks training only.
3. **Self-play generation is latency-bound at small batch sizes**, where a
   heavyweight runtime's dispatch overhead is a real cost.

The trade is accepted deliberately: this path is scalar CPU code and will be
slow for a spec-sized model. Training and large-scale evaluation belong on the
GPU in Python.

## The risk this creates, and the mitigation

A hand-written forward pass can silently disagree with the one that was trained.
That failure is invisible — no crash, no error, just a bot mysteriously weaker
than its validation loss implies.

`cpp_matches_pytorch_exactly` pins the two together against a committed fixture
(a small model plus a reference forward pass). Measured agreement: **3e-08**
maximum absolute difference on the policy distribution. If anyone changes
either implementation, that test fails before the divergence can reach a
training run.

The loader also refuses weights whose `token_features` / `action_features` do
not match the engine, so a token-layout change cannot be silently misread as
valid data.

## If this becomes the bottleneck

An ONNX Runtime or libtorch backend can be added behind the same `Evaluator`
interface without touching the search, exactly as `TetraFormerEvaluator` was.
The parity fixture then serves both backends.