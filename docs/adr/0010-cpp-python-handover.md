# ADR 0010: The C++/Python handover, and fixed-shape batching

## Status
Accepted.

## Context
With the search, self-play loop and replay buffer in place, one structural gap
remained before a Transformer could be attached: **everything upstream is ragged
and a Transformer needs rectangles.**

Measured over 720 self-play samples:

| | min | max | mean |
|---|---|---|---|
| state tokens | 48 | 65 | 63.6 |
| legal actions | 9 | 70 | 42.5 |

Somewhere the ragged data has to be padded and masked. The question was where.

## Decision: pad and mask in the engine, once

`TensorBatch` (in `batch.hpp`) produces the rectangular form, and the *same*
structure serves three consumers: C++ inference, the on-disk training set, and
the PyTorch trainer. Doing it here rather than in each backend means:

* the padding convention is defined and tested in one place,
* masks are produced alongside the data, so a backend cannot forget them and
  silently attend to padding, and
* the trainer and the engine cannot drift apart on shapes or feature order.

Layout is row-major and contiguous, so numpy and libtorch wrap it without a
copy:

```
tokens  [B, T, TOKEN_FEATURES]   token_mask  [B, T]
actions [B, A, ACTION_FEATURES]  action_mask [B, A]
```

`pad_tokens` / `pad_actions` force static dimensions for a fixed-shape backend
(ONNX, TensorRT); left at zero the batch is sized to its contents, which is
cheaper on CPU.

## Decision: `.tetradat`, a flat float32 dump

A small self-describing header followed by the batch buffers verbatim. Not
Protobuf, for the same reason as ADR 0005: it costs a toolchain this environment
cannot fetch, to buy cross-language compatibility nothing needs. The reader
(`trainer/tetra_dataset.py`) needs only numpy, and validates on load —
non-finite values, non-zero padding, policy rows that do not sum to 1, and
policy mass on padded actions are all rejected rather than quietly trained on.

The header records `token_features` and `action_features` and refuses to load a
file whose widths disagree with the engine, which is the failure that would
otherwise produce silently garbled training.

## A bug this surfaced

The first end-to-end training run showed the auxiliary loss at **~150** against
a policy loss of ~3: `time_to_terminal` was exported as a raw placement count
reaching several hundred, so under MSE it dominated the total and the model was
fitting essentially nothing else. Auxiliary targets are now squashed into
roughly [0, 1] before leaving the engine, which is also what makes the loss
weights in the trainer mean anything. Pinned by `aux_targets_are_normalised`.

## Result

The whole chain runs from one command:

```sh
./build/tetra_cli export train.tetradat 10 100 16
python trainer/train.py train.tetradat --steps 300
```

`trainer/tetraformer.py` implements the spec §9-10 network: pre-norm RMSNorm
blocks with SwiGLU (§9.5), a **variable-length** policy head where each legal
action cross-attends to the state tokens (§10.1), a WDL value head (§10.2) and
auxiliary regressions. `tetraformer_s()` is the spec size (7.2M parameters);
`tetraformer_dev()` (0.13M) trains at ~25 steps/s on this 2-core CPU.

Held-out loss on engine-generated data: **4.86 → 2.91** over 300 steps. Padded
actions receive exactly probability zero, verified for both model sizes.

## What this does not claim

The model is not *strong*. It is trained on a few hundred samples from a
heuristic-guided search, on a CPU, purely to prove the handover is correct and
that the loss decreases on held-out data. Real training belongs on a GPU, and
strength evaluation needs the Arena and gating (spec §20) that M2 still lacks.