# SPDX-License-Identifier: MIT
"""Reader for the `.tetradat` training sets exported by the C++ engine.

The engine writes the same rectangular, padded layout that its own inference
path uses (see `include/tetra/batch.hpp`), so nothing here reshapes or
re-pads -- the arrays are wrapped as-is. That is the point: the trainer and the
engine cannot drift apart on shapes, masking or feature order.

The file is a small header followed by contiguous little-endian float32
blocks::

    tokens        [N, T, token_features]
    token_mask    [N, T]
    actions       [N, A, action_features]
    action_mask   [N, A]
    policy_target [N, A]
    value_target  [N]
    aux_target    [N, aux_targets]

Only numpy is required to read it; torch is optional and used lazily.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = b"TETRADAT"
VERSION = 1


@dataclass
class Header:
    version: int
    samples: int
    max_tokens: int
    max_actions: int
    token_features: int
    action_features: int
    aux_targets: int
    ruleset_hash: int
    model_version: int


@dataclass
class Dataset:
    """A padded training set. Every array's first axis is the sample index."""

    header: Header
    tokens: np.ndarray  # [N, T, F_tok] float32
    token_mask: np.ndarray  # [N, T]    float32, 1 = real, 0 = padding
    actions: np.ndarray  # [N, A, F_act] float32
    action_mask: np.ndarray  # [N, A]   float32
    policy_target: np.ndarray  # [N, A] float32, rows sum to 1 over real actions
    value_target: np.ndarray  # [N]     float32 in [-1, 1]
    aux_target: np.ndarray  # [N, aux]  float32

    def __len__(self) -> int:
        return self.header.samples

    def action_counts(self) -> np.ndarray:
        """Number of legal actions per sample."""
        return self.action_mask.sum(axis=1).astype(np.int64)

    @staticmethod
    def concatenate(datasets: list["Dataset"]) -> "Dataset":
        """Merge replay generations, padding them to one rectangular layout.

        Generations can have different maximum token/action counts.  Padding
        here preserves each sample's masks and keeps the trainer's model input
        contract identical to a single exported dataset.  All inputs must use
        the same ruleset and feature widths; mixing rulesets would make the
        value target and action semantics invalid.
        """
        if not datasets:
            raise ValueError("cannot concatenate an empty dataset list")
        for ds in datasets:
            ds.sanity_check()

        first = datasets[0]
        h0 = first.header
        for ds in datasets[1:]:
            h = ds.header
            if h.ruleset_hash != h0.ruleset_hash:
                raise ValueError(
                    "cannot mix datasets with different ruleset hashes: "
                    f"{h0.ruleset_hash:016x} vs {h.ruleset_hash:016x}"
                )
            if (h.token_features != h0.token_features or
                    h.action_features != h0.action_features or
                    h.aux_targets != h0.aux_targets):
                raise ValueError("cannot mix datasets with different feature widths")

        total = sum(len(ds) for ds in datasets)
        max_tokens = max(ds.tokens.shape[1] for ds in datasets)
        max_actions = max(ds.actions.shape[1] for ds in datasets)
        ftok = h0.token_features
        fact = h0.action_features
        faux = h0.aux_targets

        tokens = np.zeros((total, max_tokens, ftok), dtype=np.float32)
        token_mask = np.zeros((total, max_tokens), dtype=np.float32)
        actions = np.zeros((total, max_actions, fact), dtype=np.float32)
        action_mask = np.zeros((total, max_actions), dtype=np.float32)
        policy_target = np.zeros((total, max_actions), dtype=np.float32)
        value_target = np.zeros((total,), dtype=np.float32)
        aux_target = np.zeros((total, faux), dtype=np.float32)

        at = 0
        for ds in datasets:
            n = len(ds)
            t = ds.tokens.shape[1]
            a = ds.actions.shape[1]
            sl = slice(at, at + n)
            tokens[sl, :t] = ds.tokens
            token_mask[sl, :t] = ds.token_mask
            actions[sl, :a] = ds.actions
            action_mask[sl, :a] = ds.action_mask
            policy_target[sl, :a] = ds.policy_target
            value_target[sl] = ds.value_target
            aux_target[sl] = ds.aux_target
            at += n

        header = Header(
            version=1,
            samples=total,
            max_tokens=max_tokens,
            max_actions=max_actions,
            token_features=ftok,
            action_features=fact,
            aux_targets=faux,
            ruleset_hash=h0.ruleset_hash,
            model_version=max(ds.header.model_version for ds in datasets),
        )
        return Dataset(header, tokens, token_mask, actions, action_mask,
                       policy_target, value_target, aux_target)

    def torch(self, device: Optional[str] = None):
        """Return the arrays as torch tensors, moved to `device` if given."""
        import torch  # imported lazily so numpy-only use needs no torch

        def t(a):
            # Version-1 datasets are memory-mapped views over an immutable
            # bytes object.  Make a writable contiguous copy before handing it
            # to torch so accelerator backends never see an unsafe tensor view.
            x = torch.from_numpy(np.array(a, copy=True, order="C"))
            return x.to(device) if device else x

        return {
            "tokens": t(self.tokens),
            "token_mask": t(self.token_mask),
            "actions": t(self.actions),
            "action_mask": t(self.action_mask),
            "policy_target": t(self.policy_target),
            "value_target": t(self.value_target),
            "aux_target": t(self.aux_target),
        }

    def sanity_check(self) -> None:
        """Fail loudly on the mistakes that silently ruin training."""
        h = self.header
        assert self.tokens.shape == (h.samples, h.max_tokens, h.token_features), self.tokens.shape
        assert self.actions.shape == (h.samples, h.max_actions, h.action_features)
        assert self.policy_target.shape == (h.samples, h.max_actions)
        assert self.value_target.shape == (h.samples,)

        for name, arr in (
            ("tokens", self.tokens),
            ("actions", self.actions),
            ("policy_target", self.policy_target),
            ("value_target", self.value_target),
            ("aux_target", self.aux_target),
        ):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values")

        # Padding must be exactly zero, or a model that forgets the mask would
        # still appear to work on this data and then fail elsewhere.
        pad = self.token_mask[..., None] == 0
        if self.tokens[np.broadcast_to(pad, self.tokens.shape)].any():
            raise ValueError("padded token slots are not zeroed")

        # Every row must be a probability distribution over its real actions.
        sums = self.policy_target.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            bad = int(np.argmax(np.abs(sums - 1.0)))
            raise ValueError(f"policy row {bad} sums to {sums[bad]}, not 1")

        # Policy mass must never sit on a padded action.
        if (self.policy_target * (1.0 - self.action_mask)).any():
            raise ValueError("policy mass assigned to padded actions")

        if (np.abs(self.value_target) > 1.0 + 1e-6).any():
            raise ValueError("value targets outside [-1, 1]")


def load(path: str) -> Dataset:
    with open(path, "rb") as fh:
        magic = fh.read(8)
        ver_bytes = fh.read(4)

    if len(magic) < 8 or magic != MAGIC:
        raise ValueError("not a .tetradat file")
    (ver,) = struct.unpack("<I", ver_bytes)

    if ver == 1:
        with open(path, "rb") as fh:
            blob = fh.read()
    elif ver == 2:
        # Compact Replay + π format (spec 13.5, 17, ADR 0012):
        # Decode on-the-fly via the C++ engine to eliminate padded tensor disk I/O.
        import os
        import subprocess

        cli_base = "tetra_cli"
        # On Windows, g++ produces tetra_cli.exe; add the extension if present.
        if os.name == "nt":
            cli_candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", f"{cli_base}.exe"),
                os.path.join(os.getcwd(), "build", f"{cli_base}.exe"),
            ]
        else:
            cli_candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", cli_base),
                os.path.join(os.getcwd(), "build", cli_base),
            ]
        cli_path = None
        for cand in cli_candidates:
            if os.path.exists(cand):
                cli_path = cand
                break
        if not cli_path:
            raise RuntimeError("build/tetra_cli not found; run 'make tools' first to read v2 dataset")

        proc = subprocess.run(
            [cli_path, "decode-dataset", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        blob = proc.stdout
    else:
        raise ValueError(f"unsupported dataset version {ver}")

    off = 8
    fields = struct.unpack_from("<7I", blob, off)
    off += 28
    (ruleset_hash,) = struct.unpack_from("<Q", blob, off)
    off += 8
    (model_version,) = struct.unpack_from("<I", blob, off)
    off += 4

    header = Header(
        version=1,  # in-memory unpadded structure uses version 1 layout
        samples=fields[1],
        max_tokens=fields[2],
        max_actions=fields[3],
        token_features=fields[4],
        action_features=fields[5],
        aux_targets=fields[6],
        ruleset_hash=ruleset_hash,
        model_version=model_version,
    )

    n, t, a = header.samples, header.max_tokens, header.max_actions
    ftok, fact, faux = header.token_features, header.action_features, header.aux_targets

    def take(count, shape):
        nonlocal off
        arr = np.frombuffer(blob, dtype="<f4", count=count, offset=off).reshape(shape)
        off += count * 4
        return arr

    tokens = take(n * t * ftok, (n, t, ftok))
    token_mask = take(n * t, (n, t))
    actions = take(n * a * fact, (n, a, fact))
    action_mask = take(n * a, (n, a))
    policy_target = take(n * a, (n, a))
    value_target = take(n, (n,))
    aux_target = take(n * faux, (n, faux))

    return Dataset(
        header=header,
        tokens=tokens,
        token_mask=token_mask,
        actions=actions,
        action_mask=action_mask,
        policy_target=policy_target,
        value_target=value_target,
        aux_target=aux_target,
    )


if __name__ == "__main__":
    import sys

    ds = load(sys.argv[1])
    ds.sanity_check()
    counts = ds.action_counts()
    print(f"samples        {len(ds)}")
    print(f"tokens         {ds.tokens.shape}  (real: {int(ds.token_mask.sum(1).mean())} mean)")
    print(f"actions        {ds.actions.shape}  (real: {counts.mean():.1f} mean, {counts.max()} max)")
    print(f"ruleset_hash   {ds.header.ruleset_hash:016x}")
    print(f"value target   mean {ds.value_target.mean():+.3f}")
    print("sanity check   OK")
