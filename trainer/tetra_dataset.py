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
    aux_valid_mask [N, aux_targets] (v3)
    provenance    per-sample perspective/termination/game seed (v3+)
    chosen_action [N] int32, -1 when unavailable (v4+)

Only numpy is required to read it; torch is optional and used lazily.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = b"TETRADAT"
LEGACY_VERSION = 1
COMPACT_VERSION = 2
RECTANGULAR_V3 = 3
VERSION = 4
CONTRACT_VERSION = 1
BASE_HEADER = struct.Struct("<8s7IQI")
CONTRACT_HEADER = struct.Struct("<IIQQIIIIQQ")

TOKENIZER_SCHEMA_VERSION = 2
TOKENIZER_SCHEMA_HASH = 0x5F1E2C9A7B43D816
OBSERVATION_SCHEMA_HASH = 0x8C74B1E2D6093A5F
ACTION_SCHEMA_VERSION = 1
LEGACY_AUX_TARGET_SCHEMA_VERSION = 1
INTERVAL_AUX_TARGET_SCHEMA_VERSION = 2
GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION = 3
AUX_TARGET_SCHEMA_VERSION = 4
LEGACY_AUX_TARGET_COUNT = 4
INTERVAL_AUX_TARGET_COUNT_V2 = 36
AUX_TARGET_COUNT_V3 = 44
AUX_TARGET_COUNT = 52


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
    contract_version: int = 0
    tokenizer_schema_version: int = TOKENIZER_SCHEMA_VERSION
    tokenizer_schema_hash: int = TOKENIZER_SCHEMA_HASH
    observation_schema_hash: int = OBSERVATION_SCHEMA_HASH
    action_schema_version: int = ACTION_SCHEMA_VERSION
    aux_target_schema_version: int = LEGACY_AUX_TARGET_SCHEMA_VERSION
    randomizer_type: int = 0
    termination_reason: int = 0
    self_play_seed: int = 0
    token_kind_order_hash: int = TOKENIZER_SCHEMA_HASH


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
    aux_valid_mask: Optional[np.ndarray] = None  # [N, aux], 1 = known target
    player_perspective: Optional[np.ndarray] = None  # [N], +1/-1
    termination_reason: Optional[np.ndarray] = None  # [N], enum value
    game_seed: Optional[np.ndarray] = None  # [N], zero for legacy files
    move_number: Optional[np.ndarray] = None  # [N]
    chosen_action: Optional[np.ndarray] = None  # [N], -1 for pre-v4 data

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
            if h.version not in (RECTANGULAR_V3, VERSION) or h0.version not in (RECTANGULAR_V3, VERSION):
                raise ValueError("cannot mix non-rectangular dataset versions")
            contract_fields = (
                "contract_version", "token_features", "action_features",
                "tokenizer_schema_version", "tokenizer_schema_hash",
                "observation_schema_hash", "action_schema_version",
                "token_kind_order_hash",
            )
            for field in contract_fields:
                if getattr(h, field) != getattr(h0, field):
                    raise ValueError(f"cannot mix datasets with different {field}")

        # Auxiliary schemas are append-only by contract: v1=4, v2=36,
        # v3=44, v4=52. Older rows can therefore be widened safely by padding
        # the appended targets with validity=0. This is especially useful for
        # policy-only replay across historical generations; no unavailable
        # auxiliary label is ever fabricated.
        aux_contract = {
            LEGACY_AUX_TARGET_SCHEMA_VERSION: LEGACY_AUX_TARGET_COUNT,
            INTERVAL_AUX_TARGET_SCHEMA_VERSION: INTERVAL_AUX_TARGET_COUNT_V2,
            GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION: AUX_TARGET_COUNT_V3,
            AUX_TARGET_SCHEMA_VERSION: AUX_TARGET_COUNT,
        }
        for ds in datasets:
            expected = aux_contract.get(ds.header.aux_target_schema_version)
            if expected is None or ds.header.aux_targets != expected:
                raise ValueError(
                    "cannot mix dataset with unsupported auxiliary schema: "
                    f"v{ds.header.aux_target_schema_version}/{ds.header.aux_targets}"
                )

        total = sum(len(ds) for ds in datasets)
        max_tokens = max(ds.tokens.shape[1] for ds in datasets)
        max_actions = max(ds.actions.shape[1] for ds in datasets)
        ftok = h0.token_features
        fact = h0.action_features
        faux = max(ds.header.aux_targets for ds in datasets)
        faux_schema = max(ds.header.aux_target_schema_version for ds in datasets)

        tokens = np.zeros((total, max_tokens, ftok), dtype=np.float32)
        token_mask = np.zeros((total, max_tokens), dtype=np.float32)
        actions = np.zeros((total, max_actions, fact), dtype=np.float32)
        action_mask = np.zeros((total, max_actions), dtype=np.float32)
        policy_target = np.zeros((total, max_actions), dtype=np.float32)
        value_target = np.zeros((total,), dtype=np.float32)
        aux_target = np.zeros((total, faux), dtype=np.float32)
        aux_valid_mask = np.zeros((total, faux), dtype=np.float32)
        player_perspective = np.ones((total,), dtype=np.int32)
        termination_reason = np.zeros((total,), dtype=np.int32)
        game_seed = np.zeros((total,), dtype=np.uint64)
        move_number = np.zeros((total,), dtype=np.uint32)
        chosen_action = np.full((total,), -1, dtype=np.int32)

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
            aux_width = ds.header.aux_targets
            aux_target[sl, :aux_width] = ds.aux_target
            aux_valid_mask[sl, :aux_width] = ds.aux_valid_mask
            player_perspective[sl] = ds.player_perspective
            termination_reason[sl] = ds.termination_reason
            game_seed[sl] = ds.game_seed
            move_number[sl] = ds.move_number
            chosen_action[sl] = ds.chosen_action
            at += n

        header = Header(
            version=max(ds.header.version for ds in datasets),
            samples=total,
            max_tokens=max_tokens,
            max_actions=max_actions,
            token_features=ftok,
            action_features=fact,
            aux_targets=faux,
            ruleset_hash=h0.ruleset_hash,
            model_version=max(ds.header.model_version for ds in datasets),
            contract_version=h0.contract_version,
            tokenizer_schema_version=h0.tokenizer_schema_version,
            tokenizer_schema_hash=h0.tokenizer_schema_hash,
            observation_schema_hash=h0.observation_schema_hash,
            action_schema_version=h0.action_schema_version,
            aux_target_schema_version=faux_schema,
            randomizer_type=h0.randomizer_type,
            termination_reason=h0.termination_reason,
            self_play_seed=0,
            token_kind_order_hash=h0.token_kind_order_hash,
        )
        return Dataset(header, tokens, token_mask, actions, action_mask,
                       policy_target, value_target, aux_target, aux_valid_mask,
                       player_perspective, termination_reason, game_seed, move_number,
                       chosen_action)

    def torch(self, device: Optional[str] = None):
        """Return the arrays as torch tensors, moved to `device` if given."""
        import torch  # imported lazily so numpy-only use needs no torch

        def t(a):
            # Dataset arrays are views over an immutable
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
            "aux_valid_mask": t(self.aux_valid_mask),
            "chosen_action": t(self.chosen_action),
        }

    def sanity_check(self) -> None:
        """Fail loudly on the mistakes that silently ruin training."""
        h = self.header
        if h.version in (RECTANGULAR_V3, VERSION):
            if h.contract_version != CONTRACT_VERSION:
                raise ValueError(f"unsupported dataset contract version {h.contract_version}")
            if (h.tokenizer_schema_version != TOKENIZER_SCHEMA_VERSION or
                    h.tokenizer_schema_hash != TOKENIZER_SCHEMA_HASH or
                    h.token_kind_order_hash != TOKENIZER_SCHEMA_HASH or
                    h.observation_schema_hash != OBSERVATION_SCHEMA_HASH or
                    h.action_schema_version != ACTION_SCHEMA_VERSION):
                raise ValueError("tokenizer/observation/action schema mismatch")
            if (h.aux_target_schema_version == AUX_TARGET_SCHEMA_VERSION and
                    h.aux_targets != AUX_TARGET_COUNT):
                raise ValueError("aux target width does not match schema v4")
            if (h.aux_target_schema_version == GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION and
                    h.aux_targets != AUX_TARGET_COUNT_V3):
                raise ValueError("aux target width does not match schema v3")
            if (h.aux_target_schema_version == INTERVAL_AUX_TARGET_SCHEMA_VERSION and
                    h.aux_targets != INTERVAL_AUX_TARGET_COUNT_V2):
                raise ValueError("aux target width does not match schema v2")
            if (h.aux_target_schema_version == LEGACY_AUX_TARGET_SCHEMA_VERSION and
                    h.aux_targets != LEGACY_AUX_TARGET_COUNT):
                raise ValueError("legacy aux target width does not match schema v1")
            if h.aux_target_schema_version not in (
                    AUX_TARGET_SCHEMA_VERSION, GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION,
                    INTERVAL_AUX_TARGET_SCHEMA_VERSION, LEGACY_AUX_TARGET_SCHEMA_VERSION):
                raise ValueError("unknown aux target schema")
        assert self.tokens.shape == (h.samples, h.max_tokens, h.token_features), self.tokens.shape
        assert self.actions.shape == (h.samples, h.max_actions, h.action_features)
        assert self.policy_target.shape == (h.samples, h.max_actions)
        assert self.value_target.shape == (h.samples,)
        assert self.aux_target.shape == (h.samples, h.aux_targets)
        if self.aux_valid_mask is None:
            self.aux_valid_mask = np.ones_like(self.aux_target, dtype=np.float32)
        if self.player_perspective is None:
            self.player_perspective = np.ones((h.samples,), dtype=np.int32)
        if self.termination_reason is None:
            self.termination_reason = np.zeros((h.samples,), dtype=np.int32)
        if self.game_seed is None:
            self.game_seed = np.zeros((h.samples,), dtype=np.uint64)
        if self.move_number is None:
            self.move_number = np.zeros((h.samples,), dtype=np.uint32)
        if self.chosen_action is None:
            self.chosen_action = np.full((h.samples,), -1, dtype=np.int32)
        if self.aux_valid_mask.shape != (h.samples, h.aux_targets):
            raise ValueError(f"aux_valid_mask has shape {self.aux_valid_mask.shape}")
        for name, arr, shape in (
            ("player_perspective", self.player_perspective, (h.samples,)),
            ("termination_reason", self.termination_reason, (h.samples,)),
            ("game_seed", self.game_seed, (h.samples,)),
            ("move_number", self.move_number, (h.samples,)),
            ("chosen_action", self.chosen_action, (h.samples,)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")

        for name, arr in (
            ("tokens", self.tokens),
            ("actions", self.actions),
            ("policy_target", self.policy_target),
            ("value_target", self.value_target),
            ("aux_target", self.aux_target),
            ("aux_valid_mask", self.aux_valid_mask),
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

        if ((self.aux_valid_mask < -1e-6) | (self.aux_valid_mask > 1.0 + 1e-6)).any():
            raise ValueError("aux_valid_mask must contain only 0/1 values")

    def target_statistics(self) -> dict[str, np.ndarray]:
        """Return finite, mask-aware statistics for the pre-training audit."""
        self.sanity_check()
        mask = self.aux_valid_mask > 0.5
        mean = np.zeros((self.header.aux_targets,), dtype=np.float64)
        std = np.zeros_like(mean)
        zero = np.zeros_like(mean)
        valid_rate = mask.mean(axis=0)
        percentile = np.zeros((self.header.aux_targets, 3), dtype=np.float64)
        for i in range(self.header.aux_targets):
            values = self.aux_target[:, i][mask[:, i]].astype(np.float64)
            if values.size:
                mean[i] = values.mean()
                std[i] = values.std()
                zero[i] = np.mean(np.isclose(values, 0.0))
                percentile[i] = np.percentile(values, [50, 90, 99])
        return {"mean": mean, "std": std, "zero_rate": zero,
                "valid_rate": valid_rate, "percentile": percentile}


def load(path: str) -> Dataset:
    with open(path, "rb") as fh:
        base = fh.read(BASE_HEADER.size)

    if len(base) != BASE_HEADER.size or base[:8] != MAGIC:
        raise ValueError("not a .tetradat file")
    ver = BASE_HEADER.unpack(base)[1]

    if ver == LEGACY_VERSION:
        with open(path, "rb") as fh:
            blob = fh.read()
    elif ver == COMPACT_VERSION:
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
    elif ver in (RECTANGULAR_V3, VERSION):
        with open(path, "rb") as fh:
            blob = fh.read()
    else:
        raise ValueError(f"unsupported dataset version {ver}")

    magic, version, samples, max_tokens, max_actions, token_features, action_features, aux_targets, ruleset_hash, model_version = BASE_HEADER.unpack_from(blob, 0)
    off = BASE_HEADER.size

    if version in (RECTANGULAR_V3, VERSION):
        if len(blob) < off + CONTRACT_HEADER.size:
            raise ValueError("truncated dataset contract header")
        (contract_version, tokenizer_schema_version, tokenizer_schema_hash,
         observation_schema_hash, action_schema_version, aux_target_schema_version,
         randomizer_type, termination_reason, self_play_seed,
         token_kind_order_hash) = CONTRACT_HEADER.unpack_from(blob, off)
        off += CONTRACT_HEADER.size
    else:
        contract_version = 0
        tokenizer_schema_version = TOKENIZER_SCHEMA_VERSION
        tokenizer_schema_hash = TOKENIZER_SCHEMA_HASH
        observation_schema_hash = OBSERVATION_SCHEMA_HASH
        action_schema_version = ACTION_SCHEMA_VERSION
        aux_target_schema_version = LEGACY_AUX_TARGET_SCHEMA_VERSION
        randomizer_type = 0
        termination_reason = 0
        self_play_seed = 0
        token_kind_order_hash = TOKENIZER_SCHEMA_HASH

    header = Header(
        version=version,
        samples=samples,
        max_tokens=max_tokens,
        max_actions=max_actions,
        token_features=token_features,
        action_features=action_features,
        aux_targets=aux_targets,
        ruleset_hash=ruleset_hash,
        model_version=model_version,
        contract_version=contract_version,
        tokenizer_schema_version=tokenizer_schema_version,
        tokenizer_schema_hash=tokenizer_schema_hash,
        observation_schema_hash=observation_schema_hash,
        action_schema_version=action_schema_version,
        aux_target_schema_version=aux_target_schema_version,
        randomizer_type=randomizer_type,
        termination_reason=termination_reason,
        self_play_seed=self_play_seed,
        token_kind_order_hash=token_kind_order_hash,
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

    if version in (RECTANGULAR_V3, VERSION):
        aux_valid_mask = take(n * faux, (n, faux))
        def take_raw(count, dtype, shape):
            nonlocal off
            itemsize = np.dtype(dtype).itemsize
            arr = np.frombuffer(blob, dtype=dtype, count=count, offset=off).reshape(shape)
            off += count * itemsize
            return arr
        player_perspective = take_raw(n, "<i4", (n,))
        termination_per_sample = take_raw(n, "<i4", (n,))
        game_seed = take_raw(n, "<u8", (n,))
        move_number = take_raw(n, "<u4", (n,))
        if version == VERSION:
            chosen_action = take_raw(n, "<i4", (n,))
        else:
            chosen_action = np.full((n,), -1, dtype=np.int32)
    else:
        aux_valid_mask = np.ones_like(aux_target, dtype=np.float32)
        player_perspective = np.ones((n,), dtype=np.int32)
        termination_per_sample = np.zeros((n,), dtype=np.int32)
        game_seed = np.zeros((n,), dtype=np.uint64)
        move_number = np.zeros((n,), dtype=np.uint32)
        chosen_action = np.full((n,), -1, dtype=np.int32)

    return Dataset(
        header=header,
        tokens=tokens,
        token_mask=token_mask,
        actions=actions,
        action_mask=action_mask,
        policy_target=policy_target,
        value_target=value_target,
        aux_target=aux_target,
        aux_valid_mask=aux_valid_mask,
        player_perspective=player_perspective,
        termination_reason=termination_per_sample,
        game_seed=game_seed,
        move_number=move_number,
        chosen_action=chosen_action,
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
