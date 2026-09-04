# SPDX-License-Identifier: MIT
"""Unit tests for schema-v4 garbage-cancellation auxiliary supervision."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

try:
    from .tetraformer import _cancellation_auxiliary_loss, losses
except ImportError:  # pragma: no cover - supports direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tetraformer import _cancellation_auxiliary_loss, losses  # type: ignore


CANCELLATION = tuple(range(44, 52))


class _FixedModel(nn.Module):
    def __init__(self, aux: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_aux", aux)

    def forward(self, tokens, token_mask, actions, action_mask):
        batch = tokens.shape[0]
        action_count = actions.shape[1]
        logits = self.fixed_aux.new_zeros((batch, action_count))
        wdl = self.fixed_aux.new_zeros((batch, 3))
        return logits, wdl, self.fixed_aux.expand(batch, -1)


def _base_batch(aux_target: torch.Tensor, aux_valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.zeros((1, 1, 24)),
        "token_mask": torch.ones((1, 1)),
        "actions": torch.zeros((1, 2, 24)),
        "action_mask": torch.ones((1, 2)),
        "policy_target": torch.tensor([[0.5, 0.5]]),
        "value_target": torch.zeros((1,)),
        "aux_target": aux_target,
        "aux_valid_mask": aux_valid_mask,
    }


class CancellationAuxiliaryLossTests(unittest.TestCase):
    def test_only_cancellation_channels_contribute(self) -> None:
        prediction = torch.zeros((1, 52))
        target = torch.zeros((1, 52))
        valid = torch.ones((1, 52))
        target[0, list(CANCELLATION)] = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        )
        target[0, 4] = 100.0  # unrelated attack channel must not affect this term

        loss, valid_count = _cancellation_auxiliary_loss(prediction, target, valid)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0] * 2).square().mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(valid_count.item(), 8.0)

    def test_valid_mask_is_respected(self) -> None:
        prediction = torch.zeros((1, 52))
        target = torch.zeros((1, 52))
        valid = torch.zeros((1, 52))
        target[0, 44] = 2.0
        target[0, 45] = 100.0
        valid[0, 44] = 1.0

        loss, valid_count = _cancellation_auxiliary_loss(prediction, target, valid)

        self.assertAlmostEqual(loss.item(), 4.0, places=6)
        self.assertEqual(valid_count.item(), 1.0)

    def test_cancellation_weight_contributes_exactly_to_total(self) -> None:
        prediction = torch.zeros((1, 52))
        target = torch.zeros((1, 52))
        valid = torch.ones((1, 52))
        target[0, 44] = 2.0
        model = _FixedModel(prediction)
        weights = {
            "policy": 0.0,
            "value": 0.0,
            "aux": 0.0,
            "vs_aux": 0.0,
            "cancellation_aux": 0.25,
        }

        total, parts = losses(model, _base_batch(target, valid), weights=weights)

        self.assertGreater(parts["cancellation_aux"], 0.0)
        self.assertAlmostEqual(
            total.item(), 0.25 * parts["cancellation_aux"], places=6
        )

    def test_schema_v3_rejects_nonzero_cancellation_weight(self) -> None:
        prediction = torch.zeros((1, 44))
        target = torch.zeros((1, 44))
        valid = torch.ones((1, 44))
        model = _FixedModel(prediction)
        weights = {
            "policy": 0.0,
            "value": 0.0,
            "aux": 0.0,
            "vs_aux": 0.0,
            "cancellation_aux": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "schema-v4"):
            losses(model, _base_batch(target, valid), weights=weights)


if __name__ == "__main__":
    unittest.main()
