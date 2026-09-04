# SPDX-License-Identifier: MIT
"""Unit tests for schema-v3 VS-style auxiliary supervision."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

try:
    from .tetraformer import _vs_auxiliary_loss, losses
except ImportError:  # pragma: no cover - supports direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tetraformer import _vs_auxiliary_loss, losses  # type: ignore


ATTACK = (4, 8, 12, 16)
GARBAGE_CLEARED = (36, 37, 38, 39)


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


class VsAuxiliaryLossTests(unittest.TestCase):
    def test_interval_counts_are_accumulated_into_vs_over_100(self) -> None:
        prediction = torch.zeros((1, 44))
        target = torch.zeros((1, 44))
        valid = torch.ones((1, 44))
        target[0, list(ATTACK)] = torch.tensor([1.0, 2.0, 4.0, 8.0])

        loss, valid_count = _vs_auxiliary_loss(prediction, target, valid)

        expected_rates = torch.tensor([1.0, 1.5, 1.75, 1.875])
        expected = expected_rates.square().mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(valid_count.item(), 4.0)

    def test_cumulative_validity_stops_after_first_unknown_interval(self) -> None:
        prediction = torch.zeros((1, 44))
        target = torch.zeros((1, 44))
        valid = torch.ones((1, 44))
        target[0, list(ATTACK)] = 1.0
        valid[0, ATTACK[1]] = 0.0

        loss, valid_count = _vs_auxiliary_loss(prediction, target, valid)

        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        self.assertEqual(valid_count.item(), 1.0)

    def test_no_valid_horizon_reports_zero_valid_count(self) -> None:
        prediction = torch.zeros((1, 44))
        target = torch.zeros((1, 44))
        valid = torch.zeros((1, 44))

        loss, valid_count = _vs_auxiliary_loss(prediction, target, valid)

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(valid_count.item(), 0.0)

    def test_vs_weight_contributes_to_total_loss(self) -> None:
        prediction = torch.zeros((1, 44))
        target = torch.zeros((1, 44))
        valid = torch.ones((1, 44))
        target[0, ATTACK[0]] = 2.0
        model = _FixedModel(prediction)
        weights = {
            "policy": 0.0,
            "value": 0.0,
            "aux": 0.0,
            "vs_aux": 0.5,
        }

        total, parts = losses(model, _base_batch(target, valid), weights=weights)

        self.assertGreater(parts["vs_aux"], 0.0)
        self.assertAlmostEqual(total.item(), 0.5 * parts["vs_aux"], places=6)

    def test_schema_v2_rejects_nonzero_vs_weight(self) -> None:
        prediction = torch.zeros((1, 36))
        target = torch.zeros((1, 36))
        valid = torch.ones((1, 36))
        model = _FixedModel(prediction)
        weights = {
            "policy": 0.0,
            "value": 0.0,
            "aux": 0.0,
            "vs_aux": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "schema-v3"):
            losses(model, _base_batch(target, valid), weights=weights)


if __name__ == "__main__":
    unittest.main()
