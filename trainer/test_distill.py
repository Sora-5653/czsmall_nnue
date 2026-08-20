#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from trainer.distill import (
    distillation_losses,
    masked_temperature_kl,
    temperature_kl,
    wdl_expected_value,
)


class DistillationLossTest(unittest.TestCase):
    def test_identical_logits_have_zero_kl(self) -> None:
        logits = torch.tensor([[1.0, -0.5, 2.0], [0.1, 0.2, 0.3]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.assertAlmostEqual(
            float(masked_temperature_kl(logits, logits, mask, 3.0)), 0.0, places=6
        )
        self.assertAlmostEqual(float(temperature_kl(logits, logits, 3.0)), 0.0, places=6)

    def test_illegal_action_logits_do_not_affect_policy_kl(self) -> None:
        student = torch.tensor([[1.0, 2.0, 1000.0]])
        teacher = torch.tensor([[1.5, 1.0, -1000.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        base = masked_temperature_kl(student, teacher, mask, 2.0)
        student[:, 2] = -9999.0
        teacher[:, 2] = 9999.0
        changed = masked_temperature_kl(student, teacher, mask, 2.0)
        self.assertAlmostEqual(float(base), float(changed), places=6)

    def test_expected_value_maps_wdl_to_win_minus_loss(self) -> None:
        logits = torch.tensor([[20.0, 0.0, -20.0], [-20.0, 0.0, 20.0]])
        value = wdl_expected_value(logits)
        self.assertGreater(float(value[0]), 0.999)
        self.assertLess(float(value[1]), -0.999)

    def test_distillation_total_is_weighted_sum(self) -> None:
        action_mask = torch.tensor([[1.0, 1.0]])
        teacher = (
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([[1.0, 0.0, -1.0]]),
            torch.zeros((1, 1)),
        )
        student = (
            torch.tensor([[0.0, 0.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True),
            torch.zeros((1, 1)),
        )
        total, parts = distillation_losses(
            student,
            teacher,
            action_mask,
            temperature=3.0,
            policy_weight=1.0,
            value_kl_weight=0.5,
            value_mse_weight=2.0,
        )
        expected = parts["policy_kl"] + 0.5 * parts["value_kl"] + 2.0 * parts["value_mse"]
        self.assertTrue(torch.allclose(total, expected))
        total.backward()
        self.assertIsNotNone(student[0].grad)
        self.assertIsNotNone(student[1].grad)


if __name__ == "__main__":
    unittest.main()
