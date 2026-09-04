#!/usr/bin/env python3
from __future__ import annotations

import unittest

from trainer.ttrm_exact_replay import ReplayMachine


class ExactReplayRegressionTest(unittest.TestCase):
    def test_v19_gravity_ramp_matches_long_league_snapshot(self) -> None:
        machine = ReplayMachine.__new__(ReplayMachine)
        machine.gravity_base = 0.02
        machine.gravity_increase = 0.0035
        machine.gravity_margin = 7200.0
        machine.gravity = machine.gravity_base

        machine._update_gravity_for_frame(7200)
        self.assertAlmostEqual(machine.gravity, 0.02, places=12)

        machine._update_gravity_for_frame(9524)
        self.assertAlmostEqual(machine.gravity, 0.1555666666666568, places=12)

        machine._update_gravity_for_frame(9547)
        self.assertAlmostEqual(machine.gravity, 0.15690833333332332, places=12)

    def test_gravity_is_constant_before_margin(self) -> None:
        machine = ReplayMachine.__new__(ReplayMachine)
        machine.gravity_base = 0.02
        machine.gravity_increase = 0.0035
        machine.gravity_margin = 7200.0
        machine.gravity = machine.gravity_base

        for frame in (0, 1, 7199, 7200):
            machine._update_gravity_for_frame(frame)
            self.assertAlmostEqual(machine.gravity, 0.02, places=12)


if __name__ == "__main__":
    unittest.main()
