from __future__ import annotations

import unittest

import torch

from psi_r2.models.schedule import PiR2RollingBuffer, PiR2Schedule


class PiR2ScheduleTest(unittest.TestCase):
    def test_requested_psi_horizon_uses_six_step_overlap(self) -> None:
        prediction_horizon = 30
        execution_horizon = 24
        schedule = PiR2Schedule(
            horizon=prediction_horizon,
            slide_steps=prediction_horizon - execution_horizon,
        )

        sigma = schedule.initial_sigma()
        self.assertEqual(schedule.slide_steps, 6)
        self.assertEqual(schedule.ramp_width, 18)
        self.assertTrue(torch.equal(sigma[:6], torch.zeros(6)))
        self.assertTrue(torch.equal(sigma[-6:], torch.ones(6)))
        self.assertTrue(torch.all(sigma[1:] >= sigma[:-1]))

        target = schedule.target_sigma()
        self.assertTrue(torch.equal(target[:6], torch.zeros(6)))
        self.assertTrue(torch.allclose(target[6:], sigma[:-6]))

    def test_tau_to_psi_timestep_orientation(self) -> None:
        schedule = PiR2Schedule(horizon=9, slide_steps=2, train_timesteps=1000)
        tau = schedule.initial_tau()
        sigma = schedule.initial_sigma()

        self.assertTrue(torch.allclose(sigma, 1.0 - tau / schedule.noise_s))
        self.assertTrue(
            torch.equal(schedule.model_timesteps(sigma[:2]), torch.zeros(2))
        )
        self.assertTrue(
            torch.equal(schedule.model_timesteps(sigma[-2:]), torch.full((2,), 1000.0))
        )

    def test_rolling_update_closes_cycle_and_emits_new_clean_window(self) -> None:
        schedule = PiR2Schedule(horizon=10, slide_steps=2)
        rolling = PiR2RollingBuffer(schedule)
        clean = torch.zeros(1, 10, 3)
        initial_noise = torch.ones_like(clean)
        rolling.seed(clean, noise=initial_noise)

        def unit_velocity(actions: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            self.assertEqual(actions.shape[:2], sigma.shape)
            return torch.ones_like(actions)

        for _ in range(3):
            snapshot = rolling.denoise_and_slide(
                unit_velocity,
                substeps=2,
                tail_noise=torch.ones(1, 2, 3),
            )
            emitted = rolling.emitted(snapshot)
            self.assertTrue(torch.allclose(emitted, torch.zeros_like(emitted)))
            expected_sigma = schedule.initial_sigma().unsqueeze(0)
            self.assertTrue(torch.allclose(rolling.sigma, expected_sigma))
            self.assertTrue(
                torch.allclose(
                    rolling.actions,
                    expected_sigma.unsqueeze(-1).expand_as(rolling.actions),
                )
            )

    def test_requires_bootstrap_seed(self) -> None:
        rolling = PiR2RollingBuffer(PiR2Schedule(horizon=7, slide_steps=2))
        with self.assertRaisesRegex(RuntimeError, "seeded"):
            rolling.denoise_and_slide(lambda actions, sigma: actions)

    def test_rejects_invalid_ramp(self) -> None:
        for horizon, slide in [(30, 0), (30, 15), (30, 24)]:
            with (
                self.subTest(horizon=horizon, slide=slide),
                self.assertRaises(ValueError),
            ):
                PiR2Schedule(horizon=horizon, slide_steps=slide)


if __name__ == "__main__":
    unittest.main()
