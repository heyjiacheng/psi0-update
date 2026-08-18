from __future__ import annotations

import unittest
from types import MethodType, SimpleNamespace

import torch

from psi.models.psi0 import ActionTransformerModel
from psi_r2.models.psi_r2 import PsiR2Model, SlowFeatures


def make_runtime_only_model() -> PsiR2Model:
    model = PsiR2Model.__new__(PsiR2Model)
    torch.nn.Module.__init__(model)
    model.action_horizon = 30
    model.action_dim = 4
    model.device = "cpu"
    model.noise_scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1000)
    )
    model._init_psi_r2_runtime()
    return model


class PsiR2RuntimeTest(unittest.TestCase):
    @staticmethod
    def _small_action_head() -> ActionTransformerModel:
        return ActionTransformerModel(
            attention_head_dim=4,
            num_attention_heads=2,
            action_num_blocks=2,
            action_pred_horizon=5,
            action_dim=2,
            action_hidden_dim=8,
            action_nheads=2,
            n_conditions=0,
            odim=3,
            view_feature_dim=6,
        ).eval()

    def test_per_position_head_maps_pi_r2_zero_tau_to_psi_noise_time(self) -> None:
        model = make_runtime_only_model()
        model.action_horizon = 5
        model.action_dim = 2
        head = self._small_action_head()
        model.action_header = head
        features = SlowFeatures(
            views=torch.randn(1, 1, 4, 6),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            captured_at=0.0,
        )
        captured_obs_temb = []
        handles = []

        def capture_temb(module, args, kwargs):
            del module
            embedding = kwargs.get("emb", args[1] if len(args) > 1 else None)
            captured_obs_temb.append(embedding.detach().clone())

        for block in head.transformer_blocks:
            handles.append(
                block.norm1_obs.register_forward_pre_hook(
                    capture_temb,
                    with_kwargs=True,
                )
            )
        try:
            output = model._predict_velocity(
                torch.randn(1, 5, 2),
                torch.tensor([[0.0, 0.125, 0.5, 0.875, 1.0]]),
                torch.randn(1, 1, 3),
                features,
            )
        finally:
            for handle in handles:
                handle.remove()

        expected = head.time_ins_embed(torch.full((1,), 1000.0))
        wrong_numeric_zero = head.time_ins_embed(torch.zeros(1))
        self.assertEqual(output.shape, (1, 5, 2))
        self.assertEqual(len(captured_obs_temb), 2)
        for observed in captured_obs_temb:
            self.assertTrue(torch.allclose(observed, expected))
            self.assertFalse(torch.allclose(observed, wrong_numeric_zero))

    def test_per_position_adapter_matches_legacy_head_at_uniform_noise_time(self) -> None:
        model = make_runtime_only_model()
        model.action_horizon = 5
        model.action_dim = 2
        head = self._small_action_head()
        model.action_header = head
        actions = torch.randn(1, 5, 2)
        states = torch.randn(1, 1, 3)
        features = SlowFeatures(
            views=torch.randn(1, 1, 4, 6),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            captured_at=0.0,
        )
        with torch.inference_mode():
            expected = head(
                hidden_states=None,
                timestep=torch.full((1, 5), 1000.0),
                joint_attention_kwargs={
                    "action_hidden_embeds": actions,
                    "views": features.views,
                    "obs": states,
                    "traj2ds": None,
                },
                vlm_attn_mask=features.attention_mask,
            ).action
            actual = model._predict_velocity(
                actions,
                torch.ones(1, 5),
                states,
                features,
            )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_legacy_head_itself_accepts_per_position_timesteps(self) -> None:
        head = self._small_action_head()
        output = head(
            hidden_states=None,
            timestep=torch.tensor([[0.0, 125.0, 500.0, 875.0, 1000.0]]),
            joint_attention_kwargs={
                "action_hidden_embeds": torch.randn(1, 5, 2),
                "views": torch.randn(1, 1, 4, 6),
                "obs": torch.randn(1, 1, 3),
                "traj2ds": None,
            },
            vlm_attn_mask=torch.ones(1, 4, dtype=torch.long),
        ).action
        self.assertEqual(output.shape, (1, 5, 2))

    def test_runtime_state_adds_no_checkpoint_keys(self) -> None:
        model = PsiR2Model.__new__(PsiR2Model)
        torch.nn.Module.__init__(model)
        model.register_parameter("sentinel", torch.nn.Parameter(torch.ones(1)))
        keys_before = set(model.state_dict())

        model._init_psi_r2_runtime()

        self.assertEqual(set(model.state_dict()), keys_before)

    def test_slow_cache_rejects_old_episode_and_older_image(self) -> None:
        model = make_runtime_only_model()
        episode = model.episode_id
        newer = SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=20.0,
        )
        older = SlowFeatures(
            views=torch.ones(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=10.0,
        )

        self.assertTrue(model.install_slow_features(newer, episode))
        self.assertFalse(model.install_slow_features(older, episode))
        self.assertEqual(model.get_slow_features().captured_at, 20.0)

        model.reset_runtime()
        self.assertFalse(model.install_slow_features(newer, episode))
        self.assertIsNone(model.get_slow_features())

    def test_legacy_24_action_response_batches_four_six_step_updates(self) -> None:
        model = make_runtime_only_model()
        clean = torch.zeros(1, 30, 4)
        model.seed_streaming_buffer(clean, slide_steps=6)
        features = SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=0.0,
        )
        states = torch.zeros(1, 1, 4)
        calls = []
        test_case = self

        def fake_velocity(
            self: PsiR2Model,
            actions: torch.Tensor,
            sigma: torch.Tensor,
            states_arg: torch.Tensor,
            features_arg: SlowFeatures,
        ) -> torch.Tensor:
            calls.append(sigma.clone())
            test_case.assertEqual(states_arg.shape, states.shape)
            test_case.assertIs(features_arg, features)
            return torch.zeros_like(actions)

        # Bypass the learned head while exercising the real batching/buffer path.
        object.__setattr__(model, "_predict_velocity", MethodType(fake_velocity, model))
        output = model.predict_streaming_with_features(
            states,
            features,
            output_horizon=24,
            slide_steps=6,
            substeps=1,
        )

        self.assertEqual(output.shape, (1, 24, 4))
        self.assertEqual(len(calls), 4)
        expected_sigma = model._rolling.schedule.initial_sigma().unsqueeze(0)
        self.assertTrue(torch.allclose(model._rolling.sigma, expected_sigma))

    def test_response_horizon_must_align_to_slide(self) -> None:
        model = make_runtime_only_model()
        model.seed_streaming_buffer(torch.zeros(1, 30, 4), slide_steps=6)
        features = SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=0.0,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            model.predict_streaming_with_features(
                torch.zeros(1, 1, 4),
                features,
                output_horizon=23,
                slide_steps=6,
            )

    def test_failed_multi_cycle_update_restores_entire_rolling_state(self) -> None:
        model = make_runtime_only_model()
        model.seed_streaming_buffer(torch.zeros(1, 30, 4), slide_steps=6)
        features = SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=0.0,
        )
        actions_before = model._rolling.actions.clone()
        sigma_before = model._rolling.sigma.clone()
        calls = 0

        def failing_velocity(
            self: PsiR2Model,
            actions: torch.Tensor,
            sigma: torch.Tensor,
            states_arg: torch.Tensor,
            features_arg: SlowFeatures,
        ) -> torch.Tensor:
            del self, sigma, states_arg, features_arg
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic action-head failure")
            return torch.zeros_like(actions)

        object.__setattr__(
            model,
            "_predict_velocity",
            MethodType(failing_velocity, model),
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            model.predict_streaming_with_features(
                torch.zeros(1, 1, 4),
                features,
                output_horizon=24,
                slide_steps=6,
            )

        self.assertEqual(calls, 2)
        self.assertTrue(torch.equal(model._rolling.actions, actions_before))
        self.assertTrue(torch.equal(model._rolling.sigma, sigma_before))

    def test_nonfinite_output_restores_entire_rolling_state(self) -> None:
        model = make_runtime_only_model()
        model.seed_streaming_buffer(torch.zeros(1, 30, 4), slide_steps=6)
        features = SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=0.0,
        )
        actions_before = model._rolling.actions.clone()
        sigma_before = model._rolling.sigma.clone()

        def nonfinite_velocity(
            self: PsiR2Model,
            actions: torch.Tensor,
            sigma: torch.Tensor,
            states_arg: torch.Tensor,
            features_arg: SlowFeatures,
        ) -> torch.Tensor:
            del self, sigma, states_arg, features_arg
            return torch.full_like(actions, torch.nan)

        object.__setattr__(
            model,
            "_predict_velocity",
            MethodType(nonfinite_velocity, model),
        )
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            model.predict_streaming_with_features(
                torch.zeros(1, 1, 4),
                features,
                output_horizon=6,
                slide_steps=6,
            )

        self.assertTrue(torch.equal(model._rolling.actions, actions_before))
        self.assertTrue(torch.equal(model._rolling.sigma, sigma_before))


if __name__ == "__main__":
    unittest.main()
