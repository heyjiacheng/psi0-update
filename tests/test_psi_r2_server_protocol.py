from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import Future
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi_r2.deploy.psi_r2_serve_simple import (
    PSI0_DENOISE_STEPS,
    AsyncSlowChannel,
    PsiR2ServerConfig,
    Server,
)
from psi_r2.models.psi_r2 import PsiR2Model, SlowFeatures


class _IdentityModelTransform:
    @staticmethod
    def resize():
        return lambda image: image

    @staticmethod
    def center_crop():
        return lambda image: image


class _IdentityActionStateTransform:
    normalize_state = False

    @staticmethod
    def denormalize(actions: np.ndarray) -> np.ndarray:
        return actions


def make_server() -> Server:
    model = PsiR2Model.__new__(PsiR2Model)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.action_horizon = 30
    model.action_dim = 4
    model._init_psi_r2_runtime()
    model.encode_calls = 0
    model.bootstrap_horizons = []
    model.bootstrap_denoise_steps = []
    model.stream_horizons = []
    model.fail_next_stream = False

    def encode_slow(
        self: PsiR2Model,
        observations,
        instructions,
        *,
        captured_at=None,
    ) -> SlowFeatures:
        self.encode_calls += 1
        return SlowFeatures(
            views=torch.zeros(1, 1, 2, 3),
            attention_mask=torch.ones(1, 2),
            captured_at=float(captured_at),
        )

    def bootstrap(self: PsiR2Model, states, features, **kwargs) -> torch.Tensor:
        self.bootstrap_horizons.append(kwargs["output_horizon"])
        self.bootstrap_denoise_steps.append(kwargs["bootstrap_inference_steps"])
        return torch.zeros(1, kwargs["output_horizon"], self.action_dim)

    def stream(self: PsiR2Model, states, features, **kwargs) -> torch.Tensor:
        self.stream_horizons.append(kwargs["output_horizon"])
        if self.fail_next_stream:
            self.fail_next_stream = False
            raise RuntimeError("injected fast failure")
        return torch.ones(1, kwargs["output_horizon"], self.action_dim)

    object.__setattr__(model, "encode_slow", MethodType(encode_slow, model))
    object.__setattr__(
        model, "bootstrap_streaming_with_features", MethodType(bootstrap, model)
    )
    object.__setattr__(
        model, "predict_streaming_with_features", MethodType(stream, model)
    )

    server = Server.__new__(Server)
    server.cfg = SimpleNamespace(
        rtc=True,
        async_slow=True,
        bootstrap_inference_steps=10,
        fast_substeps=1,
        noise_s=0.999,
    )
    server.device = torch.device("cpu")
    server.model = model
    server.model_transform = _IdentityModelTransform()
    server.maxmin = _IdentityActionStateTransform()
    server.Da = 4
    server.Tp = 30
    server.Ta = 24
    server.slide_steps = 6
    server._request_lock = threading.Lock()
    server._metrics_lock = threading.Lock()
    server._episode_active = False
    server._stream_started = False
    server._request_count = 0
    server._episode_count = 0
    server._last_metrics = {}
    server._last_serve_time = time.monotonic()
    server.slow_channel = AsyncSlowChannel(model)
    return server


def payload(*, reset: bool) -> dict:
    history = {"reset": True} if reset else {}
    request = RequestMessage(
        image={"camera": np.zeros((8, 8, 3), dtype=np.uint8)},
        instruction="move the object",
        history=history,
        state={"states": np.zeros((1, 4), dtype=np.float32)},
        condition={},
        gt_action=np.zeros((1, 4), dtype=np.float32),
        dataset_name="test",
        timestamp="0",
    )
    return request.serialize()


class PsiR2ServerProtocolTest(unittest.TestCase):
    def test_preserves_original_psi0_ten_step_full_flow_default(self) -> None:
        self.assertEqual(PSI0_DENOISE_STEPS, 10)
        self.assertEqual(
            PsiR2ServerConfig.model_fields["bootstrap_inference_steps"].default,
            10,
        )
        self.assertEqual(
            PsiR2ServerConfig.model_fields["fast_substeps"].default,
            10,
        )

    def test_act_keeps_psi_response_shape_across_bootstrap_and_fast_calls(self) -> None:
        server = make_server()
        try:
            first = server.predict_action(payload(reset=True))
            self.assertEqual(first.status_code, 200)
            first_message = ResponseMessage.deserialize(json.loads(first.body))
            self.assertEqual(first_message.action.shape, (24, 4))
            self.assertTrue(np.all(first_message.action == 0.0))

            second = server.predict_action(payload(reset=False))
            self.assertEqual(second.status_code, 200)
            second_message = ResponseMessage.deserialize(json.loads(second.body))
            self.assertEqual(second_message.action.shape, (24, 4))
            self.assertTrue(np.all(second_message.action == 1.0))
            self.assertEqual(server._episode_count, 1)
            self.assertEqual(server._request_count, 2)
            self.assertEqual(server.model.bootstrap_horizons, [24])
            self.assertEqual(server.model.bootstrap_denoise_steps, [10])
            self.assertEqual(server.model.stream_horizons, [24])
        finally:
            server.close()

    def test_slow_then_fast_bootstraps_from_cache_at_true_r2_cadence(self) -> None:
        server = make_server()
        try:
            slow = server.update_slow(payload(reset=True))
            self.assertEqual(slow.status_code, 200)
            self.assertEqual(server.model.encode_calls, 1)
            self.assertTrue(server._episode_active)
            self.assertFalse(server._stream_started)

            fast = server.predict_fast(payload(reset=False))
            self.assertEqual(fast.status_code, 200)
            fast_message = ResponseMessage.deserialize(json.loads(fast.body))
            self.assertEqual(fast_message.action.shape, (6, 4))
            self.assertTrue(np.all(fast_message.action == 0.0))
            self.assertEqual(server.model.encode_calls, 1)
            self.assertEqual(server.model.bootstrap_horizons, [6])
            self.assertTrue(server._stream_started)
            self.assertEqual(server._episode_count, 1)
        finally:
            server.close()

    def test_async_channel_does_not_deadlock_for_already_done_future(self) -> None:
        class ImmediateExecutor:
            def submit(self, function, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except Exception as exc:  # noqa: BLE001 - emulate Future contract.
                    future.set_exception(exc)
                return future

            def shutdown(self, **kwargs):
                return None

        server = make_server()
        channel = AsyncSlowChannel(server.model)
        channel._executor.shutdown(wait=True)
        channel._executor = ImmediateExecutor()
        result: list[str] = []

        def submit() -> None:
            result.append(
                channel.submit(
                    [[np.zeros((1, 1, 3), dtype=np.uint8)]],
                    ["instruction"],
                    captured_at=1.0,
                    episode_id=server.model.episode_id,
                )
            )

        thread = threading.Thread(target=submit, daemon=True)
        thread.start()
        thread.join(timeout=1.0)
        try:
            self.assertFalse(thread.is_alive(), "submit deadlocked in done callback")
            self.assertEqual(result, ["started"])
            self.assertEqual(
                channel.status(),
                {"running": False, "pending": False, "closed": False},
            )
        finally:
            if not thread.is_alive():
                channel.close()
            server.close()

    def test_act_submits_slow_refresh_only_after_fast_response_copy(self) -> None:
        server = make_server()
        events: list[str] = []
        try:
            first = server.predict_action(payload(reset=True))
            self.assertEqual(first.status_code, 200)

            original_stream = server.model.predict_streaming_with_features
            original_response = server._response

            def stream(self, *args, **kwargs):
                events.append("fast")
                return original_stream(*args, **kwargs)

            def response(self, raw, *, expected_horizon):
                events.append("response-copy")
                return original_response(raw, expected_horizon=expected_horizon)

            def submit(self, *args, **kwargs):
                events.append("slow-submit")
                return "started"

            object.__setattr__(
                server.model,
                "predict_streaming_with_features",
                MethodType(stream, server.model),
            )
            server._response = MethodType(response, server)
            server.slow_channel.submit = MethodType(submit, server.slow_channel)

            second = server.predict_action(payload(reset=False))
            self.assertEqual(second.status_code, 200)
            self.assertEqual(events, ["fast", "response-copy", "slow-submit"])
            self.assertIn("fast_enqueue_ms", server._last_metrics)
            self.assertIn("server_total_ms", server._last_metrics)
            self.assertNotIn("fast_ms", server._last_metrics)
        finally:
            server.close()

    def test_failed_fast_request_invalidates_stream_and_next_call_recovers(
        self,
    ) -> None:
        server = make_server()
        try:
            first = server.predict_action(payload(reset=True))
            self.assertEqual(first.status_code, 200)
            failed_episode_id = server.model.episode_id
            server.model.fail_next_stream = True

            failed = server.predict_action(payload(reset=False))
            self.assertEqual(failed.status_code, 500)
            self.assertFalse(server._episode_active)
            self.assertFalse(server._stream_started)
            self.assertIsNone(server.model.get_slow_features())
            self.assertGreater(server.model.episode_id, failed_episode_id)

            recovered = server.predict_action(payload(reset=False))
            self.assertEqual(recovered.status_code, 200)
            recovered_message = ResponseMessage.deserialize(json.loads(recovered.body))
            self.assertTrue(np.all(recovered_message.action == 0.0))
            self.assertEqual(server._episode_count, 2)
            self.assertTrue(server._episode_active)
            self.assertTrue(server._stream_started)
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
