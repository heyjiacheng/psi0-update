"""Serve Psi-R2 through Psi0's existing SIMPLE HTTP protocol."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms import v2

from psi.config.transform import (
    ActionStateTransform,
    Psi0ModelTransform,
    SimpleRepackTransform,
)
from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi.utils import (
    pad_to_len,
    parse_args_to_tyro_config,
    seed_everything,
    timing,
)
from psi.utils.overwatch import initialize_overwatch
from psi_r2.models.psi_r2 import PsiR2Model, SlowFeatures
from psi_r2.models.schedule import PiR2Schedule

overwatch = initialize_overwatch(__name__)

# Preserve the original SIMPLE Psi0 integration budget on both the scalar
# bootstrap and each PI-R2 rolling update. The rolling time grid remains the
# PI-R2 per-position schedule; only its Euler subdivision count differs from
# the released PI-R2 default.
PSI0_DENOISE_STEPS = 10


class PsiR2ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"
    policy: str = "psi-r2"
    action_exec_horizon: int | None = None
    rtc: bool = False
    run_dir: str
    ckpt_step: int

    pir2_slide_steps: int | None = Field(
        default=None,
        description=(
            "PI-R2 rolling width d. Defaults to action_chunk_size - "
            "action_exec_horizon, matching the existing Psi RTC overlap."
        ),
    )
    bootstrap_inference_steps: int = Field(
        default=PSI0_DENOISE_STEPS,
        ge=1,
        description=(
            "Scalar-flow NFEs used for plain inference and rolling-buffer seeding. "
            "The default preserves Psi0's original 10-step denoise path."
        ),
    )
    fast_substeps: int = Field(
        default=PSI0_DENOISE_STEPS,
        ge=1,
        description=(
            "Velocity evaluations per PI-R2 fast update. The default preserves "
            "Psi0's original 10-step denoise budget while retaining PI-R2's "
            "per-position schedule."
        ),
    )
    noise_s: float = Field(default=0.999, gt=0.0, le=1.0)
    async_slow: bool = Field(
        default=True,
        description="Refresh Qwen image/language features after returning fast actions.",
    )


@dataclass(frozen=True)
class _SlowJob:
    observations: list[list[Image.Image]]
    instructions: list[str]
    captured_at: float
    episode_id: int


class AsyncSlowChannel:
    """Single-worker, latest-wins slow-channel refresh queue."""

    def __init__(self, model: PsiR2Model) -> None:
        self.model = model
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="psi-r2-slow"
        )
        self._lock = threading.Lock()
        self._active: Future | None = None
        self._pending: _SlowJob | None = None
        self._closed = False

    def _run(self, job: _SlowJob) -> bool:
        try:
            features = self.model.encode_slow(
                job.observations,
                job.instructions,
                captured_at=job.captured_at,
            )
            return self.model.install_slow_features(features, job.episode_id)
        except Exception as exc:
            self.model.record_slow_error(exc, job.episode_id)
            raise

    def _launch_locked(self, job: _SlowJob) -> Future:
        """Submit while holding ``_lock``; register callbacks after releasing it.

        ``Future.add_done_callback`` invokes the callback inline when a future
        has already completed.  Keeping callback registration outside the
        non-reentrant queue lock avoids deadlocking with very fast jobs (and
        with synchronous executors used by tests).
        """
        future = self._executor.submit(self._run, job)
        self._active = future
        return future

    def _finished(self, future: Future) -> None:
        try:
            future.result()
        except Exception:  # noqa: BLE001 - callback logs worker failures.
            overwatch.warning("Slow-channel refresh failed:\n" + traceback.format_exc())
        next_future: Future | None = None
        with self._lock:
            if self._active is future:
                self._active = None
                pending = self._pending
                self._pending = None
                if pending is not None and not self._closed:
                    next_future = self._launch_locked(pending)
        if next_future is not None:
            next_future.add_done_callback(self._finished)

    def submit(
        self,
        observations: list[list[Image.Image]],
        instructions: list[str],
        *,
        captured_at: float,
        episode_id: int,
    ) -> str:
        job = _SlowJob(observations, instructions, captured_at, episode_id)
        future: Future | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("slow channel is closed")
            # A done future remains active until its callback has drained the
            # latest-wins slot.  Replacing it early could start two jobs.
            if self._active is not None:
                self._pending = job
                return "queued-latest"
            future = self._launch_locked(job)
        # This may invoke _finished immediately, so it must stay outside _lock.
        future.add_done_callback(self._finished)
        return "started"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._active is not None and not self._active.done(),
                "pending": self._pending is not None,
                "closed": self._closed,
            }

    def discard_pending(self) -> None:
        """Drop a queued refresh when a new episode supersedes it."""
        with self._lock:
            self._pending = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class _PreparedRequest:
    request: RequestMessage
    observations: list[list[Image.Image]]
    instructions: list[str]
    states: torch.Tensor
    received_at: float


class Server:
    """Psi-R2 server for one active client episode at a time.

    The legacy SIMPLE request has no session identifier, so reset and rolling
    state are intentionally process-global and all mutating endpoints share one
    request lock.
    """

    def __init__(self, cfg: PsiR2ServerConfig) -> None:
        device_type = cfg.device.split(":", 1)[0]
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please check your CUDA installation."
            )

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.run_dir = Path(cfg.run_dir)
        self.ckpt_step = cfg.ckpt_step
        self._request_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._episode_active = False
        self._stream_started = False
        self._request_count = 0
        self._episode_count = 0
        self._last_metrics: dict[str, Any] = {}
        self._last_serve_time = time.monotonic()

        checkpoint_dir = self.run_dir / "checkpoints" / f"ckpt_{self.ckpt_step}"
        if not self.run_dir.exists():
            raise FileNotFoundError(f"run_dir {self.run_dir} does not exist")
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint {checkpoint_dir} does not exist")
        if not (self.run_dir / "run_config.json").exists():
            raise FileNotFoundError(
                f"{self.run_dir / 'run_config.json'} does not exist"
            )
        if not (self.run_dir / "argv.txt").exists():
            raise FileNotFoundError(f"{self.run_dir / 'argv.txt'} does not exist")

        # Reuse the original run schema so existing Psi0 artifacts remain valid.
        config_from_argv = parse_args_to_tyro_config(self.run_dir / "argv.txt")
        launch_config = config_from_argv.model_validate_json(
            (self.run_dir / "run_config.json").read_text()
        )
        seed_everything(launch_config.seed or 42)
        self.launch_config = launch_config

        overwatch.info(f"Loading Psi-R2 on {self.device}")
        self.model = PsiR2Model.from_pretrained(
            self.run_dir,
            self.ckpt_step,
            launch_config,
            device=cfg.device,
        )
        self.model.to(self.device)
        self.model.eval()

        self.maxmin: ActionStateTransform = launch_config.data.transform.field
        self.repack_transform: SimpleRepackTransform = (
            launch_config.data.transform.repack
        )
        self.model_transform: Psi0ModelTransform = launch_config.data.transform.model

        self.Da = int(launch_config.model.action_dim)
        self.Tp = int(launch_config.model.action_chunk_size)
        self.Ta = int(
            cfg.action_exec_horizon or launch_config.model.action_exec_horizon
        )
        if not 1 <= self.Ta <= self.Tp:
            raise ValueError(
                f"action_exec_horizon must be in [1,{self.Tp}], got {self.Ta}"
            )
        if launch_config.model.noise_scheduler != "flow":
            raise ValueError("Psi-R2 requires a flow-matching Psi0 checkpoint")

        default_slide = self.Tp - self.Ta
        self.slide_steps = int(
            cfg.pir2_slide_steps if cfg.pir2_slide_steps is not None else default_slide
        )
        if cfg.rtc:
            if self.slide_steps <= 0:
                raise ValueError(
                    "Cannot infer PI-R2 slide width when action_exec_horizon equals "
                    "action_chunk_size; pass --pir2-slide-steps explicitly"
                )
            PiR2Schedule(
                horizon=self.Tp,
                slide_steps=self.slide_steps,
                train_timesteps=int(launch_config.model.train_diffusion_steps),
                noise_s=cfg.noise_s,
            )
            if self.Ta % self.slide_steps != 0:
                raise ValueError(
                    "The unchanged Psi /act protocol needs action_exec_horizon to be "
                    "divisible by pir2_slide_steps; "
                    f"got Ta={self.Ta}, d={self.slide_steps}"
                )

        self.slow_channel = AsyncSlowChannel(self.model)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        log = timing.open_log(
            self.run_dir
            / "inference_timing"
            / f"ckpt{self.ckpt_step}_Ta{self.Ta}"
            f"_{'rtc' if cfg.rtc else 'nortc'}_{stamp}.jsonl"
        )
        if log is not None:
            overwatch.info(f"Per-request inference timings -> {log}")

        num_params = sum(parameter.numel() for parameter in self.model.parameters())
        overwatch.info(f"Parameters (millions): {num_params * 1e-6:.3f}")
        if cfg.rtc:
            overwatch.info(
                "PI-R2 enabled: "
                f"Tp={self.Tp}, Ta={self.Ta}, d={self.slide_steps}, "
                f"fast_updates_per_response={self.Ta // self.slide_steps}, "
                f"substeps={cfg.fast_substeps}"
            )

    def _prepare_request(self, payload: dict[str, Any]) -> _PreparedRequest:
        received_at = time.time()
        with timing.timed("request_preprocess", self.device):
            request = RequestMessage.deserialize(payload)
            transforms = v2.Compose(
                [self.model_transform.resize(), self.model_transform.center_crop()]
            )
            observations = [
                [transforms(Image.fromarray(image)) for image in request.image.values()]
            ]

            states = torch.from_numpy(request.state["states"].copy())
            if self.maxmin.normalize_state:
                padded = pad_to_len(states.numpy(), self.maxmin.pad_state_dim, dim=1)[0]
                states = torch.from_numpy(self.maxmin.normalize_state_func(padded))
            states = states.to(self.device, dtype=torch.float32).unsqueeze(0)
        return _PreparedRequest(
            request=request,
            observations=observations,
            instructions=[request.instruction],
            states=states,
            received_at=received_at,
        )

    def _refresh_slow_sync(
        self, prepared: _PreparedRequest, episode_id: int
    ) -> SlowFeatures:
        features = self.model.encode_slow(
            prepared.observations,
            prepared.instructions,
            captured_at=prepared.received_at,
        )
        if not self.model.install_slow_features(features, episode_id):
            raise RuntimeError("episode changed during synchronous slow refresh")
        cached = self.model.get_slow_features()
        assert cached is not None
        return cached

    def _begin_episode_locked(self) -> int:
        """Start one process-global episode while ``_request_lock`` is held."""
        self.slow_channel.discard_pending()
        episode_id = self.model.reset_runtime()
        self._episode_active = True
        self._stream_started = False
        self._episode_count += 1
        timing.new_episode()
        return episode_id

    def _invalidate_runtime_locked(self) -> None:
        """Make a partially advanced stream unusable after a request failure."""
        self._episode_active = False
        self._stream_started = False
        try:
            self.slow_channel.discard_pending()
            self.model.reset_runtime()
        except Exception:  # noqa: BLE001 - preserve the original request failure.
            overwatch.warning(
                "Failed to invalidate Psi-R2 runtime after request error:\n"
                + traceback.format_exc()
            )

    def _raw_action(
        self,
        prepared: _PreparedRequest,
        *,
        allow_slow_refresh: bool,
        output_horizon: int,
    ) -> tuple[torch.Tensor, dict[str, Any], _SlowJob | None]:
        history = prepared.request.history
        episode_start = not self._episode_active or "reset" in history
        if episode_start:
            episode_id = self._begin_episode_locked()
        else:
            episode_id = self.model.episode_id

        stream_bootstrap = not self._stream_started
        features = self.model.get_slow_features()
        deferred_slow_job: _SlowJob | None = None
        if features is None:
            features = self._refresh_slow_sync(prepared, episode_id)
            slow_mode = "bootstrap-sync" if stream_bootstrap else "cache-miss-sync"
        elif stream_bootstrap:
            # An explicit /slow call may establish the episode and cache before
            # the first /fast call.  Reuse it when seeding the rolling buffer.
            slow_mode = "bootstrap-cached"
        elif not allow_slow_refresh:
            slow_mode = "cached-only"
        elif self.cfg.async_slow:
            # Submission is deferred until the fast tensor has been copied to
            # CPU.  CUDA work from this image can therefore never jump ahead of
            # the action calculation for the same /act request.
            deferred_slow_job = _SlowJob(
                prepared.observations,
                prepared.instructions,
                prepared.received_at,
                episode_id,
            )
            slow_mode = "deferred"
        else:
            features = self._refresh_slow_sync(prepared, episode_id)
            slow_mode = "refresh-sync"

        # Slow cost only lands on this request when it was refreshed inline;
        # the deferred/cached paths pay it on the worker thread instead.
        slow_on_critical_path = slow_mode.endswith("-sync")
        timing.record(
            "vlm_preprocess", features.preprocess_ms if slow_on_critical_path else 0.0
        )
        timing.record(
            "vlm_forward", features.forward_ms if slow_on_critical_path else 0.0
        )
        # Cost of the encode that produced the features actually used, wherever
        # it ran. Shows what the async channel is hiding from the client.
        timing.record("slow_vlm_offpath", 0.0 if slow_on_critical_path else features.forward_ms)

        cycles = output_horizon // self.slide_steps if self.cfg.rtc else 0
        fast_start = time.perf_counter()
        with timing.timed("action_expert", self.device):
            if not self.cfg.rtc:
                # Plain compatibility mode remains a complete, synchronous Psi0 flow.
                raw = self.model.predict_full_flow_with_features(
                    prepared.states,
                    features,
                    num_inference_steps=self.cfg.bootstrap_inference_steps,
                )[:, :output_horizon]
                fast_mode = "plain-flow"
                nfe = self.cfg.bootstrap_inference_steps
            elif stream_bootstrap:
                raw = self.model.bootstrap_streaming_with_features(
                    prepared.states,
                    features,
                    output_horizon=output_horizon,
                    slide_steps=self.slide_steps,
                    bootstrap_inference_steps=self.cfg.bootstrap_inference_steps,
                    substeps=self.cfg.fast_substeps,
                    noise_s=self.cfg.noise_s,
                )
                fast_mode = "bootstrap+stream"
                nfe = self.cfg.bootstrap_inference_steps + cycles * self.cfg.fast_substeps
            else:
                raw = self.model.predict_streaming_with_features(
                    prepared.states,
                    features,
                    output_horizon=output_horizon,
                    slide_steps=self.slide_steps,
                    substeps=self.cfg.fast_substeps,
                    noise_s=self.cfg.noise_s,
                )
                fast_mode = "stream"
                nfe = cycles * self.cfg.fast_substeps
        fast_enqueue_ms = (time.perf_counter() - fast_start) * 1000.0
        timing.record("action_expert_per_step", timing.get("action_expert") / max(nfe, 1))
        cache_age_ms = max(0.0, (prepared.received_at - features.captured_at) * 1000.0)
        timing.record("cache_age_ms", cache_age_ms)
        timing.tag(
            path=fast_mode,
            slow_mode=slow_mode,
            stream_bootstrap=stream_bootstrap,
            episode_id=episode_id,
            cache_id=features.cache_id,
            nfe=nfe,
        )
        return (
            raw,
            {
                "episode_start": episode_start,
                "stream_bootstrap": stream_bootstrap,
                "episode_id": episode_id,
                "fast_mode": fast_mode,
                "slow_mode": slow_mode,
                "cache_id": features.cache_id,
                "cache_age_ms": cache_age_ms,
                # Model calls enqueue asynchronously on CUDA. server_total_ms is
                # recorded after _response performs the synchronizing CPU copy.
                "fast_enqueue_ms": fast_enqueue_ms,
            },
            deferred_slow_job,
        )

    def _response(self, raw: torch.Tensor, *, expected_horizon: int) -> JSONResponse:
        # The D2H copy is where the asynchronously enqueued fast work is
        # actually awaited, so it is timed as its own stage.
        with timing.timed("response_decode", self.device):
            raw_np = raw.reshape(-1, self.Da).cpu().numpy()
        if raw_np.shape != (expected_horizon, self.Da):
            raise RuntimeError(
                "policy returned "
                f"{raw_np.shape}, expected {(expected_horizon, self.Da)}"
            )
        actions = self.maxmin.denormalize(raw_np)[:expected_horizon]
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite actions")
        return JSONResponse(content=ResponseMessage(actions, 0.0).serialize())

    def _record_metrics(self, metrics: dict[str, Any], total_ms: float) -> None:
        with self._metrics_lock:
            self._request_count += 1
            self._last_metrics = {**metrics, "server_total_ms": total_ms}

    def _flush_timing(self, total_ms: float) -> None:
        """Commit this request's stages to the JSONL log and print them."""
        timing.record("server_total", total_ms)
        episode = timing.episode()
        last = timing.flush()
        if not last:
            return
        overwatch.info(
            f"[timing] ep{episode} #{timing.num_requests()} {timing.format_last(last)}"
        )
        if timing.num_requests() % 50 == 0:
            overwatch.info(f"[timing] {timing.full_report()}")

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        started = time.perf_counter()
        try:
            with self._request_lock:
                try:
                    timing.reset()
                    now = time.monotonic()
                    timing.record(
                        "client_loop_gap_ms", (now - self._last_serve_time) * 1e3
                    )
                    prepared = self._prepare_request(payload)
                    raw, metrics, deferred_slow_job = self._raw_action(
                        prepared,
                        allow_slow_refresh=True,
                        output_horizon=self.Ta,
                    )
                    response = self._response(raw, expected_horizon=self.Ta)
                    if deferred_slow_job is not None:
                        metrics["slow_mode"] = self.slow_channel.submit(
                            deferred_slow_job.observations,
                            deferred_slow_job.instructions,
                            captured_at=deferred_slow_job.captured_at,
                            episode_id=deferred_slow_job.episode_id,
                        )
                    self._stream_started = True
                    total_ms = (time.perf_counter() - started) * 1000.0
                    self._record_metrics(metrics, total_ms)
                    overwatch.info(
                        f"Psi-R2 action {(self.Ta, self.Da)} | "
                        f"{metrics['fast_mode']} | slow={metrics['slow_mode']} | "
                        f"cache_age={metrics['cache_age_ms']:.1f}ms"
                    )
                    timing.tag(episode_start=metrics["episode_start"])
                    self._flush_timing(total_ms)
                    self._last_serve_time = time.monotonic()
                    return response
                except Exception:
                    timing.reset()
                    self._invalidate_runtime_locked()
                    raise
        except Exception as exc:  # noqa: BLE001 - HTTP boundary returns structured errors.
            overwatch.warning(traceback.format_exc())
            return JSONResponse(status_code=500, content={"status": str(exc)})

    def update_slow(self, payload: dict[str, Any]) -> JSONResponse:
        """Explicit slow endpoint for clients that want decoupled control."""
        try:
            with self._request_lock:
                try:
                    prepared = self._prepare_request(payload)
                    if not self._episode_active or "reset" in prepared.request.history:
                        episode_id = self._begin_episode_locked()
                    else:
                        episode_id = self.model.episode_id
                        # The explicit observation supersedes queued automatic
                        # refreshes captured before this request.
                        self.slow_channel.discard_pending()
                    features = self._refresh_slow_sync(prepared, episode_id)
                    return JSONResponse(
                        content={
                            "status": "ok",
                            "episode_id": episode_id,
                            "cache_id": features.cache_id,
                        }
                    )
                except Exception:
                    self._invalidate_runtime_locked()
                    raise
        except Exception as exc:  # noqa: BLE001 - HTTP boundary returns structured errors.
            overwatch.warning(traceback.format_exc())
            return JSONResponse(status_code=500, content={"status": str(exc)})

    def predict_fast(self, payload: dict[str, Any]) -> JSONResponse:
        """Run one PI-R2 cycle using fresh state and cached slow features."""
        started = time.perf_counter()
        try:
            with self._request_lock:
                try:
                    timing.reset()
                    now = time.monotonic()
                    timing.record(
                        "client_loop_gap_ms", (now - self._last_serve_time) * 1e3
                    )
                    prepared = self._prepare_request(payload)
                    output_horizon = self.slide_steps if self.cfg.rtc else self.Ta
                    raw, metrics, deferred_slow_job = self._raw_action(
                        prepared,
                        allow_slow_refresh=False,
                        output_horizon=output_horizon,
                    )
                    assert deferred_slow_job is None
                    response = self._response(raw, expected_horizon=output_horizon)
                    self._stream_started = True
                    total_ms = (time.perf_counter() - started) * 1000.0
                    self._record_metrics(metrics, total_ms)
                    timing.tag(episode_start=metrics["episode_start"])
                    self._flush_timing(total_ms)
                    self._last_serve_time = time.monotonic()
                    return response
                except Exception:
                    timing.reset()
                    self._invalidate_runtime_locked()
                    raise
        except Exception as exc:  # noqa: BLE001 - HTTP boundary returns structured errors.
            overwatch.warning(traceback.format_exc())
            return JSONResponse(status_code=500, content={"status": str(exc)})

    def health(self) -> JSONResponse:
        with self._metrics_lock:
            metrics = dict(self._last_metrics)
            request_count = self._request_count
            episode_count = self._episode_count
        return JSONResponse(
            content={
                "status": "ok",
                "policy": "psi-r2",
                "session_scope": "single-active-episode",
                "requests": request_count,
                "episodes": episode_count,
                "cache": self.model.slow_cache_info(),
                "slow_channel": self.slow_channel.status(),
                "last": metrics,
            }
        )

    def close(self) -> None:
        self.slow_channel.close()

    def timing_report(self) -> JSONResponse:
        return JSONResponse(
            content={
                "requests": timing.num_requests(),
                "episodes": timing.episode(),
                "log": str(timing.log_path()),
                "report": timing.full_report(),
            }
        )

    def run(self) -> None:
        app = FastAPI()
        app.post("/act")(self.predict_action)
        app.post("/slow")(self.update_slow)
        app.post("/fast")(self.predict_fast)
        app.get("/health")(self.health)
        app.get("/timing")(self.timing_report)
        overwatch.info(f"Psi-R2 server listens on {self.cfg.host}:{self.cfg.port}")
        try:
            uvicorn.run(app, host=self.cfg.host, port=self.cfg.port)
        finally:
            # Runs on Ctrl-C too, so a full eval session always ends in a report.
            if timing.num_requests():
                overwatch.info(
                    f"[timing] final report over {timing.num_requests()} requests, "
                    f"{timing.episode()} episodes:\n{timing.full_report()}"
                )
            self.close()


def serve(cfg: PsiR2ServerConfig) -> None:
    if cfg.policy.lower().replace("_", "-") not in {"psi-r2", "psir2"}:
        raise ValueError(
            f"serve_psi_r2 only supports --policy=psi-r2, got {cfg.policy!r}"
        )
    Server(cfg).run()


def _disable_cuda_launch_blocking() -> None:
    value = os.environ.get("CUDA_LAUNCH_BLOCKING", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        overwatch.warning(
            "CUDA_LAUNCH_BLOCKING was enabled; setting it to 0 for Psi-R2's "
            "asynchronous slow/fast channels"
        )
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    _disable_cuda_launch_blocking()
    overwatch.info(f"Args: {sys.argv}")
    cfg = tyro.cli(
        PsiR2ServerConfig,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
        args=sys.argv[1:],
    )
    serve(cfg)


if __name__ == "__main__":
    main()
