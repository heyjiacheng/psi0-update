"""HTTP server for a trained PI-R2 checkpoint, speaking psi's SIMPLE wire format.

Same CLI shape as ``serve_psi0`` so an existing SIMPLE eval client needs no
change:

    uv run --active --group pir2 --group serve serve_pir2 \\
      --host 0.0.0.0 --port 9000 \\
      --run-dir=$run_dir --ckpt-step=40000 \\
      --action-exec-horizon=24 --rtc

``--run-dir`` is the training ``--output-dir``; the checkpoint served is
``$run_dir/checkpoint-$ckpt_step`` (HF Trainer layout, with the processor copied
in by ``CheckpointFormatCallback``).  psi's ``checkpoints/ckpt_<step>`` layout is
accepted too, as is ``--ckpt-step=latest``.

Three checkpoint variants are supported, auto-detected from the checkpoint's own
model config (override with ``--ckpt-type``):

  ==========  ============================  ===========================================
  ckpt-type   detected when                 what ``--rtc`` switches on
  ==========  ============================  ===========================================
  plain_flow  ``streaming=False``           nothing — ``--rtc`` is rejected
  rtc         ``streaming_rtc_weight>0``    flow loop + clean-action inpaint of the
                                            front ``d = Tp - Ta`` in-flight actions
  pir2        ``streaming=True`` otherwise  rolling-buffer streaming inference: one
                                            DiT substep per request, buffer slides
                                            ``Ta`` positions per request
  ==========  ============================  ===========================================

Without ``--rtc`` every variant is served through the plain non-streaming flow
loop (``force_nonstreaming``), which is the honest apples-to-apples baseline.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pir2.data.embodiment_tags import EmbodimentTag
from pir2.policy.decoupled_policy import DecoupledGr00tPolicy

from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi.utils import timing
from psi.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)

# psi's flat SIMPLE action vector, in order. Sums to 36.
PSI_ACTION_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
    "torso_vx",
    "torso_vy",
    "torso_vyaw",
    "target_yaw",
]


@dataclass
class ServerConfig:
    """Serve a trained PI-R2 checkpoint over psi's ``/act`` HTTP protocol."""

    run_dir: str
    """Training output dir (the trainer's --output-dir), containing checkpoint-<step>/."""

    ckpt_step: int | str = "latest"
    """Checkpoint step to serve, or "latest" to pick the highest-numbered one."""

    action_exec_horizon: int | None = None
    """Actions returned (and executed) per request. Defaults to the full predicted
    chunk. For --rtc this is also the streaming buffer's slide per request, so it
    must divide the chunk length."""

    rtc: bool = False
    """Enable the checkpoint's reactive path (streaming for pir2, clean-action
    inpaint for rtc). Off = plain non-streaming flow for every variant."""

    ckpt_type: str = "auto"
    """One of auto / plain_flow / rtc / pir2. "auto" reads the checkpoint config."""

    nfe: int = 10
    """Denoising steps for the non-streaming flow loop, and for the episode-start
    warm-start that seeds the pir2 streaming buffer."""

    substeps_per_call: int = 1
    """DiT substeps per request on the pir2 streaming path. 1 is the πR² recipe:
    the per-position schedule closes its cycle in exactly one substep per slide."""

    rtc_inpaint_steps: int | None = None
    """Train-time-RTC only: how many front positions to clamp to already-committed
    actions. Defaults to (chunk_length - action_exec_horizon)."""

    image_delay: int | None = None
    """Optional image staleness in control ticks, fed to the checkpoint's
    delay_embedding. Only meaningful for a checkpoint trained with
    --image-delay-max > 0. None = do not condition on delay."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to serve. Must be present in the checkpoint's processor."""

    pad_action_dim: int | None = None
    """Right-pad the returned action with zeros to this width. None = no padding."""

    device: str = "cuda:0"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    port: int = 9000

    policy: str | None = None
    """Accepted and ignored — kept so serve_psi0 invocations copy over verbatim."""


def resolve_ckpt_dir(run_dir: Path, ckpt_step: int | str) -> tuple[Path, int | str]:
    """Map (run_dir, ckpt_step) onto a checkpoint directory.

    Understands the GR00T/HF trainer layout this baseline writes
    (``$run_dir/checkpoint-<step>``) and psi's layout
    (``$run_dir/checkpoints/ckpt_<step>``), so a run dir from either training
    stack can be served with the same command.
    """
    assert run_dir.exists(), f"run_dir {run_dir} does not exist!"

    candidates = [
        lambda step: run_dir / f"checkpoint-{step}",
        lambda step: run_dir / "checkpoints" / f"ckpt_{step}",
        lambda step: run_dir / f"ckpt_{step}",
    ]

    if str(ckpt_step) == "latest":
        found: list[tuple[int, Path]] = []
        for parent, prefix in ((run_dir, "checkpoint-"), (run_dir / "checkpoints", "ckpt_")):
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                if child.is_dir() and child.name.startswith(prefix):
                    suffix = child.name[len(prefix) :]
                    if suffix.isdigit():
                        found.append((int(suffix), child))
        if not found:
            raise FileNotFoundError(
                f"No checkpoint-<step>/ or checkpoints/ckpt_<step>/ found under {run_dir}"
            )
        step, path = max(found, key=lambda pair: pair[0])
        return path, step

    for build in candidates:
        path = build(ckpt_step)
        if path.is_dir():
            return path, ckpt_step
    raise FileNotFoundError(
        f"ckpt {ckpt_step} not found under {run_dir} "
        f"(tried {[str(build(ckpt_step)) for build in candidates]})"
    )


def detect_ckpt_type(model_config: Any) -> str:
    """Classify a checkpoint into plain_flow / rtc / pir2 from its model config.

    Mirrors the rule in the pi-r2-flow README: ``streaming=False`` is standard
    flow, a positive RTC regime weight means the train-time-RTC variant, and any
    other streaming checkpoint is πR².
    """
    if not bool(getattr(model_config, "streaming", False)):
        return "plain_flow"
    if float(getattr(model_config, "streaming_rtc_weight", 0.0)) > 0.0:
        return "rtc"
    return "pir2"


class Server:
    def __init__(self, cfg: ServerConfig):
        device_kind = str(cfg.device).split(":")[0]
        if device_kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        run_dir = Path(cfg.run_dir)
        ckpt_dir, ckpt_step = resolve_ckpt_dir(run_dir, cfg.ckpt_step)
        overwatch.info(f"Serving PI-R2 checkpoint {ckpt_dir}")
        overwatch.info(f"Using device: {cfg.device}")

        # DecoupledGr00tPolicy (not the plain Gr00tPolicy) because it owns the
        # streaming endpoints — seed_streaming_from_obs / reset_streaming_buffer —
        # and it is the subclass that threads `options` through to the action head.
        self.policy = DecoupledGr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(cfg.embodiment_tag),
            model_path=str(ckpt_dir),
            device=cfg.device,
            strict=True,
        )
        self.modality_configs = self.policy.modality_configs
        self.model_config = self.policy.model.config

        num_params = sum(p.numel() for p in self.policy.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params * 1e-6:.3f} Total", ctx_level=1)

        # The psi wire format is a single flat vector, so the checkpoint has to
        # expose exactly the SIMPLE action groups — otherwise the flat layout the
        # client unpacks would silently disagree with what the model predicts.
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        if set(self.action_keys) != set(PSI_ACTION_KEYS):
            raise ValueError(
                f"checkpoint action keys {self.action_keys} do not match psi's SIMPLE "
                f"action layout {PSI_ACTION_KEYS}"
            )

        self.ckpt_type = (
            detect_ckpt_type(self.model_config) if cfg.ckpt_type == "auto" else cfg.ckpt_type
        )
        if self.ckpt_type not in ("plain_flow", "rtc", "pir2"):
            raise ValueError(f"--ckpt-type must be auto/plain_flow/rtc/pir2, got {cfg.ckpt_type}")
        overwatch.info(
            f"Checkpoint type: {self.ckpt_type} "
            f"(streaming={getattr(self.model_config, 'streaming', False)}, "
            f"schedule_mode={getattr(self.model_config, 'streaming_schedule_mode', None)}, "
            f"rtc_weight={getattr(self.model_config, 'streaming_rtc_weight', 0.0)})"
        )

        # Tp = chunk length the head predicts, Ta = actions handed back per request.
        self.Tp = len(self.modality_configs["action"].delta_indices)
        self.Ta = cfg.action_exec_horizon or self.Tp
        assert self.Ta <= self.Tp, (
            f"action_exec_horizon {self.Ta} exceeds the checkpoint's action horizon {self.Tp}"
        )

        self.enable_rtc = cfg.rtc
        self.rtc_inpaint_steps = None
        if self.enable_rtc:
            if self.ckpt_type == "plain_flow":
                raise ValueError(
                    "--rtc is not supported for a plain_flow checkpoint: it was trained "
                    "without any clean-action / per-position exposure, so a reactive path "
                    "would be out of distribution. Drop --rtc, or serve a pir2/rtc ckpt."
                )
            if self.ckpt_type == "pir2":
                schedule_mode = getattr(self.model_config, "streaming_schedule_mode", "pir2")
                if schedule_mode != "pir2":
                    raise ValueError(
                        f"streaming checkpoint has streaming_schedule_mode={schedule_mode!r}; "
                        "only the 'pir2' schedule is supported at inference (the legacy 'v0' "
                        "staircase raises NotImplementedError in _streaming_inference)."
                    )
                assert self.Tp % self.Ta == 0, (
                    f"pir2 streaming slides action_exec_horizon={self.Ta} positions per "
                    f"request, so it must divide the action horizon {self.Tp}"
                )
                overwatch.info(
                    f"RTC (pir2 streaming) enabled: slide_steps={self.Ta}, "
                    f"substeps_per_call={cfg.substeps_per_call}, chunk={self.Tp}"
                )
            else:  # rtc
                d_max = int(getattr(self.model_config, "streaming_rtc_d_max", 0))
                self.rtc_inpaint_steps = (
                    cfg.rtc_inpaint_steps
                    if cfg.rtc_inpaint_steps is not None
                    else self.Tp - self.Ta
                )
                assert self.rtc_inpaint_steps > 0, (
                    "train-time-RTC inpaint needs a positive front width: either lower "
                    "--action-exec-horizon below the action horizon, or set "
                    "--rtc-inpaint-steps explicitly"
                )
                assert self.rtc_inpaint_steps <= d_max, (
                    f"rtc_inpaint_steps={self.rtc_inpaint_steps} exceeds the checkpoint's "
                    f"trained streaming_rtc_d_max={d_max}"
                )
                overwatch.info(
                    f"RTC (train-time-RTC inpaint) enabled: d={self.rtc_inpaint_steps}, "
                    f"d_max={d_max}, chunk={self.Tp}"
                )

        self.cfg = cfg
        # Last predicted chunk, kept per action group in raw (unnormalized) robot
        # units. The RTC path re-sends the slice the robot committed to as the clean
        # front; keeping the group dict (rather than the flat vector) means the
        # inpaint is assembled in the checkpoint's own key order, which is what
        # _normalize_pad_inpaint splits on.
        self.previous_action: dict[str, np.ndarray] | None = None
        self.last_serve_time = time.monotonic()

        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = timing.open_log(
            run_dir
            / "inference_timing"
            / f"pir2_ckpt{ckpt_step}_Ta{self.Ta}_"
            f"{'rtc' if self.enable_rtc else 'nortc'}_{stamp}.jsonl"
        )
        if path is not None:
            overwatch.info(f"Per-request inference timings -> {path}")

    # ---- psi <-> pir2 observation / action mapping ---------------------------

    @staticmethod
    def _ensure_btd(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:  # (D,)
            return arr[None, None, ...]
        if arr.ndim == 2:  # (T, D)
            return arr[None, ...]
        if arr.ndim == 3:  # (B, T, D)
            return arr
        raise ValueError(f"Array must be 1-3D, got shape {arr.shape}")

    @staticmethod
    def _to_batched_video(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if arr.ndim == 3:  # (H, W, C)
            arr = arr[None, None, ...]
        elif arr.ndim == 4:  # (T, H, W, C)
            arr = arr[None, ...]
        elif arr.ndim != 5:
            raise ValueError(f"Video array must be 3-5D, got shape {arr.shape}")
        return arr

    @staticmethod
    def _split_state_from_psi(
        proprio_joint_positions: np.ndarray,
        amo_policy_command: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Real-robot psi state layout -> SIMPLE per-modality state dict."""
        proprio = Server._ensure_btd(np.asarray(proprio_joint_positions, dtype=np.float32))
        command = Server._ensure_btd(np.asarray(amo_policy_command, dtype=np.float32))

        left_hand = np.concatenate(
            [proprio[..., 29:32], proprio[..., 34:36], proprio[..., 32:34]],
            axis=-1,
        )
        return {
            "left_hand": left_hand,
            "right_hand": proprio[..., 36:43],
            "left_arm": proprio[..., 15:22],
            "right_arm": proprio[..., 22:29],
            "rpy": np.concatenate([proprio[..., 13:15], proprio[..., 12:13]], axis=-1),
            "height": command[..., 6:7],
        }

    @staticmethod
    def _split_state_from_simple(states: Any) -> dict[str, np.ndarray]:
        """SIMPLE's flat 32-dim ``states`` -> per-modality state dict.

        Layout: left_hand thumb(3) + middle(2) + index(2), right_hand(7),
        left_arm(7), right_arm(7), rpy(3), height(1).
        """
        arr = Server._ensure_btd(np.asarray(states, dtype=np.float32))
        if arr.shape[-1] != 32:
            raise ValueError(f"Expected 'states' with 32 dims, got {arr.shape[-1]}")
        return {
            "left_hand": np.concatenate(
                [arr[..., 0:3], arr[..., 3:5], arr[..., 5:7]], axis=-1
            ),
            "right_hand": arr[..., 7:14],
            "left_arm": arr[..., 14:21],
            "right_arm": arr[..., 21:28],
            "rpy": arr[..., 28:31],
            "height": arr[..., 31:32],
        }

    def _build_observation(self, request: RequestMessage) -> dict[str, Any]:
        image_dict = request.image
        state_dict = request.state

        video_key = self.modality_configs["video"].modality_keys[0]
        if video_key in image_dict:
            raw_video = image_dict[video_key]
        elif "rgb_head_stereo_left" in image_dict:
            raw_video = image_dict["rgb_head_stereo_left"]
        elif len(image_dict) == 1:
            raw_video = next(iter(image_dict.values()))
        else:
            raise KeyError(
                f"Missing video key '{video_key}' in request.image (got {list(image_dict)})"
            )

        if "proprio_joint_positions" in state_dict and "amo_policy_command" in state_dict:
            state_parts = self._split_state_from_psi(
                state_dict["proprio_joint_positions"], state_dict["amo_policy_command"]
            )
        elif "states" in state_dict:
            state_parts = self._split_state_from_simple(state_dict["states"])
        else:
            raise KeyError(
                "Missing psi-style state ('proprio_joint_positions' + 'amo_policy_command') "
                "or SIMPLE 'states' in request.state"
            )

        state: dict[str, np.ndarray] = {}
        for state_key in self.modality_configs["state"].modality_keys:
            if state_key not in state_parts:
                raise KeyError(f"Missing state key '{state_key}' in the psi state mapping")
            state[state_key] = np.ascontiguousarray(state_parts[state_key], dtype=np.float32)

        language_key = self.modality_configs["language"].modality_keys[0]
        return {
            "video": {video_key: self._to_batched_video(raw_video)},
            "state": state,
            "language": {language_key: [[request.instruction]]},
        }

    @staticmethod
    def _pick_action_group(action: dict[str, Any], key: str) -> np.ndarray:
        """Read one action group, tolerating the ``action.<key>`` prefix."""
        raw = action.get(key)
        if raw is None:
            raw = action.get(f"action.{key}")
        if raw is None:
            raise KeyError(f"Missing action key '{key}' in the model output")
        return Server._ensure_btd(np.asarray(raw, dtype=np.float32))

    def _action_to_psi_format(self, action: dict[str, Any]) -> np.ndarray:
        """Per-modality action dict -> psi's flat (T, 36) chunk, in psi's order."""
        chunk = np.concatenate(
            [self._pick_action_group(action, key) for key in PSI_ACTION_KEYS], axis=-1
        )  # (B, T, 36)
        if chunk.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {chunk.shape[0]}")
        return chunk[0]

    # ---- inference paths -----------------------------------------------------

    def _plain_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_inference_timesteps": self.cfg.nfe,
            "force_nonstreaming": True,
        }
        self._maybe_add_image_delay(options)
        return options

    def _maybe_add_image_delay(self, options: dict[str, Any]) -> None:
        if self.cfg.image_delay is None:
            return
        d_max = int(getattr(self.model_config, "image_delay_max", 0))
        if d_max <= 0:
            return
        options["image_delay"] = max(0, min(int(self.cfg.image_delay), d_max))

    def _rtc_inpaint(self) -> np.ndarray:
        """The already-committed actions that pin the next chunk's front d slots.

        ``previous_action`` is the last full chunk; the client executed its first
        ``Ta`` entries, so positions ``[Ta, Ta + d)`` of that chunk are the ones the
        new chunk's front ``d`` slots must agree with. Assembled in the checkpoint's
        action-key order and left in raw robot units — the policy normalizes and
        pads it to the head's action dim.
        """
        assert self.previous_action is not None
        d = int(self.rtc_inpaint_steps)
        parts = []
        for key in self.action_keys:
            group = self.previous_action[key][self.Ta : self.Ta + d]
            if group.shape[0] < d:  # chunk ran out: hold the last commanded value
                pad = np.repeat(group[-1:], d - group.shape[0], axis=0)
                group = np.concatenate([group, pad], axis=0)
            parts.append(group)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _predict(
        self, observation: dict[str, Any], is_episode_start: bool
    ) -> dict[str, np.ndarray]:
        """Run one inference; returns the unnormalized per-group chunk, each (B, Tp, d)."""
        if not self.enable_rtc:
            timing.tag(path="plain")
            action, _ = self.policy.get_action(observation, self._plain_options())
            return action

        if self.ckpt_type == "pir2":
            if is_episode_start:
                # The rolling buffer has no cold-start path: warm it up with a full
                # non-streaming flow chunk placed on the per-position τ manifold.
                overwatch.info("===Episode start, seeding pir2 streaming buffer===")
                timing.tag(path="seed")
                self.policy.reset_streaming_buffer()
                action, _ = self.policy.seed_streaming_from_obs(
                    observation,
                    num_inference_timesteps=self.cfg.nfe,
                    t_image_capture=time.time(),
                    slide_steps=self.Ta,
                )
                return action

            timing.tag(path="pir2")
            options: dict[str, Any] = {
                "slide_steps": self.Ta,
                "num_inference_steps_per_call": self.cfg.substeps_per_call,
                "num_inference_timesteps": self.cfg.nfe,
            }
            self._maybe_add_image_delay(options)
            action, _ = self.policy.get_action(observation, options)
            return action

        # Train-time RTC: non-streaming flow loop with the front d positions pinned
        # to the actions the robot already committed to.
        if is_episode_start or self.previous_action is None:
            overwatch.info("===Episode start, running unconditioned flow===")
            timing.tag(path="uncond")
            action, _ = self.policy.get_action(observation, self._plain_options())
            return action

        timing.tag(path="rtc")
        options = {
            "num_inference_timesteps": self.cfg.nfe,
            "force_nonstreaming": True,
            "inpaint": self._rtc_inpaint(),
            "action_horizon": int(self.rtc_inpaint_steps),
        }
        self._maybe_add_image_delay(options)
        action, _ = self.policy.get_action(observation, options)
        return action

    # ---- HTTP ----------------------------------------------------------------

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            timing.reset()
            request_t0 = time.perf_counter()
            request = RequestMessage.deserialize(payload)

            overwatch.info(f"Instruction: {request.instruction}")
            overwatch.info(f"history_dict: {request.history}")

            is_episode_start = self.previous_action is None or "reset" in request.history
            if is_episode_start:
                timing.new_episode()
            timing.tag(episode_start=is_episode_start)

            current_time = time.monotonic()
            if not is_episode_start:
                timing.record("client_loop_gap_ms", (current_time - self.last_serve_time) * 1e3)

            observation = self._build_observation(request)
            action_dict = self._predict(observation, is_episode_start)

            # Drop the batch dim and keep the groups for the next RTC inpaint.
            self.previous_action = {
                key: self._pick_action_group(action_dict, key)[0].copy()
                for key in self.action_keys
            }
            pred_actions = self._action_to_psi_format(self.previous_action)[: self.Ta]
            if self.cfg.pad_action_dim is not None:
                pad_n = self.cfg.pad_action_dim - pred_actions.shape[-1]
                if pad_n < 0:
                    raise ValueError(
                        f"--pad-action-dim={self.cfg.pad_action_dim} is narrower than the "
                        f"model's {pred_actions.shape[-1]}-dim action"
                    )
                if pad_n > 0:
                    pred_actions = np.concatenate(
                        [pred_actions, np.zeros((pred_actions.shape[0], pad_n), np.float32)],
                        axis=-1,
                    )
            overwatch.info(f"Return Action ({pred_actions.shape})")

            timing.record("server_total", (time.perf_counter() - request_t0) * 1e3)
            ep = timing.episode()
            last = timing.flush()
            overwatch.info(f"[timing] ep{ep} #{timing.num_requests()} {timing.format_last(last)}")
            if timing.num_requests() % 50 == 0:
                overwatch.info(f"[timing] {timing.full_report()}")

            self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions.astype(np.float32), 0.0)
            return JSONResponse(content=response.serialize())

        except Exception as exc:
            import traceback

            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{exc}"}}')

    def reset(self) -> JSONResponse:
        """Explicit episode boundary, for clients that don't send history["reset"]."""
        self.previous_action = None
        if self.ckpt_type == "pir2":
            self.policy.reset_streaming_buffer()
        return JSONResponse(content={"status": "ok"})

    def run(self, host: str, port: int) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.post("/reset")(self.reset)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        self.app.get("/timing")(
            lambda: JSONResponse(
                content={
                    "requests": timing.num_requests(),
                    "episodes": timing.episode(),
                    "log": str(timing.log_path()),
                    "report": timing.full_report(),
                }
            )
        )
        overwatch.info(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as exc:
            overwatch.warning(f"Server crashed, {exc}")
        finally:
            if timing.num_requests():
                overwatch.info(
                    f"[timing] final report over {timing.num_requests()} requests, "
                    f"{timing.episode()} episodes:\n{timing.full_report()}"
                )
            overwatch.info("Server stopped.")
            sys.exit(1)


def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing PI-R2")
    server = Server(cfg)
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


def main() -> None:
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(ServerConfig, args=sys.argv[1:])
    serve(cfg)


if __name__ == "__main__":
    main()
