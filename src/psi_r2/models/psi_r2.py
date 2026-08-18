"""Psi0 model wrapper with PI-R2 reactive inference channels.

The learned architecture and checkpoint key layout stay identical to Psi0.
Only inference runtime state is added:

* a slow image/language channel that caches Qwen3-VL hidden states;
* a fast channel that combines cached hidden states with fresh proprioception;
* a PI-R2 rolling action buffer with per-position Psi0 flow timesteps.

Training support is intentionally out of scope for this module. A legacy Psi0
checkpoint is structurally loadable, but it was not trained on the PI-R2 ramp
schedule or stale vision features; meaningful policy quality requires a later
PI-R2 finetuning pass.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from psi.models.psi0 import ActionTransformerModel, Psi0Model
from psi.utils import timing
from psi_r2.models.schedule import PiR2RollingBuffer, PiR2Schedule


@dataclass(frozen=True)
class SlowFeatures:
    """Cached output of the slow image/language channel."""

    views: torch.Tensor
    attention_mask: torch.Tensor
    captured_at: float
    cache_id: int = 0
    # Wall time this encode cost. Carried on the features rather than written
    # to the global timing row because encode_slow usually runs on the async
    # slow worker, off the request thread that owns that row.
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0

    @property
    def batch_size(self) -> int:
        return int(self.views.shape[0])


class PsiR2Model(Psi0Model):
    """Psi0 backbone plus PI-R2 inference-only runtime behavior."""

    def __init__(self, model_cfg: Any, vlm_model: Any):
        super().__init__(model_cfg=model_cfg, vlm_model=vlm_model)
        self._init_psi_r2_runtime()

    def _init_psi_r2_runtime(self) -> None:
        # These are ordinary Python attributes, not registered tensors, so the
        # state_dict remains byte-for-byte compatible with a Psi0 checkpoint.
        self._slow_cache_lock = threading.Lock()
        self._slow_forward_lock = threading.Lock()
        self._rolling_lock = threading.RLock()
        self._slow_cache: SlowFeatures | None = None
        self._slow_error: str | None = None
        self._episode_id = 0
        self._cache_counter = 0
        self._rolling: PiR2RollingBuffer | None = None

    @classmethod
    def from_pretrained(cls, run_dir, ckpt_step, launch_config, device):
        """Load the exact legacy Psi0 checkpoint layout, then attach R2 state.

        ``Psi0Model.from_pretrained`` constructs ``Psi0Model`` explicitly rather
        than ``cls``. Rebinding the instance to this parameter-compatible
        subclass lets us reuse its strict, well-tested loader without copying
        it or changing anything under ``src/psi``.
        """
        model = Psi0Model.from_pretrained(
            run_dir=run_dir,
            ckpt_step=ckpt_step,
            launch_config=launch_config,
            device=device,
        )
        model.__class__ = cls
        cls._init_psi_r2_runtime(model)
        return model

    def _autocast_context(self):
        device_type = str(self.device).split(":", 1)[0]
        if device_type in {"cpu", "cuda", "xpu"}:
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return nullcontext()

    @property
    def episode_id(self) -> int:
        with self._slow_cache_lock:
            return self._episode_id

    def reset_runtime(self) -> int:
        """Clear slow/fast episode state and invalidate in-flight slow jobs."""
        with self._slow_cache_lock:
            self._episode_id += 1
            episode_id = self._episode_id
            self._slow_cache = None
            self._slow_error = None
        with self._rolling_lock:
            if self._rolling is not None:
                self._rolling.reset()
            self._rolling = None
        return episode_id

    def install_slow_features(self, features: SlowFeatures, episode_id: int) -> bool:
        """Atomically install slow features if their episode is still current."""
        with self._slow_cache_lock:
            if episode_id != self._episode_id:
                return False
            if (
                self._slow_cache is not None
                and features.captured_at < self._slow_cache.captured_at
            ):
                # A reset bootstrap or explicit /slow call may finish before an
                # older background refresh. Never let that older image win.
                return False
            self._cache_counter += 1
            self._slow_cache = replace(features, cache_id=self._cache_counter)
            self._slow_error = None
            return True

    def record_slow_error(self, error: BaseException, episode_id: int) -> None:
        with self._slow_cache_lock:
            if episode_id == self._episode_id:
                self._slow_error = f"{type(error).__name__}: {error}"

    def get_slow_features(self) -> SlowFeatures | None:
        with self._slow_cache_lock:
            return self._slow_cache

    def slow_cache_info(self) -> dict[str, Any]:
        with self._slow_cache_lock:
            cached = self._slow_cache
            return {
                "episode_id": self._episode_id,
                "cache_id": cached.cache_id if cached is not None else None,
                "captured_at": cached.captured_at if cached is not None else None,
                "error": self._slow_error,
            }

    def _prepare_slow_inputs(
        self,
        observations: list[list[Image.Image]],
        instructions: list[str],
    ) -> dict[str, torch.Tensor]:
        if len(observations) != len(instructions):
            raise ValueError(
                f"observations batch ({len(observations)}) does not match "
                f"instructions batch ({len(instructions)})"
            )
        if not observations:
            raise ValueError("slow channel requires at least one observation")

        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        for observation, instruction in zip(observations, instructions):
            content = [{"type": "image", "image": image} for image in observation]
            content.append({"type": "text", "text": instruction})
            messages = [[{"role": "user", "content": content}]]
            texts = [
                self.vlm_processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(
                messages, image_patch_size=16
            )
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            batch_input_ids.append(inputs["input_ids"].squeeze(0))
            batch_attention_mask.append(inputs["attention_mask"].squeeze(0))
            batch_pixel_values.append(inputs["pixel_values"])
            batch_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))

        try:
            return {
                "input_ids": torch.stack(batch_input_ids),
                "attention_mask": torch.stack(batch_attention_mask),
                "pixel_values": torch.stack(batch_pixel_values),
                "image_grid_thw": torch.stack(batch_image_grid_thw),
            }
        except RuntimeError as exc:
            raise ValueError(
                "Psi-R2 currently requires equal token/patch lengths within a batch"
            ) from exc

    @torch.inference_mode()
    def encode_slow(
        self,
        observations: list[list[Image.Image]],
        instructions: list[str],
        *,
        captured_at: float | None = None,
    ) -> SlowFeatures:
        """Run image/text preprocessing and the Qwen backbone only."""
        # Qwen processor/model calls are serialized; this also prevents a reset
        # bootstrap from overlapping an older queued slow refresh on one model.
        with self._slow_forward_lock:
            timing.sync(self.device)
            t0 = time.perf_counter()
            inputs = self._prepare_slow_inputs(observations, instructions)
            timing.sync(self.device)
            t1 = time.perf_counter()
            with self._autocast_context():
                output = self.vlm_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = output.hidden_states[-1].unsqueeze(1)
            timing.sync(self.device)
            t2 = time.perf_counter()
        return SlowFeatures(
            views=hidden,
            attention_mask=inputs["attention_mask"],
            captured_at=float(captured_at if captured_at is not None else time.time()),
            preprocess_ms=(t1 - t0) * 1e3,
            forward_ms=(t2 - t1) * 1e3,
        )

    def _validate_fast_inputs(
        self, states: torch.Tensor, features: SlowFeatures
    ) -> None:
        if states.ndim != 3:
            raise ValueError(
                f"states must have shape [B,To,Ds], got {tuple(states.shape)}"
            )
        if states.shape[0] != features.batch_size:
            raise ValueError(
                f"state batch {states.shape[0]} does not match cached slow batch "
                f"{features.batch_size}"
            )

    def _predict_velocity(
        self,
        actions: torch.Tensor,
        sigma: torch.Tensor,
        states: torch.Tensor,
        features: SlowFeatures,
    ) -> torch.Tensor:
        self._validate_fast_inputs(states, features)
        if sigma.ndim not in {1, 2}:
            raise ValueError(
                f"sigma must have shape [B] or [B,T], got {tuple(sigma.shape)}"
            )
        timesteps = sigma * float(self.noise_scheduler.config.num_train_timesteps)
        if sigma.ndim == 2:
            return self._predict_per_position_velocity(
                actions,
                timesteps,
                states,
                features,
            )
        return self.action_header(
            hidden_states=None,
            timestep=timesteps,
            joint_attention_kwargs={
                "action_hidden_embeds": actions,
                "views": features.views,
                "obs": states,
                "traj2ds": None,
            },
            vlm_attn_mask=features.attention_mask,
            return_dict=True,
        ).action

    @staticmethod
    def _forward_block_with_observation_time(
        block: torch.nn.Module,
        action_hidden_states: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        action_temb: torch.Tensor,
        obs_temb: torch.Tensor,
        obs_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one legacy Psi block with an explicit observation timestep.

        The original block uses the final action position's embedding for every
        observation token when action time is per-position. PI-R2 instead has
        a dedicated state-token time. Keeping this adapter in ``psi_r2``
        preserves the original implementation and every checkpoint parameter.
        """
        (
            norm_action_hidden_states,
            gate_msa_act,
            shift_mlp_act,
            scale_mlp_act,
            gate_mlp_act,
        ) = block.norm1_act(action_hidden_states, emb=action_temb)

        if block.context_pre_only:
            norm_obs_hidden_states = block.norm1_obs(obs_hidden_states, obs_temb)
            gate_msa_obs = None
            shift_mlp_obs = None
            scale_mlp_obs = None
            gate_mlp_obs = None
        else:
            (
                norm_obs_hidden_states,
                gate_msa_obs,
                shift_mlp_obs,
                scale_mlp_obs,
                gate_mlp_obs,
            ) = block.norm1_obs(obs_hidden_states, emb=obs_temb)

        act_attn_output, obs_attn_output = block.attn(
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask,
        )

        action_hidden_states = action_hidden_states + gate_msa_act * act_attn_output
        norm_action_hidden_states = block.norm2_act(action_hidden_states)
        norm_action_hidden_states = norm_action_hidden_states * (
            1 + scale_mlp_act
        ) + shift_mlp_act
        action_hidden_states = action_hidden_states + gate_mlp_act * block.ff_act(
            norm_action_hidden_states
        )

        if block.context_pre_only:
            return action_hidden_states, None

        assert gate_msa_obs is not None
        assert shift_mlp_obs is not None
        assert scale_mlp_obs is not None
        assert gate_mlp_obs is not None
        assert block.norm2_obs is not None
        assert block.ff_obs is not None
        obs_hidden_states = obs_hidden_states + gate_msa_obs * obs_attn_output
        norm_obs_hidden_states = block.norm2_obs(obs_hidden_states)
        norm_obs_hidden_states = norm_obs_hidden_states * (
            1 + scale_mlp_obs
        ) + shift_mlp_obs
        obs_hidden_states = obs_hidden_states + gate_mlp_obs * block.ff_obs(
            norm_obs_hidden_states
        )
        return action_hidden_states, obs_hidden_states

    def _predict_per_position_velocity(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        features: SlowFeatures,
    ) -> torch.Tensor:
        """Psi transformer forward with token-wise action and fixed obs time.

        This mirrors ``ActionTransformerModel.forward`` with one intentional
        PI-R2 difference: observation/VLM tokens receive their own fixed
        endpoint rather than inheriting the evolving final action slot's time.
        It adds no modules or learned parameters.
        """
        head = self.action_header
        if not isinstance(head, ActionTransformerModel):
            raise TypeError(
                "Psi-R2 per-position inference requires the Psi0 transformer head"
            )
        if head.combined_temb:
            raise ValueError(
                "Psi-R2 per-position inference does not support combined_temb"
            )
        if timesteps.shape != actions.shape[:2]:
            raise ValueError(
                "per-position timesteps must match action positions: "
                f"got {tuple(timesteps.shape)} and {tuple(actions.shape[:2])}"
            )

        action_temb = head.time_ins_embed(timesteps)
        # PI-R2 fixes its state token at tau=0, the reference policy's pure-noise
        # endpoint. Under sigma = 1 - tau/noise_s, this is Psi's sigma=1 endpoint,
        # i.e. train_timesteps. Psi combines cached VLM and state tokens into one
        # observation sequence, so that fixed condition applies to all of them.
        obs_timestep = torch.full(
            (timesteps.shape[0],),
            float(self.noise_scheduler.config.num_train_timesteps),
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        obs_temb = head.time_ins_embed(obs_timestep)
        action_hidden_states = head.action_proj_in(actions)
        obs_hidden_states, obs_token_mask = head.obs_proj(
            views=features.views,
            obs=states,
            traj2ds=None,
            text_embeddings=None,
            vlm_attn_mask=features.attention_mask,
        )

        for block in head.transformer_blocks:
            action_hidden_states, obs_hidden_states = (
                self._forward_block_with_observation_time(
                    block,
                    action_hidden_states,
                    obs_hidden_states,
                    action_temb,
                    obs_temb,
                    obs_token_mask,
                )
            )

        return head.action_proj_out(x=action_hidden_states, t=action_temb)

    @torch.inference_mode()
    def predict_full_flow_with_features(
        self,
        states: torch.Tensor,
        features: SlowFeatures,
        *,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Run ordinary scalar-time Psi0 flow using an already-cached VLM."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self._validate_fast_inputs(states, features)
        batch_size = states.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=self.device,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        with self._autocast_context():
            for timestep in self.noise_scheduler.timesteps:
                sigma = timestep.expand(batch_size).to(self.device) / float(
                    self.noise_scheduler.config.num_train_timesteps
                )
                velocity = self._predict_velocity(actions, sigma, states, features)
                actions = self.noise_scheduler.step(
                    model_output=velocity,
                    timestep=timestep,
                    sample=actions,
                ).prev_sample
        return actions.float()

    def seed_streaming_buffer(
        self,
        clean_chunk: torch.Tensor,
        *,
        slide_steps: int,
        noise_s: float = 0.999,
    ) -> None:
        schedule = PiR2Schedule(
            horizon=self.action_horizon,
            slide_steps=slide_steps,
            train_timesteps=int(self.noise_scheduler.config.num_train_timesteps),
            noise_s=noise_s,
        )
        rolling = PiR2RollingBuffer(schedule)
        rolling.seed(clean_chunk)
        self._rolling = rolling

    @torch.inference_mode()
    def predict_streaming_with_features(
        self,
        states: torch.Tensor,
        features: SlowFeatures,
        *,
        output_horizon: int,
        slide_steps: int,
        substeps: int = 1,
        noise_s: float = 0.999,
    ) -> torch.Tensor:
        """Run enough PI-R2 fast updates to fill the requested response.

        The reference client consumes exactly ``d`` actions per fast query. The
        explicit fast channel uses that cadence. For the legacy SIMPLE/Psi
        ``/act`` response, ``output_horizon`` may be a multiple of ``d``; this
        method batches that many cycles while keeping every emitted slice in
        PI-R2's newly-clean ``[d:2d]`` region.
        """
        if output_horizon < 1:
            raise ValueError("output_horizon must be positive")
        if output_horizon % slide_steps != 0:
            raise ValueError(
                "action_exec_horizon must be divisible by PI-R2 slide_steps; "
                f"got {output_horizon} and {slide_steps}"
            )
        self._validate_fast_inputs(states, features)

        with self._rolling_lock:
            if self._rolling is None or not self._rolling.seeded:
                raise RuntimeError("PI-R2 buffer is not seeded; run bootstrap first")
            if self._rolling.schedule.slide_steps != slide_steps:
                raise ValueError(
                    "slide_steps cannot change within an episode: "
                    f"seeded with {self._rolling.schedule.slide_steps}, got {slide_steps}"
                )
            if abs(self._rolling.schedule.noise_s - noise_s) > 1e-9:
                raise ValueError("noise_s cannot change within an episode")

            rolling = self._rolling
            assert rolling.actions is not None and rolling.sigma is not None
            actions_before = rolling.actions.clone()
            sigma_before = rolling.sigma.clone()
            emitted = []

            def velocity_fn(actions: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                return self._predict_velocity(actions, sigma, states, features)

            try:
                with self._autocast_context():
                    for _ in range(output_horizon // slide_steps):
                        snapshot = rolling.denoise_and_slide(
                            velocity_fn, substeps=substeps
                        )
                        emitted.append(rolling.emitted(snapshot))
                output = torch.cat(emitted, dim=1).float()
                if not torch.isfinite(output).all():
                    raise FloatingPointError(
                        "PI-R2 action head produced non-finite actions"
                    )
                return output
            except Exception:
                # A failed HTTP request executes no actions. Roll the sampler
                # back as one transaction so a retry cannot silently skip a
                # partially advanced region.
                rolling.actions = actions_before
                rolling.sigma = sigma_before
                raise

    @torch.inference_mode()
    def bootstrap_streaming_with_features(
        self,
        states: torch.Tensor,
        features: SlowFeatures,
        *,
        output_horizon: int,
        slide_steps: int,
        bootstrap_inference_steps: int,
        substeps: int = 1,
        noise_s: float = 0.999,
    ) -> torch.Tensor:
        """Warm-start with ordinary flow, manifold-seed, then emit PI-R2 actions."""
        with self._rolling_lock:
            clean = self.predict_full_flow_with_features(
                states,
                features,
                num_inference_steps=bootstrap_inference_steps,
            )
            self.seed_streaming_buffer(clean, slide_steps=slide_steps, noise_s=noise_s)
            return self.predict_streaming_with_features(
                states,
                features,
                output_horizon=output_horizon,
                slide_steps=slide_steps,
                substeps=substeps,
                noise_s=noise_s,
            )

    @torch.inference_mode()
    def predict_action(
        self,
        observations: list[list[Image.Image]],
        states: torch.Tensor,
        instructions: list[str],
        num_inference_steps: int,
        traj2ds=None,
        **kwargs: str,
    ) -> torch.Tensor:
        """Synchronous compatibility path using the split implementation."""
        if traj2ds is not None:
            raise ValueError("Psi-R2 inference currently supports traj2ds=None only")
        features = self.encode_slow(observations, instructions)
        return self.predict_full_flow_with_features(
            states, features, num_inference_steps=num_inference_steps
        )
