# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from pir2.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from pir2.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from pir2.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Async-VLM training: optional learnable embedding for image staleness in control ticks.
        # Added to action_features (broadcast over T positions), same pattern as positional embedding.
        # Zero-init so untrained model behaves identically to no-embedding case.
        if config.image_delay_embed_dim > 0:
            self.delay_embedding = nn.Embedding(config.image_delay_max + 1, self.input_embedding_dim)
            nn.init.zeros_(self.delay_embedding.weight)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def sample_time_per_position(self, batch_size, device, dtype):
        # Per-position τ for streaming (TEDI) flow training. Mirrors StreamingFlowPolicy
        # regime mix: constant / linearly-increasing / random / chunk-wise / train-time RTC.
        T = self.config.action_horizon
        weights = torch.tensor([
            self.config.streaming_constant_weight,
            self.config.streaming_linear_weight,
            self.config.streaming_random_weight,
            self.config.streaming_chunk_wise_weight,
            self.config.streaming_rtc_weight,
        ], dtype=torch.float32)
        regime = torch.multinomial(weights, num_samples=1).item()

        eps = 0.0 # self.config.noise_s / self.config.num_timestep_buckets

        if regime == 0:                                                       # constant: same τ for all T
            t = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            t = (1 - t) * self.config.noise_s
            return t.unsqueeze(1).expand(-1, T)
        if regime == 1:                                                       # linearly increasing along chunk
            raise NotImplementedError("Linearly increasing along chunk is not implemented")
        if regime == 2:                                                       # random per position
            raise NotImplementedError("Random per position is not implemented")
        if regime == 4:                                                       # train-time RTC
            # Per BATCH: pick d ∈ [0, rtc_d_max]. Front d positions at noise_s (clean clamp),
            # rest at scalar τ sampled from the Beta distribution. Equivalent to flow training
            # with random "front d clean actions as input" conditioning. Stacks with
            # mask_clean_end=True so loss is masked at clamped positions.
            d_max = max(self.config.streaming_rtc_d_max, 0)
            d = int(torch.randint(0, d_max + 1, (1,)).item()) if d_max > 0 else 0
            t_scalar = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            t_scalar = (1 - t_scalar) * self.config.noise_s                   # (B,)
            t = t_scalar.unsqueeze(1).expand(-1, T).contiguous().clone()      # (B, T)
            if d > 0:
                t[:, :d] = self.config.noise_s
            return t.clamp(min=eps, max=self.config.noise_s)
        # regime 3: Path B schedule. Two modes selected by streaming_schedule_mode
        # (config attr, default "v0"):
        #   v0 = canonical staircase (d in divisors of T only)
        #   v2 = piecewise-linear ramp (any integer d in [1, d_max])
        # Both apply quarter-gap symj jitter for training robustness; endpoints exact
        # half the time (when shift clamps), off by <= 0.25*gap otherwise.
        # See docs/path_b_schedule_viz.html sections 13 (v0) and 15 (v2).
        schedule_mode = getattr(self.config, "streaming_schedule_mode", "pir2")
        d_max = max(self.config.streaming_chunk_size_max, 1)

        if schedule_mode == "pir2":
            # v2: sample d from any integer in [1, d_max] s.t. T-2d >= 1 (ramp width).
            d_lo, d_hi = 1, min(d_max, (T - 1) // 2)
            if d_hi < d_lo:
                d_hi = d_lo
            d = int(torch.randint(d_lo, d_hi + 1, (1,)).item())
            D_ramp = T - 2 * d                                                       # ramp width
            L = d                                                                    # clean prefix
            p = torch.arange(T, device=device, dtype=dtype)
            ramp_val = self.config.noise_s * (1.0 - (p - L + 0.5) / D_ramp)
            tau_per_pos = torch.where(
                p < L, torch.tensor(self.config.noise_s, device=device, dtype=dtype),
                torch.where(p >= T - d, torch.tensor(0.0, device=device, dtype=dtype), ramp_val),
            )
            gap = self.config.noise_s / max(D_ramp, 1)
        else:
            raise NotImplementedError("v0 schedule is not supported anymore")
            # v0: sample d from divisors of T.
            valid_d = [d for d in range(1, min(d_max, T - 1) + 1) if T % d == 0]
            if not valid_d:
                valid_d = [1]
            d = int(valid_d[torch.randint(0, len(valid_d), (1,)).item()])
            M = T // d
            chunk_idx = (torch.arange(T, device=device) // d).to(dtype=dtype)
            if M > 1:
                tau_per_pos = self.config.noise_s * (1.0 - chunk_idx / (M - 1))
                gap = self.config.noise_s / (M - 1)
            else:
                tau_per_pos = torch.full((T,), self.config.noise_s, device=device, dtype=dtype)
                gap = self.config.noise_s

        # Quarter-gap symj jitter (both v0 and v2).
        delta = 0.25 * gap
        u = torch.rand(batch_size, 1, device=device, dtype=dtype)
        shift = (2.0 * u - 1.0) * delta                                              # (B, 1)
        tau = (tau_per_pos.unsqueeze(0) - shift).clamp(0, self.config.noise_s)       # (B, T)
        return tau.contiguous()

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        # Return a NEW BatchFeature instead of mutating ``backbone_output`` in
        # place. Callers that cache backbone_output (decoupled inference) would
        # otherwise see compounding vlln+self_attn applications across calls
        # because the dict slot was being overwritten with each pass's output.
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        return BatchFeature(data={**backbone_output, "backbone_features": backbone_features})

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if self.config.streaming:
            # Per-position τ: (B, T) → broadcast over action_dim for noising.
            t_per_pos = self.sample_time_per_position(
                actions.shape[0], device=actions.device, dtype=actions.dtype)
            t_b = t_per_pos.unsqueeze(-1)                                           # (B, T, 1)
            noisy_trajectory = (1 - t_b) * noise + t_b * actions
            velocity = actions - noise
            t_discretized = (t_per_pos * self.num_timestep_buckets).long()          # (B, T)
        else:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()         # (B,)
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Optional async-VLM delay embedding (added to action_features, broadcast over T).
        # At training, expect action_input.image_delay shape (B,) of long ticks ∈ [0, image_delay_max].
        if hasattr(self, "delay_embedding") and "image_delay" in action_input:
            d_idx = action_input["image_delay"].long().clamp_(0, self.config.image_delay_max)
            delay_emb = self.delay_embedding(d_idx)                              # (B, hidden)
            action_features = action_features + delay_emb.unsqueeze(1)            # broadcast over T

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        # DiT timestep: per-token in streaming mode (state slot τ=0, then per-position action τ).
        # In single-τ mode, pass the (B,) scalar as before.
        if self.config.streaming:
            state_t = torch.zeros(t_discretized.shape[0], 1,
                                  dtype=t_discretized.dtype, device=t_discretized.device)
            dit_timestep = torch.cat([state_t, t_discretized], dim=1)               # (B, 1+T)
        else:
            dit_timestep = t_discretized                                            # (B,)

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=dit_timestep,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=dit_timestep,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        # Clean-end mask: positions clamped at τ ≈ noise_s contribute uninformative
        # loss (target = actions - noise, model sees clean action → best output is 0).
        if self.config.streaming and self.config.streaming_mask_clean_end and "t_per_pos" in locals():
            eps_clean = self.config.noise_s / self.config.num_timestep_buckets
            keep = (t_per_pos < self.config.noise_s - eps_clean).to(action_mask.dtype).unsqueeze(-1)
            action_mask = action_mask * keep
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        force_nonstreaming: bool = False,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            force_nonstreaming: when True, skips the streaming-inference branch even on a
                streaming-trained checkpoint. Used by warm-start (seed_streaming_buffer)
                to produce a clean chunk via the model's flow-trained capability (sflow
                training mix includes the constant-τ regime).
        """
        vl_embeds = backbone_features

        # Streaming (per-position TEDI) inference: stateful rolling buffer across calls.
        # Falls back to the original lockstep loop when streaming=False.
        # Runtime override via options["force_nonstreaming"] — needed for RTC ckpts
        # which set config.streaming=True at training (just to enable per-position τ)
        # but want non-streaming flow inference + RTC inpaint at deploy.
        if options is not None and options.get("force_nonstreaming", False):
            force_nonstreaming = True
        if self.config.streaming and not force_nonstreaming:
            return self._streaming_inference(
                vl_embeds, state_features, embodiment_id, backbone_output, options)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        # Per-request override of denoising steps via options dict (for ODE-step ablations).
        n_steps = self.num_inference_timesteps
        if options is not None and "num_inference_timesteps" in options:
            n_steps = int(options["num_inference_timesteps"])
        dt = 1.0 / n_steps
        vel_strength = torch.ones_like(actions)

        # Train-time RTC deployment path: simpler than paper RTC.
        # Client supplies options["inpaint"] = (d, max_action_dim) clean actions
        # (already normalized + padded by _maybe_inject_inpaint server-side).
        # Front d positions = inpaint, hard-frozen (vel_strength=0) throughout the
        # substep loop. No exponential ramp. Matches train-time RTC training
        # distribution (front d at noise_s with clean actions, rest at scalar τ).
        if options is not None and "inpaint" in options:
            inp = torch.as_tensor(options["inpaint"], device=device, dtype=actions.dtype)
            if inp.ndim == 2:
                inp = inp.unsqueeze(0)                # (B, d, A)
            d = inp.shape[1]
            actions[:, :d, :] = inp[:, :d, :]
            vel_strength[:, :d, :] = 0.0
        elif "action" in action_input:
            raise NotImplementedError("Test time RTC is not supported anymore")
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # If train-time-RTC inpaint is on, the front d positions should be
        # encoded at τ=noise_s (= max bucket) while the rest evolve at scalar τ
        # — matches the sample_time_per_position regime 4 training distribution.
        # Without this, the encoder broadcasts the scalar t across all positions
        # and the model can't distinguish the clean front from the noisy back.
        inpaint_d = 0
        if options is not None and "inpaint" in options:
            inpaint_d = int(options.get("action_horizon",
                                        torch.as_tensor(options["inpaint"]).shape[-2]))

        # inpaint_schedule: "scalar" (Config 2 RTC-style: rest at flat scalar τ)
        # OR "staircase" (Config 3 chunk-wise-style: rest at staircase per sub-chunk).
        # Default scalar for backward compat / Config 2 RTC.
        inpaint_schedule = "scalar"
        if options is not None and "inpaint_schedule" in options:
            inpaint_schedule = str(options["inpaint_schedule"])

        T_pos = actions.shape[1]
        # Precompute staircase offset for rest of positions when staircase mode
        # is on. Sub-chunk length = inpaint_d. Sub-chunk k starts at τ = noise_s
        # * (K-k)/K where K = T/d. So position p in sub-chunk k = ceil(p/d) has
        # offset = (K-k)/K - 1 from noise_s (= -k/K).
        staircase_offset = None    # per-position bucket OFFSET from current global t_discretized
        if inpaint_schedule == "staircase" and inpaint_d > 0:
            # v0: assumes inpaint_d divides T_pos so K = T_pos/inpaint_d is integer (no partial SC).
            # tau* = noise_s * (M-1-chunk_idx)/(M-1) where M = K. So offset = (K-1-chunk_idx)/(K-1).
            # This matches training (regime 3), endpoints pinned at 0 (back) and noise_s (front).
            K = max(1, (T_pos + inpaint_d - 1) // inpaint_d)
            position_to_chunk = torch.arange(T_pos, device=device) // inpaint_d   # (T_pos,)
            position_to_chunk = position_to_chunk.clamp(max=K - 1)
            # Bucket offset: how many buckets ABOVE the current global t for each position.
            denom = max(K - 1, 1)                                                 # K=1 -> all clean
            staircase_offset = (
                (K - 1 - position_to_chunk).float() / denom * (self.num_timestep_buckets - 1)
            ).long()                                                              # (T_pos,)

        # Run denoising steps.
        for t in range(n_steps):
            t_cont = t / float(n_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            if inpaint_d > 0:
                # Per-position τ: front d at noise_s (last bucket), rest at global scalar
                # (or staircase offset, for Config 3 chunk-wise-style).
                # action_encoder takes (B, T); DiT takes (B, 1+T) with state_t=0 at pos 0.
                action_ts = torch.full(
                    size=(batch_size, T_pos), fill_value=t_discretized,
                    device=device, dtype=torch.long,
                )
                if staircase_offset is not None:
                    # Add per-position staircase offset (positions in deeper sub-chunks
                    # get earlier τ values).
                    action_ts = action_ts + staircase_offset.unsqueeze(0)
                action_ts[:, :inpaint_d] = self.num_timestep_buckets - 1
                action_ts = action_ts.clamp(0, self.num_timestep_buckets - 1)
                state_t = torch.zeros(batch_size, 1, dtype=action_ts.dtype, device=device)
                dit_ts = torch.cat([state_t, action_ts], dim=1)                     # (B, 1+T)
                timesteps_tensor = action_ts
            else:
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device,
                )
                dit_ts = timesteps_tensor
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Async-VLM delay embedding (broadcast over T, same pattern as positional embedding).
            if hasattr(self, "delay_embedding") and options is not None and "image_delay" in options:
                d = int(options["image_delay"])
                d = max(0, min(d, self.config.image_delay_max))
                d_t = torch.tensor([d], device=device, dtype=torch.long)
                delay_emb = self.delay_embedding(d_t).expand(batch_size, -1)        # (B, hidden)
                action_features = action_features + delay_emb.unsqueeze(1)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=dit_ts,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=dit_ts,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    def reset_streaming_buffer(self):
        """Reset the rolling buffer between episodes (streaming inference only)."""
        self._stream_buf = None
        self._stream_buf_t = None

    @torch.no_grad()
    def seed_streaming_buffer(self, clean_chunk: torch.Tensor, slide_steps: int | None = None):
        """Initialize the rolling streaming buffer at the on-manifold point for each
        position's init τ, using a clean chunk as the target. Schedule shape matches
        streaming_schedule_mode (pir2 ramp), mirroring training distribution
        and _streaming_inference cold-init (lines 718-737).
            buf_i = (1 - ratio_i) * noise_i + ratio_i * clean_chunk_i
        where ratio_i = t_init[i] / noise_s. Front d slots = clean, back d slots = noise.

        Args:
            clean_chunk: (B, T, action_dim) NORMALIZED + PADDED actions, in the same
                convention as _stream_buf and action_head outputs. Typically obtained
                from model.action_head.get_action(...)["action_pred"], or from
                state_action_processor.apply_action + manual zero-pad to action_dim.
                Do NOT pass raw joint angles — buf would be off-manifold and the next
                streaming call would emit garbage. Soft-checked below.
            slide_steps: (optional) sub-chunk size d for the cycle close shape. Falls
                back to streaming_num_chunks-derived chunk_size if None — that fallback
                is d=1 for ckpts with streaming_num_chunks=-1, which is a train-test
                mismatch for v2 ckpts deployed at d=5 or 10. Always pass explicitly
                in production.
        """
        assert clean_chunk.ndim == 3, (
            f"clean_chunk must be (B, T, A), got shape {tuple(clean_chunk.shape)}"
        )
        assert clean_chunk.shape[-1] == self.action_dim, (
            f"clean_chunk action_dim={clean_chunk.shape[-1]} != model.action_dim={self.action_dim}; "
            "did you forget to pad to max_action_dim? See processor.__call__ for the pad pattern."
        )
        assert clean_chunk.shape[1] == self.config.action_horizon, (
            f"clean_chunk T={clean_chunk.shape[1]} != model.action_horizon={self.config.action_horizon}"
        )
        # Soft sanity: normalized actions live in roughly [-3, 3]. Raw joint angles
        # are typically [-pi, pi] for arm + [0, 1.5] for gripper — both can exceed
        # this loose threshold. Warn (not fail) to avoid breaking legitimate edge cases.
        max_abs = float(clean_chunk.abs().max())
        if max_abs > 10.0:
            import warnings
            warnings.warn(
                f"seed_streaming_buffer: clean_chunk |max|={max_abs:.2f} > 10. This often "
                "indicates raw (unnormalized) actions. The buffer holds normalized values; "
                "pass model.action_head.get_action(...)[\"action_pred\"] or normalize via "
                "state_action_processor.apply_action first.",
                RuntimeWarning,
            )
        B, T, A = clean_chunk.shape
        device = clean_chunk.device
        dtype = clean_chunk.dtype
        if slide_steps is None:
            K = self.config.streaming_num_chunks if self.config.streaming_num_chunks > 0 else T
            slide_steps = max(T // max(K, 1), 1)
        slide_steps = max(1, min(int(slide_steps), T))

        schedule_mode = getattr(self.config, "streaming_schedule_mode", "pir2")
        if schedule_mode == "pir2":
            # v2 ramp: front L=d clean, middle ramp width D_ramp=T-2d, tail d pure noise.
            D_ramp = max(T - 2 * slide_steps, 1)
            L = slide_steps
            p = torch.arange(T, device=device, dtype=dtype)
            ramp_val = self.config.noise_s * (1.0 - (p - L + 0.5) / D_ramp)
            t_init = torch.where(
                p < L, torch.tensor(self.config.noise_s, device=device, dtype=dtype),
                torch.where(p >= T - slide_steps,
                            torch.tensor(0.0, device=device, dtype=dtype),
                            ramp_val),
            )
        else:
            # v0 staircase: M = T/d sub-chunks with gap = noise_s/(M-1), endpoints exact.
            M = max(T // slide_steps, 1)
            chunk_idx = (torch.arange(T, device=device) // slide_steps).clamp(max=M - 1).to(dtype)
            if M > 1:
                t_init = self.config.noise_s * (1.0 - chunk_idx / (M - 1))
            else:
                t_init = torch.full((T,), self.config.noise_s, device=device, dtype=dtype)
        ratio = (t_init / self.config.noise_s).unsqueeze(-1)             # (T, 1)
        noise = torch.randn_like(clean_chunk) # * 0.0
        self._stream_buf = (1.0 - ratio) * noise + ratio * clean_chunk
        self._stream_buf_t = t_init.unsqueeze(0).expand(B, -1).contiguous()

    @torch.no_grad()
    def _streaming_inference(
        self, vl_embeds, state_features, embodiment_id, backbone_output, options=None
    ) -> BatchFeature:
        """Mode 3 — rolling-buffer streaming inference. Per-position τ; emit cleanest-front
        each call, slide buffer, append fresh noise at back. Stateful across calls within
        an episode; call `reset_streaming_buffer()` between episodes.

        v0 Path B inpaint: client passes options['action'] = d clean actions just sent to
        robot during the previous inference window. Server clamps front d of post-slide
        buffer to those actions at τ=noise_s, so next call's front d sub-chunk is clean.
        Cycle close: M = T/d sub-chunks with gap = noise_s/(M-1), boundaries exact.
        See docs/path_b_schedule_viz.html section 13 for the schedule derivation."""
        B = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        T = self.config.action_horizon
        A = self.action_dim
        n_buckets = self.config.num_timestep_buckets
        n_steps = self.num_inference_timesteps
        if options is not None and "num_inference_timesteps" in options:
            n_steps = int(options["num_inference_timesteps"])
        dt = self.config.noise_s / n_steps

        # v0 sub-chunk size: client specifies via slide_steps (= deployed d). Defaults to
        # chunk_size derived from streaming_num_chunks for backward compat.
        K_chunks = self.config.streaming_num_chunks if self.config.streaming_num_chunks > 0 else T
        slide_steps = max(T // max(K_chunks, 1), 1)
        
        assert options is not None and "slide_steps" in options, "slide_steps must be provided"
        
        if options is not None and "slide_steps" in options:
            slide_steps = max(1, min(int(options["slide_steps"]), T))
        M = max(T // slide_steps, 1)                                                # num sub-chunks for v0

        # Lazy-init / re-init buffer on episode reset or batch-size change.
        buf = getattr(self, "_stream_buf", None)
        schedule_mode = getattr(self.config, "streaming_schedule_mode", "pir2")
        if buf is None or buf.shape[0] != B or buf.shape[1] != T:
            raise NotImplementedError("you should have it since you run seed_streaming_buffer before this call")
            # Cold init: schedule shape per streaming_schedule_mode (pir2 ramp).
            # Matches training (sample_time_per_position regime 3) at the deterministic mean.
            if schedule_mode == "pir2":
                # v2 ramp: clean prefix L=slide_steps, ramp width D_ramp=T-2*slide_steps,
                # noise suffix slide_steps. Endpoints exact (front=noise_s, back=0).
                D_ramp = max(T - 2 * slide_steps, 1)
                L = slide_steps
                p = torch.arange(T, device=device, dtype=dtype)
                ramp_val = self.config.noise_s * (1.0 - (p - L + 0.5) / D_ramp)
                t_init = torch.where(
                    p < L, torch.tensor(self.config.noise_s, device=device, dtype=dtype),
                    torch.where(p >= T - slide_steps,
                                torch.tensor(0.0, device=device, dtype=dtype),
                                ramp_val),
                )
            else:
                assert NotImplementedError("v0 schedule is not supported anymore")
                # v0 staircase.
                chunk_idx = (torch.arange(T, device=device) // slide_steps).clamp(max=M - 1).to(dtype)
                if M > 1:
                    t_init = self.config.noise_s * (1.0 - chunk_idx / (M - 1))
                else:
                    t_init = torch.full((T,), self.config.noise_s, device=device, dtype=dtype)
            t_init = t_init.unsqueeze(0).expand(B, -1).contiguous()                 # (B, T)
            noise = torch.randn(B, T, A, device=device, dtype=dtype)
            self._stream_buf = (1.0 - t_init.unsqueeze(-1) / self.config.noise_s) * noise
            self._stream_buf_t = t_init

        # v0 Path B inpaint applied POST-slide at end of call (see line ~795). The
        # cold-init path above seeds front d at τ=noise_s with action value=0 (no real
        # actions yet); cold-start callers should use seed_streaming_buffer(clean_chunk,
        # slide_steps=d) instead of relying on lazy init to seed actual robot state.

        # Two iteration modes:
        # 1) Default ("denoise-until-clean"): while-loop runs until min(t[front_chunk]) >=
        #    noise_s - dt/2. Variable iter count depending on buffer state.
        # 2) Override (options["num_inference_steps_per_call"] = K): for-loop runs exactly K
        #    iterations regardless of buffer state. Mirrors diffusion_policy's
        #    num_inference_steps_per_call semantic — useful for fixed-cost-per-call sweeps
        #    and for the "1 big step per emitted action" mode.
        #
        # v2 ramp REQUIRES the explicit substeps override. The default while-loop checks
        # buf_t[:, :slide_steps].min() < threshold, but for v2 the front d positions are
        # pinned at τ=noise_s with dt_per_pos[:, :d] = 0 (target == actual), so the loop
        # exits immediately with zero substeps. Default to K = nfe // (T/d - 1) so a v2
        # caller that forgets to set K still does roughly the matched-budget denoising.
        substeps_per_call = 1
        if options is not None and "num_inference_steps_per_call" in options:
            substeps_per_call = int(options["num_inference_steps_per_call"])
        # if schedule_mode == "pir2" and substeps_per_call is None:
        #     num_emits = max(T // max(slide_steps, 1) - 1, 1)
        #     substeps_per_call = max(1, n_steps // num_emits)

        # For v2: precompute per-position dt for cycle close. dt[p] = target[p] - actual[p].
        # target[p] = noise_s for emit positions (p < slide_steps), else target_full[p-slide_steps].
        # Split across substeps so total advance = target - actual.
        dt_per_pos = None
        if schedule_mode == "pir2":
            D_ramp = max(T - 2 * slide_steps, 1)
            L = slide_steps
            p_idx = torch.arange(T, device=device, dtype=dtype)
            ramp_val = self.config.noise_s * (1.0 - (p_idx - L + 0.5) / D_ramp)
            target_full = torch.where(
                p_idx < L, torch.tensor(self.config.noise_s, device=device, dtype=dtype),
                torch.where(p_idx >= T - slide_steps,
                            torch.tensor(0.0, device=device, dtype=dtype),
                            ramp_val),
            )
            target_per_p = torch.cat([
                torch.full((slide_steps,), self.config.noise_s, device=device, dtype=dtype),
                target_full[:T - slide_steps],
            ])                                                                       # (T,)
            n_substeps = max(substeps_per_call if substeps_per_call is not None else 1, 1)
            dt_total = (target_per_p.unsqueeze(0) - self._stream_buf_t).clamp(min=0) # (B, T)
            dt_per_pos = dt_total / n_substeps                                       # (B, T) per-substep

        def _one_denoise_step():
            t_disc = (self._stream_buf_t / self.config.noise_s * n_buckets).long().clamp(0, n_buckets - 1)
            action_features = self.action_encoder(self._stream_buf, t_disc, embodiment_id)
            # Async-VLM delay embedding (broadcast over T, same as positional embedding).
            if hasattr(self, "delay_embedding") and options is not None and "image_delay" in options:
                d = int(options["image_delay"])
                d = max(0, min(d, self.config.image_delay_max))
                d_t = torch.tensor([d], device=device, dtype=torch.long)
                delay_emb = self.delay_embedding(d_t).expand(B, -1)
                action_features = action_features + delay_emb.unsqueeze(1)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(T, dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            state_t = torch.zeros(B, 1, dtype=t_disc.dtype, device=device)
            dit_t = torch.cat([state_t, t_disc], dim=1)                                  # (B, 1+T)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds, timestep=dit_t,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds, timestep=dit_t)
            pred = self.action_decoder(model_output, embodiment_id)
            velocity = pred[:, -T:]
            if dt_per_pos is not None:
                # v2: per-position dt for cycle close. dt_per_pos already zero at clamped
                # positions (target == actual), so no separate mask needed.
                step = dt_per_pos.to(velocity.dtype)                                  # (B, T)
                self._stream_buf = self._stream_buf + step.unsqueeze(-1) * velocity
                self._stream_buf_t = self._stream_buf_t + step
            else:
                # v0: scalar dt + mask (clamped positions excluded).
                keep = (self._stream_buf_t < self.config.noise_s).to(velocity.dtype)  # (B, T)
                self._stream_buf = self._stream_buf + keep.unsqueeze(-1) * dt * velocity
                self._stream_buf_t = self._stream_buf_t + keep * dt

        if substeps_per_call is not None:
            for _ in range(substeps_per_call):
                _one_denoise_step()
        else:
            raise NotImplementedError("v0 schedule is not supported anymore")
            threshold = self.config.noise_s - dt * 0.5
            while self._stream_buf_t[:, :slide_steps].min() < threshold:
                _one_denoise_step()

        # Snapshot before sliding (return the full buffer; caller uses front n_action_steps).
        action_pred = self._stream_buf.clone()

        # FIFO: drop front slide_steps, append fresh noise at back at τ=0 (v0 pure noise).
        new_noise = torch.randn(B, slide_steps, A, device=device, dtype=dtype) #* 0.0
        new_t = torch.zeros(B, slide_steps, device=device, dtype=dtype)
        self._stream_buf = torch.cat([self._stream_buf[:, slide_steps:], new_noise], dim=1)
        self._stream_buf_t = torch.cat([self._stream_buf_t[:, slide_steps:], new_t], dim=1)

        # Latency-aware inpaint: client passes the actions actually sent to the robot
        # over the inference window via options["inpaint"]. Overwrite the front N slots so
        # next call's denoising conditions on them as clean (τ=noise_s). Matches the
        # training-time clean-front distribution (mask_clean_end + symj j-shift). Shares
        # the canonical inpaint key used by Path A (train-time RTC) — server routes
        # on force_nonstreaming to pick the flow loop vs this streaming branch.
        if options is not None and "inpaint" in options:
            raise NotImplementedError("inpaint is not supported here")
            prev = torch.as_tensor(options["inpaint"], device=device, dtype=dtype)
            if prev.ndim == 2:
                 g = prev.unsqueeze(0)  # (B, N, A)
            N = min(prev.shape[1], T)
            if N > 0:
                self._stream_buf[:, :N, :] = prev[:, :N, :]
                self._stream_buf_t[:, :N] = self.config.noise_s

        return BatchFeature(data={
            "action_pred": action_pred,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        })

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from pir2.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    def forward_vlm(self, inputs: dict) -> BatchFeature:
        """Run only the VLM backbone. Output is cacheable as long as
        (images, language) don't change. State/action fields are ignored."""
        backbone_inputs, _ = self.prepare_input(inputs)
        return self.backbone(backbone_inputs)

    def forward_dit(
        self,
        backbone_outputs: BatchFeature,
        inputs: dict,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """Run only the action head given a previously-computed backbone output.
        `options` carries RTC params (action, rtc_overlap_steps, ...) when used."""
        _, action_inputs = self.prepare_input(inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs, options)

    def forward_dit_action_only(
        self,
        backbone_outputs: BatchFeature,
        action_inputs_dict: dict,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """Fast-path variant of ``forward_dit`` that skips ``prepare_input`` —
        ``action_inputs_dict`` must already contain the fields the action head
        needs (``state``, ``embodiment_id``, optionally ``action`` for RTC),
        on the correct device and dtype.

        Used by the decoupled cached path when the client sends a state-only
        observation: the processor's ``process_state_only`` builds the dict
        directly, bypassing image transforms and tokenization.
        """
        # Promote to BatchFeature (action_head.get_action accesses .state etc.).
        # Move tensors to model device + cast floats to compute dtype.
        device = self.device
        dtype = self.dtype

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                if torch.is_floating_point(x):
                    return x.to(device=device, dtype=dtype)
                return x.to(device=device)
            return x

        prepared = {k: _to_device(v) for k, v in action_inputs_dict.items()}
        action_inputs = BatchFeature(data=prepared)
        return self.action_head.get_action(backbone_outputs, action_inputs, options)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
