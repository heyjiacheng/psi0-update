# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoupled VLM/DiT inference policy.

Splits the monolithic ``Gr00tPolicy.get_action`` into two endpoints so the
deployment client can run the VLM backbone continuously in a background thread
while the DiT/action-head fires per control tick against the cached embedding.

Endpoints exposed (registered by ``PolicyServer`` when present):

- ``update_vlm_cache(observation)`` — runs only the VLM backbone, stores its
  output in a shared slot, returns ``{cache_id, t_obs_vlm, seq_len}``.
- ``get_action_chunk_cached(observation, options)`` — runs only the DiT head
  against the cached VLM features and fresh state. ``options`` carries RTC
  parameters when the client opts into chunk-boundary smoothing.

The baseline ``Gr00tPolicy`` and ``get_action`` endpoint are unchanged.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from pir2.data.types import MessageType

from .gr00t_policy import Gr00tPolicy, _rec_to_dtype


class DecoupledGr00tPolicy(Gr00tPolicy):
    """Gr00tPolicy with a cached VLM forward and decoupled DiT endpoint.

    A single VLM cache slot is shared across all callers. ``update_vlm_cache``
    overwrites it on each call; ``get_action_chunk_cached`` reads whatever was
    most recently posted. The cache is protected by a lock — the slot may be
    swapped at any time, but a DiT call already in flight always sees a
    consistent ``BatchFeature`` instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vl_cache_lock = threading.Lock()
        self._vl_cache: Any = None
        self._vl_cache_id: int = 0
        # t_image_capture in the CLIENT's clock domain — the wall-clock moment
        # the cached image was read from the camera, set verbatim from the
        # client-supplied timestamp. Server time.time() is used only as a
        # fallback when the client doesn't send one (legacy callers). Compare
        # against the SAME clock you sent in (client perf_counter or time.time).
        self._vl_cache_t_image_capture: float | None = None

    # ---- inpaint helper for sflow/RTC clean-action conditioning -------------

    def _normalize_pad_inpaint(
        self,
        raw_action_chunk: np.ndarray,
        observation: dict[str, Any],
    ) -> np.ndarray:
        """Normalize raw joint-space action chunk and pad to max_action_dim.

        Client sends `options["inpaint"]` as a (N, raw_dim) array of joint
        positions in robot units. The action head's streaming buffer holds
        normalized + dim-padded actions, so the inpaint write
        ``buf[:, :N, :] = options["inpaint"][:, :N, :]`` requires matching
        layout. This helper:
          1. Splits raw_action_chunk into per-modality dict via modality_configs
          2. Runs state_action_processor.apply_action (normalize + relative-
             conversion if any modality is RELATIVE)
          3. Concatenates normalized groups in modality_config order
          4. Right-pads trailing dims with zeros up to action_head.action_dim
        """
        action_cfg = self.modality_configs["action"]
        keys = action_cfg.modality_keys
        # Split raw chunk into per-modality dicts via norm_params dims (same
        # order the policy uses internally).
        norm_params = self.processor.state_action_processor.norm_params[
            self.embodiment_tag.value
        ]["action"]
        act_dict: dict[str, np.ndarray] = {}
        start = 0
        for k in keys:
            d = int(norm_params[k]["dim"].item())
            act_dict[k] = raw_action_chunk[:, start : start + d].astype(np.float32)
            start += d
        # State for any RELATIVE action conversion. Most embodiments use
        # ABSOLUTE; the helper passes state defensively so RELATIVE configs
        # work too.
        st_dict: dict[str, np.ndarray] = {}
        state_cfg = self.modality_configs["state"]
        state_keys = state_cfg.modality_keys
        for k in state_keys:
            arr = observation.get("state", {}).get(k)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3:  # (B, T, D)
                arr = arr[0, -1:]
            elif arr.ndim == 2:  # (T, D)
                arr = arr[-1:]
            st_dict[k] = arr
        normalized = self.processor.state_action_processor.apply_action(
            act_dict, self.embodiment_tag.value, state=st_dict if st_dict else None,
        )
        cat = np.concatenate([normalized[k] for k in keys], axis=-1)
        max_action_dim = self.model.action_head.action_dim
        pad_n = max_action_dim - cat.shape[-1]
        if pad_n > 0:
            cat = np.concatenate(
                [cat, np.zeros((cat.shape[0], pad_n), dtype=np.float32)], axis=-1
            )
        return cat.astype(np.float32)

    def _maybe_inject_inpaint(
        self,
        options: dict[str, Any] | None,
        observation: dict[str, Any],
    ) -> dict[str, Any] | None:
        """If options has 'inpaint' in raw joint space, normalize + pad it
        in-place so the action head's inpaint write matches the buffer layout.

        Idempotent: if the array is already wide enough (== max_action_dim),
        assumes it was preprocessed and skips. This lets callers that already
        pre-pad keep working. Paper RTC uses action_input["action"] (a
        different parameter) and does not go through this normalizer.
        """
        if options is None or "inpaint" not in options:
            return options
        max_action_dim = self.model.action_head.action_dim
        raw = np.asarray(options["inpaint"], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]  # (B, N, D) → (N, D)
        if raw.shape[-1] != max_action_dim:
            options["inpaint"] = self._normalize_pad_inpaint(raw, observation)
            options["action_horizon"] = int(raw.shape[0])
        return options

    # ---- shared helpers (steps 1-3 and step 5 of _get_action) ---------------

    @staticmethod
    def _is_state_only(observation: dict[str, Any]) -> bool:
        """True if the obs has only 'state' (no 'video' / 'language')."""
        return "video" not in observation and "language" not in observation

    def _collate_state_only(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Fast-path collate for the decoupled cached endpoint when the client
        sends a state-only observation (no video, no language). Bypasses image
        transforms + tokenization. Returns ``(action_inputs_dict, batched_states)``
        where action_inputs_dict is fed straight to ``forward_dit_action_only``.
        """
        # Unbatch: expect obs["state"] = {key: (B, T, D)} with B==1.
        state_batch = observation["state"]
        first_key = next(iter(state_batch))
        batch_size = state_batch[first_key].shape[0]

        per_sample_state_dicts = []
        for i in range(batch_size):
            per_sample_state_dicts.append(
                {k: v[i] for k, v in state_batch.items()}
            )

        # Process each sample through the lightweight state-only path.
        processed = [
            self.processor.process_state_only(s, self.embodiment_tag)
            for s in per_sample_state_dicts
        ]
        # Stack across batch dim.
        action_inputs_dict = {
            "state": torch.stack([p["state"] for p in processed], dim=0)
            .to(torch.bfloat16),
            "embodiment_id": torch.tensor(
                [p["embodiment_id"] for p in processed], dtype=torch.long
            ),
        }
        # Keep raw states for un-normalization in _decode_action.
        batched_states: dict[str, np.ndarray] = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack(
                [s[k] for s in per_sample_state_dicts], axis=0
            )
        return action_inputs_dict, batched_states

    def _collate_observation(
        self, observation: dict[str, Any]
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Steps 1-3 of ``Gr00tPolicy._get_action``: unbatch → VLA-process → collate.

        Returns ``(collated_inputs, batched_states)`` where ``batched_states`` is
        needed later by ``_decode_action`` to unnormalize the predicted action.
        """
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        
        batched_states: dict[str, np.ndarray] = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        return collated_inputs, batched_states

    def _decode_action(
        self, model_pred: Any, batched_states: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Step 5 of ``Gr00tPolicy._get_action``: unnormalize + cast to float32."""
        normalized_action = model_pred["action_pred"].float()
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        return {k: v.astype(np.float32) for k, v in unnormalized_action.items()}

    def _prime_vlm_cache(self, features: Any,
                         t_image_capture: float | None = None) -> tuple[int, float]:
        """Atomically install ``features`` as the latest VLM cache and return
        the (cache_id, t_image_capture) the caller should report.

        ``t_image_capture`` should be the CLIENT's wall-clock timestamp of when
        the image was captured (camera read time), so downstream staleness
        measurements stay in a single clock domain. Falls back to server
        ``time.time()`` only when omitted (legacy callers without timestamping)."""
        if t_image_capture is None:
            raise ValueError("t_image_capture must be provided")
            t_image_capture = time.time()
        with self._vl_cache_lock:
            self._vl_cache = features
            self._vl_cache_id += 1
            self._vl_cache_t_image_capture = t_image_capture
            return self._vl_cache_id, self._vl_cache_t_image_capture

    # ---- new endpoints -------------------------------------------------------

    def update_vlm_cache(
        self,
        observation: dict[str, Any],
        return_features: bool = False,
        t_image_capture: float | None = None,
    ) -> dict[str, Any]:
        """Run the VLM backbone on ``observation`` and store the result.

        State/action fields in the observation are ignored — only video and
        language affect the VLM output.

        Returns metadata so the client can track cache freshness:
          - ``cache_id`` — monotonic counter; clients can detect skipped updates
          - ``t_obs_vlm`` — wall-clock time the obs was received
          - ``seq_len`` — token sequence length of the VLM output

        When ``return_features=True``, the response also includes ``vl_embeds``:
        a dict of CPU ndarrays (one per tensor field of the BatchFeature). Used
        by the 2-server / 2-GPU deployment where the VLM server returns features
        to the client which then ships them to a separate DiT server.
        """
        collated_inputs, _ = self._collate_observation(observation)
        inputs = collated_inputs["inputs"]
        with torch.inference_mode():
            features = self.model.forward_vlm(inputs)
        feat_tensor = features.get("backbone_features")
        seq_len = int(feat_tensor.shape[1]) if feat_tensor is not None else -1
        cache_id, t_image = self._prime_vlm_cache(features, t_image_capture)
        result = {"cache_id": cache_id, "t_image_capture": t_image, "seq_len": seq_len}
        if return_features:
            # Numpy can't represent bfloat16, so float tensors get up-cast to
            # fp32 for transport. Bool / int tensors (attention_mask, image_mask)
            # MUST stay in their original dtype — the action head does mask
            # bit-ops (`mask & ...`) that aren't defined for bfloat16.
            def _t_to_npy(t: torch.Tensor) -> np.ndarray:
                if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
                    return t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
                return t.detach().to("cpu").contiguous().numpy()
            result["vl_embeds"] = {
                k: _t_to_npy(v) for k, v in features.items() if torch.is_tensor(v)
            }
        return result

    def seed_streaming_from_obs(
        self,
        observation: dict[str, Any],
        num_inference_timesteps: int | None = None,
        t_image_capture: float | None = None,
        slide_steps: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Warm-start the streaming buffer at episode start using flow bootstrap.

        1. Run forward_vlm on the obs (also primes the VLM cache).
        2. Run non-streaming flow on the action head (init=pure noise, scalar τ,
           full denoise with `num_inference_timesteps` substeps). This works on a
           sflow ckpt because sflow's training mix includes the constant-τ regime.
        3. Take the clean chunk and seed the streaming buffer on the τ-manifold:
              buf_i = (1 - τ_i/noise_s) * fresh_noise + (τ_i/noise_s) * clean_i
           where τ_i follows streaming_schedule_mode (pir2 ramp).
           Pass slide_steps=d to match deployment cycle close shape — otherwise
           the server falls back to streaming_num_chunks (= d=1 for -1 ckpts),
           which is a train-test mismatch for v2 ckpts deployed at d=5 or 10.

        After this, subsequent streaming inference calls with substeps=1 work
        from a training-distribution state, not the off-manifold cold init.

        Returns the decoded clean chunk for the client's home→first_action ramp.
        """
        nfe = int(num_inference_timesteps) if num_inference_timesteps is not None else 30
        collated_inputs, batched_states = self._collate_observation(observation)
        inputs = collated_inputs["inputs"]

        with torch.inference_mode():
            backbone_outputs = self.model.forward_vlm(inputs)
            # Prime the VLM cache so subsequent streaming calls find it warm.
            cache_id, _ = self._prime_vlm_cache(backbone_outputs, t_image_capture)

            # Defensive shallow-copy (mirrors get_action_chunk_cached pattern).
            backbone_for_call = BatchFeature(data={**backbone_outputs})
            _, action_inputs = self.model.prepare_input(inputs)

            # Non-streaming flow → clean target chunk → seed the streaming buffer.
            # force_nonstreaming=True routes through the full flow loop even on
            # an sflow ckpt (sflow training mix includes the constant-τ regime).
            model_pred = self.model.action_head.get_action(
                backbone_for_call, action_inputs,
                options={"num_inference_timesteps": nfe, "force_nonstreaming": True},
            )
            self.model.action_head.seed_streaming_buffer(
                model_pred["action_pred"], slide_steps=slide_steps,
            )

        action = self._decode_action(model_pred, batched_states)
        return action, {
            "cache_id": cache_id, "seeded": True, "nfe": nfe,
            "slide_steps": slide_steps,
        }

    def get_action_chunk_cached(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        vl_embeds: dict[str, np.ndarray] | None = None,
        t_state_capture: float | None = None,
        t_image_capture: float | None = None,
        period_ms: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Run the DiT head against cached VLM features and fresh state.

        If ``vl_embeds`` is provided (2-server / 2-GPU deployment), uses those
        client-supplied features instead of the local cache. The client owns
        cache freshness in this mode.

        Otherwise uses the server-local cache populated by ``update_vlm_cache``.
        On cold start (no cache yet) this seeds the cache from the current
        observation, so the first call costs a full forward; subsequent calls
        only run the DiT.

        State-only fast path: if ``observation`` has only the ``state`` field
        (no video, no language), processor.process_state_only is used and
        forward_dit_action_only bypasses prepare_input — saves ~3-4 ms of
        wasted image transforms per call. Requires either a populated
        local cache or a client-supplied vl_embeds (no cold-start support
        because cold-start needs a full obs to run forward_vlm).
        """
        state_only = self._is_state_only(observation)
        if state_only:
            # print("State-only fast path")
            action_inputs_dict, batched_states = self._collate_state_only(observation)
            inputs = None  # unused on fast path
        else:
            print("Full observation path")
            raise NotImplementedError("Full observation path shouldn't be used in cached mode")
            collated_inputs, batched_states = self._collate_observation(observation)
            inputs = collated_inputs["inputs"]

        if vl_embeds is not None:
            # Client-supplied features (2-server mode). Reconstruct BatchFeature
            # on this server's device + dtype, bypass local cache entirely.
            # Important: only cast FLOAT tensors to the model's compute dtype.
            # Bool / int tensors (attention_mask, image_mask) must keep their
            # original dtype — action_head uses bit-ops on them which are not
            # defined for bfloat16.
            device = self.model.device
            dtype = self.model.dtype

            def _npy_to_t(arr: np.ndarray) -> torch.Tensor:
                t = torch.from_numpy(arr).to(device=device)
                if t.dtype.is_floating_point:
                    t = t.to(dtype=dtype)
                return t

            backbone_outputs = BatchFeature({
                k: _npy_to_t(v) for k, v in vl_embeds.items()
            })
            cache_id = -1
            # Client-supplied features: client must also supply the matching
            # image-capture timestamp; -1.0 means "unknown" (legacy callers).
            t_image = t_image_capture if t_image_capture is not None else -1.0
        else:
            with self._vl_cache_lock:
                backbone_outputs = self._vl_cache
                cache_id = self._vl_cache_id
                t_image = self._vl_cache_t_image_capture

            if backbone_outputs is None:
                if state_only:
                    raise RuntimeError(
                        "Cold-start path requires a full observation (video + "
                        "language) to run forward_vlm; got state-only obs and "
                        "no cached vl_embeds. Call update_vlm_cache (with full "
                        "obs) at least once before sending state-only requests."
                    )
                # Cold start: seed the cache from this observation.
                with torch.inference_mode():
                    backbone_outputs = self.model.forward_vlm(inputs)
                cache_id, t_image = self._prime_vlm_cache(backbone_outputs, t_image_capture)

        # Defensive shallow-copy of the cached BatchFeature before passing it
        # downstream. As of the fix in gr00t_n1d7.py, process_backbone_output
        # returns a NEW BatchFeature instead of mutating its input, so this
        # copy is no longer load-bearing — kept as a safety net in case future
        # action-head code re-introduces in-place writes against the dict.
        backbone_outputs_for_call = BatchFeature(data={**backbone_outputs})

        options = self._maybe_inject_inpaint(options, observation)

        # Image-staleness measurement: how old is the cached image vs this
        # call's state? Always compute when timestamps are available — useful
        # diagnostic for tuning vlm-refresh rate even on models without
        # delay_embedding. Inject into options as image_delay ONLY when the
        # model has the embedding (else the option is a silent no-op anyway).
        image_delay_ticks = None
        image_delay_ms = None
        if (period_ms and period_ms > 0
                and t_state_capture is not None
                and t_image is not None and t_image > 0):
            image_delay_ms = max(0.0, (t_state_capture - t_image) * 1000.0)
            image_delay_ticks = int(round(image_delay_ms / period_ms))
            action_head = getattr(self.model, "action_head", None)
            if action_head is not None and hasattr(action_head, "delay_embedding"):
                max_d = getattr(action_head.config, "image_delay_max", 0)
                options = {**(options or {}),
                           "image_delay": max(0, min(image_delay_ticks, max_d))}
        with torch.inference_mode():
            if state_only:
                # Fast path: skip prepare_input's backbone half, take action
                # inputs dict directly to action_head.
                model_pred = self.model.forward_dit_action_only(
                    backbone_outputs_for_call, action_inputs_dict, options
                )
            else:
                model_pred = self.model.forward_dit(
                    backbone_outputs_for_call, inputs, options
                )
        action = self._decode_action(model_pred, batched_states)
        # Both timestamps are in the CLIENT's clock domain (verbatim from the
        # client-supplied values). Server time.time() is used only as a fallback
        # when the client doesn't send a timestamp.
        info = {
            "cache_id_used": cache_id,
            "t_image_capture": t_image,
            "t_state_capture": t_state_capture if t_state_capture is not None else time.time(),
            # Measured staleness diagnostics (None if period_ms or timestamps missing).
            "image_delay_ms": image_delay_ms,
            "image_delay_ticks": image_delay_ticks,
        }
        return action, info

    def reset_streaming_buffer(self) -> dict[str, Any]:
        """Reset the action head's rolling streaming buffer.

        Call between episodes when the loaded checkpoint was trained with
        ``streaming=True``. No-op on non-streaming checkpoints.
        """
        action_head = getattr(self.model, "action_head", None)
        if action_head is not None and hasattr(action_head, "reset_streaming_buffer"):
            action_head.reset_streaming_buffer()
            return {"ok": True}
        return {"ok": False, "reason": "action_head has no reset_streaming_buffer"}

    # ---- options pass-through fix (baseline drops options at line 408) ------

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Override of ``Gr00tPolicy._get_action`` that threads ``options`` through
        to ``Gr00tN1d7.get_action``, which the baseline drops. Same shape and
        semantics otherwise — kept here so users of the legacy ``get_action``
        endpoint on this subclass also get RTC support."""
        collated_inputs, batched_states = self._collate_observation(observation)
        options = self._maybe_inject_inpaint(options, observation)
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs, options=options)
        return self._decode_action(model_pred, batched_states), {}
