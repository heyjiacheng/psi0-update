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

# Finetune config used for single node post-training.
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `pir2/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-pir2-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-pir2-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- Streaming Flow (sflow) ---
    streaming: bool = False
    """If True, enable streaming (per-position TEDI) flow training. Mirrors diffusion_policy's
    streaming_flow_policy. Defaults below match the Leap sflow recipe."""

    streaming_num_chunks: int = -1
    """K for chunk-wise regime; -1 = action_horizon (chunk_size=1, per-position τ ladder).
    Set explicitly to a positive int for coarser chunking."""

    streaming_constant_weight: float = 0.2
    streaming_linear_weight: float = 0.0
    streaming_random_weight: float = 0.0
    streaming_chunk_wise_weight: float = 0.8
    """Regime mix (must sum to ~1). Default = 20% constant + 80% chunk-wise."""

    streaming_symmetric_j: bool = False
    """If True, chunk-wise random shift ranges over ±delta instead of [0, delta). Lets the
    front chunk see τ both above and below the natural baseline. Enables inference-time
    re-noising via streaming_inference_j_shift."""

    streaming_j_range_max: int | None = None
    """Override shift half-range in bucket units. None = noise_s/K_eff (one chunk band).
    Used to widen the chunk-wise shift band beyond the natural K-derived width."""

    streaming_inference_j_shift: int = 0
    """Inference-time τ shift applied at _streaming_inference lazy-init. Positive → init
    front-chunk τ lower (more noise) so model re-denoises against fresh VL each call.
    Only meaningful when trained with streaming_symmetric_j=True."""

    streaming_mask_clean_end: bool = False
    """If True, mask out per-position loss for positions clamped at τ ≈ noise_s (clean
    end). Target velocity = actions - noise is uninformative when input is already clean.
    Mirrors diffusion_policy's t=0 loss-mask, adapted for GR00T's opposite convention."""

    # --- Second-gen sflow features ---
    streaming_chunk_size_max: int = 1
    """Max chunk_size for variable-chunk sflow training. >1 enables per-batch chunk_size ~ U[1..max].
    The j-shift band auto-scales to 1 chunk τ when this is > 1 (= front chunk_size positions
    can be clean at most, matching inference's inpaint width)."""

    streaming_rtc_weight: float = 0.0
    """Weight on the train-time RTC regime in the weighted regime mix. Front d positions
    at τ=noise_s (clean clamp), rest at one scalar τ. Stack with mask_clean_end=True."""

    streaming_rtc_d_max: int = 0
    """Max d for train-time RTC regime. d ~ U[0, d_max] per sample."""

    streaming_schedule_mode: str = "pir2"
    """Path B schedule mode for streaming flow training/inference (regime 3).
    'v0' = canonical staircase (d in divisors of T, scalar dt inference).
    'v2' = piecewise-linear ramp (any d in [1, d_max], per-position dt inference).
    See docs/path_b_schedule_viz.html sections 13 and 15."""

    image_delay_max: int = 0
    """Max image staleness in control ticks for async-VLM training. Requires the modality
    config to expose multiple historical frames via video.delta_indices."""

    image_delay_embed_dim: int = 0
    """Embedding dim for the image delay (positional-embedding style, added to action_features).
    0 = no embedding (model infers from obs); >0 = learnable embedding zero-init."""
