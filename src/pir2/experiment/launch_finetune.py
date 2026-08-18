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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from pir2.configs.base_config import get_default_config
from pir2.configs.finetune_config import FinetuneConfig
from pir2.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from pir2.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Derive action_horizon from the modality config's action delta_indices. Single
    # source of truth — no separate --action-horizon flag. Also bumps the processor's
    # max_action_horizon to match (gr00t/model/gr00t_n1d7/setup.py:167,197 ties them).
    from pir2.configs.data.embodiment_configs import MODALITY_CONFIGS
    _mc = MODALITY_CONFIGS.get(embodiment_tag)
    if _mc is not None and "action" in _mc and _mc["action"].delta_indices:
        _ah = len(_mc["action"].delta_indices)
        config.model.action_horizon = _ah
        if hasattr(config.model, "max_action_horizon"):
            config.model.max_action_horizon = _ah

    # Streaming flow (sflow) — defaults off; mirrors diffusion_policy streaming_flow_policy.
    config.model.streaming                    = ft_config.streaming
    config.model.streaming_num_chunks         = ft_config.streaming_num_chunks
    config.model.streaming_constant_weight    = ft_config.streaming_constant_weight
    config.model.streaming_linear_weight      = ft_config.streaming_linear_weight
    config.model.streaming_random_weight      = ft_config.streaming_random_weight
    config.model.streaming_chunk_wise_weight  = ft_config.streaming_chunk_wise_weight
    config.model.streaming_symmetric_j        = ft_config.streaming_symmetric_j
    config.model.streaming_j_range_max        = ft_config.streaming_j_range_max
    config.model.streaming_inference_j_shift  = ft_config.streaming_inference_j_shift
    config.model.streaming_mask_clean_end     = ft_config.streaming_mask_clean_end
    # Second-gen sflow features (variable chunk_size, train-time RTC, async-VLM image delay)
    config.model.streaming_chunk_size_max     = ft_config.streaming_chunk_size_max
    config.model.streaming_rtc_weight         = ft_config.streaming_rtc_weight
    config.model.streaming_rtc_d_max          = ft_config.streaming_rtc_d_max
    config.model.streaming_schedule_mode      = getattr(ft_config, "streaming_schedule_mode", "pir2")
    config.model.image_delay_max              = ft_config.image_delay_max
    config.model.image_delay_embed_dim        = ft_config.image_delay_embed_dim

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
