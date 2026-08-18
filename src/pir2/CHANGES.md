# Changes vs. NVIDIA Isaac-GR00T

This repository is a fork of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),
pinned at upstream commit `4b1dca9` (GR00T-N1.7 release). The **`pir2`** branch adds the
modifications from **πR²: Reactive Real-time Flow Policies** on top of that base. The
branch diff (`4b1dca9..pir2`) is the complete change set (18 files); it is three logical
commits.

## Two modifications to the flow-matching action head

**Mod 1 — Proprioception-reactive diffusion forcing.**
Each chunk position gets its own noise level (diffusion forcing), so each denoising step
can read a fresher observation. The DiT uses per-position AdaLN (per-token timestep
embeddings), fresh proprioception conditions every denoising step, and a learned
`delay_embedding` indexed by the (integer) image/VLM staleness lets the policy tolerate
asynchronously-stale vision.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — per-position noising, `delay_embedding`, regime mixer
- `gr00t/model/modules/{dit,embodiment_conditioned_mlp,flowmatching_modules}.py` — accept per-position timestep embeddings
- `gr00t/data/types.py`, `gr00t/data/dataset/sharded_single_step_dataset.py` — image-delay training augmentation
- `gr00t/model/gr00t_n1d7/{processing_gr00t_n1d7,setup}.py` — plumbing for the above

**Mod 2 — Latency-adaptive flow schedule.**
A per-position "staircase" noise schedule parameterized by the inference delay `d` (a
diffusion-forcing generalization of Train-Time RTC): a clamped-clean front of `d` in-flight
actions, a ramped interior that emits `d` clean actions per denoising step, and a
`d`-length noise tail. Randomizing `d` at training lets one model adapt to any measured `d`.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — `streaming_schedule_mode="pir2"` schedule + rolling-buffer inference

**Asynchronous VLM/DiT decoupled inference.**
The slow vision/VLM features are cached and refreshed in a background loop while the DiT
action head runs each control tick on fresh proprioception.
- `gr00t/policy/decoupled_policy.py`, `gr00t/policy/server_client.py`, `gr00t/eval/run_gr00t_server.py`

## The three released variants (config-selected)

| Variant | Key config |
|---|---|
| **Plain flow** (standard GR00T) | `streaming=False` (default) |
| **Train-Time RTC** | `streaming=True`, `streaming_rtc_weight>0`, `streaming_rtc_d_max=d`, `streaming_constant_weight>0`, `streaming_mask_clean_end=True` |
| **πR²** | `streaming=True`, `streaming_schedule_mode="pir2"`, `streaming_chunk_wise_weight>0`, `streaming_constant_weight>0`; add `image_delay_max>0`, `image_delay_embed_dim>0` for the asynchronous VLM channel |

See the top-level README (`learning/`) for training and eval commands.

## Cleanup relative to the research fork
- Renamed the πR² schedule mode `"v2"` → `"pir2"` (the surface API value).
- Removed a stale duplicate action-head file and a hardcoded local path.
- Legacy/experimental schedule regimes remain guarded (they raise `NotImplementedError`);
  only the three variants above are supported.

## Infrastructure
- `gr00t/experiment/launch_finetune.py`, `gr00t/configs/{finetune_config,model/gr00t_n1d7}.py` — the streaming / image-delay knobs.
- `gr00t/experiment/experiment.py` — disable tf32 on pre-Ampere GPUs.
- `gr00t/utils/video_utils.py` — optional mmap frame cache.
- `scripts/time_gr00t_inference.py` — inference-latency benchmark.
