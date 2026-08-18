## PI-R2 (πR²: Reactive Real-time Flow Policies)

Vendored from [pi-r2-flow](https://arxiv.org/abs/2607.26055) (`learning/Isaac-GR00T`
@ `pir2`), which is a fork of NVIDIA Isaac-GR00T at the GR00T-N1.7 release. The code
lives in [`src/pir2/`](../../src/pir2) with the upstream `gr00t` package renamed to
`pir2` so it can sit next to the existing GR00T-N1.6 baseline in `src/gr00t/`.

Three checkpoint variants share one training entrypoint and differ only in flags:

| Variant | What it is | Training flags |
|---|---|---|
| `plain_flow` | Standard GR00T flow matching | *(none)* |
| `rtc` | Train-Time RTC (clean-action inpainting of the front `d` slots) | `--streaming --streaming-rtc-weight 1.0 --streaming-rtc-d-max 10 --streaming-mask-clean-end` |
| `pir2` | πR²: proprio-reactive diffusion forcing + latency-adaptive schedule | `--streaming --streaming-schedule-mode pir2 --streaming-chunk-wise-weight 0.8 --streaming-constant-weight 0.2 --streaming-chunk-size-max 5 --streaming-mask-clean-end --image-delay-max 5 --image-delay-embed-dim 64` |


### Set up the environment

```bash
uv venv .venv-pir2 --python 3.10
source .venv-pir2/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --active --frozen
VIRTUAL_ENV=.venv-pir2 uv pip install -e .
VIRTUAL_ENV=.venv-pir2 uv pip install -r baselines/pir2/requirements-pir2.txt
```

`requirements-pir2.txt` carries the upstream-exact torch / transformers pins. If you
would rather stay inside the shared lockfile, use the `pir2` dependency group
instead — it holds those back to the same versions as the `psi` group:

```bash
uv sync --group pir2 --group serve --active
```

Download the GR00T-N1.7-3B base checkpoint once (or point `BASE_MODEL_PATH` at a
local copy):

```bash
hf download nvidia/GR00T-N1.7-3B --local-dir=$PSI_HOME/cache/checkpoints/GR00T-N1.7-3B
```


### Download task data

Standard Psi0 SIMPLE data — LeRobot format with `meta/modality.json`, the same
datasets the GR00T-N1.6 baseline trains on:

```bash
export task=G1WholebodyXMovePick-v0
hf download USC-PSI-Lab/psi-data simple/$task.zip --local-dir=$PSI_HOME/data --repo-type=dataset
unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```


### Train

```bash
# πR² (default)
bash baselines/pir2/train_pir2_simple.sh $task pir2

# baselines
bash baselines/pir2/train_pir2_simple.sh $task rtc
bash baselines/pir2/train_pir2_simple.sh $task plain_flow
```

Useful env knobs: `CUDA_VISIBLE_DEVICES`, `GLOBAL_BATCH_SIZE`, `MAX_STEPS`,
`LEARNING_RATE`, `SAVE_STEPS`, `OUTPUT_DIR`, `BASE_MODEL_PATH`, `DATASET_PATH`,
`PIR2_ACTION_HORIZON` (chunk length `T`, default 48), `USE_WANDB`. Anything after the
third positional argument is forwarded to `launch_finetune.py` verbatim.

Checkpoints land in `$OUTPUT_DIR/checkpoint-<step>/`, each self-contained (the
processor and `experiment_cfg/` are copied in on every save).


### Serve

```bash
export run_dir=.runs/finetune/pir2.xmove-pick.pir2.T48.b512.gpus8.2608171200
export ckpt_step=40000
uv run --active --group pir2 --group serve serve_pir2 \
  --host 0.0.0.0 \
  --port 9000 \
  --run-dir=$run_dir \
  --ckpt-step=$ckpt_step \
  --action-exec-horizon=24 \
  --rtc
```

or the wrapper, which does the same thing inside `.venv-pir2`:

```bash
bash baselines/pir2/serve_pir2_simple.sh $run_dir $ckpt_step
```

The server speaks the same `POST /act` protocol as `serve_psi0`, so an existing
SIMPLE eval client works unchanged.

What `--rtc` selects depends on the checkpoint (auto-detected from its config;
override with `--ckpt-type`):

- **`pir2`** → rolling-buffer streaming inference. At each episode start the buffer
  is warm-started with a full non-streaming flow chunk placed on the per-position τ
  manifold; every later request runs `--substeps-per-call` DiT substeps (1 by
  default, the πR² recipe) and slides the buffer `--action-exec-horizon` positions.
  `--action-exec-horizon` must divide the chunk length.
- **`rtc`** → the non-streaming flow loop with the front `d` positions clamped to
  the actions the client already committed to, where `d = T - action_exec_horizon`
  (override with `--rtc-inpaint-steps`). `d` must be ≤ the trained
  `streaming_rtc_d_max`.
- **`plain_flow`** → rejected. A plain-flow checkpoint never saw clean-action
  conditioning during training, so a reactive path would be out of distribution.

Dropping `--rtc` serves any variant through the plain flow loop, which is the
apples-to-apples baseline. Other flags: `--nfe` (flow denoising steps),
`--image-delay` (staleness in ticks, only for a checkpoint trained with
`--image-delay-max > 0`), `--device`, `--pad-action-dim`, `--ckpt-step=latest`.

Per-request timings are written to `$run_dir/inference_timing/*.jsonl` and are also
readable live at `GET /timing` — the same format `scripts/analyze_inference_timing.py`
consumes for `serve_psi0`.


### Train on one machine, serve on another

Two things are easy to get wrong here:

1. **The VLM backbone is not inside the checkpoint.** `Gr00tN1d7` rebuilds its
   Qwen3-VL backbone with `Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B")`
   and the processor rebuilds `Qwen3VLProcessor` the same way, then the finetuned
   weights are loaded over the top. So **both** the training and the serving machine
   need `nvidia/Cosmos-Reason2-2B` in their HF cache (~5 GB), not just the trainer.
2. **A training checkpoint is ~3x bigger than it needs to be for inference.** Training
   runs bf16-mixed-precision with fp32 master weights, so `checkpoint-<step>/` holds
   ~12 GB of weights plus ~24 GB of AdamW/scheduler/RNG state. `serve_pir2` reads none
   of the latter.

#### On the training machine

```bash
# 1. Code. Only these paths are new relative to a stock Psi0 checkout.
rsync -avh --relative \
  ./src/pir2 ./baselines/pir2 ./pyproject.toml ./uv.lock \
  gpubox:/data/psi/

# 2. Environment. A dedicated venv, so it can't disturb an existing psi0 env.
ssh gpubox 'cd /data/psi && uv venv .venv-pir2 --python 3.10 \
  && VIRTUAL_ENV=.venv-pir2 uv sync --active --group pir2 --group serve'

# 3. Weights (downloads on the GPU box; needs HF_HOME set, e.g. from .env)
ssh gpubox 'cd /data/psi && hf download nvidia/Cosmos-Reason2-2B \
  && hf download nvidia/GR00T-N1.7-3B --local-dir=$PSI_HOME/cache/checkpoints/GR00T-N1.7-3B'

# 4. Data: standard Psi0 SIMPLE LeRobot dataset
rsync -avh $DATA_HOME/simple/$task gpubox:/data/psi/hfm/data/simple/
```

Then train. `--save-only-model` cuts each checkpoint from ~36 GB to ~12 GB at the cost
of not being able to resume from it, which is usually the right trade for the final
run:

```bash
ssh gpubox
cd /data/psi
export task=G1WholebodyXMovePick-v0
export BASE_MODEL_PATH=$PSI_HOME/cache/checkpoints/GR00T-N1.7-3B
bash baselines/pir2/train_pir2_simple.sh $task pir2 xmove-pick --save-only-model
```

Training is resumable from `$OUTPUT_DIR` (the trainer picks up the newest
`checkpoint-<step>/`) as long as you did *not* pass `--save-only-model`.

#### Back to the inference machine

Copy only the inference slice — `config.json`, the safetensors, and the three
processor JSONs:

```bash
bash baselines/pir2/export_ckpt.sh \
  gpubox:/data/psi/.runs/finetune/pir2.xmove-pick.pir2.T48.b512.gpus8.2608171200 \
  40000 \
  .runs/finetune/pir2.xmove-pick
```

Make sure this machine also has `nvidia/Cosmos-Reason2-2B` cached, then serve. Nothing
else from the run dir is needed — `serve_pir2` re-reads the streaming / πR² flags out of
the checkpoint's own `config.json`, so `--ckpt-type` auto-detection still works on the
exported copy:

```bash
hf download nvidia/Cosmos-Reason2-2B   # once
uv run --active --group pir2 --group serve serve_pir2 \
  --run-dir=.runs/finetune/pir2.xmove-pick --ckpt-step=40000 \
  --action-exec-horizon=24 --rtc
```

If the serving machine is offline, set `HF_HUB_OFFLINE=1` so transformers resolves
`Cosmos-Reason2-2B` from the cache instead of trying to reach the hub.

`--action-exec-horizon` must divide the chunk length `T` the checkpoint was trained
with (`PIR2_ACTION_HORIZON`, default 48) on the πR² streaming path — 24, 16, 12, 8, 6,
4, 2 and 1 all work for `T=48`.


### Eval in SIMPLE

Identical to the other baselines — start the server above, then point SIMPLE's eval
entrypoint at `--host localhost --port 9000`. See
[baselines/dp/README.md](../dp/README.md#eval-in-simple) for the full command.
