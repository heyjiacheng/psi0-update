# Psi-R2 inference integration

`psi_r2` keeps the Psi0 Qwen3-VL backbone and action expert, then ports the
inference-time parts of [PI-R2-Flow](https://github.com/pi-r2-flow/pi-r2-flow):

- a per-position, latency-adaptive flow schedule;
- a stateful rolling action buffer;
- a slow image/language channel whose Qwen features are cached;
- a fast action channel that combines the cached features with fresh robot state.

See the [visual architecture comparison](../../docs/psi_r2_architecture.html) for
side-by-side diagrams of original Psi0, original PI-R2, and this integration.

This package intentionally reuses the original `psi` configuration, transforms,
checkpoint layout, and request/response protocol. Nothing under `src/psi` is
modified. The import name uses an underscore because Python package identifiers
cannot contain `-`; the uv dependency group remains `psi-r2`.

## Run

```bash
export run_dir=<psi-or-psi-r2-run-directory>
export ckpt_step=40000

uv run --active --group psi-r2 --group serve serve_psi_r2 \
  --host 0.0.0.0 \
  --port 9000 \
  --run-dir="$run_dir" \
  --ckpt-step="$ckpt_step" \
  --action-exec-horizon=24 \
  --rtc
```

`serve_psi0` remains assigned to the original policy; use `serve_psi_r2` to make
policy selection explicit while retaining the same `/act` wire protocol. Existing
SIMPLE clients therefore need no request or response changes.

For the common Psi0 `Tp=30`, `Ta=24` configuration, the server derives PI-R2's
rolling width as `d=Tp-Ta=6`. One `/act` response is filled by four fast updates,
each contributing only its newly-clean `[d:2d]` region. This preserves the old
SIMPLE wire contract, but those four cycles necessarily use the same state sample;
clients that need PI-R2 reactivity at every six actions should use the split
endpoints described below. An explicit override is available as
`--pir2-slide-steps`, with these requirements:

```text
2 * d < Tp
Ta % d == 0
```

At an episode reset, the server runs ordinary scalar-time Psi0 flow to obtain a
clean bootstrap chunk. Both this full-flow path and each PI-R2 rolling update
default to Psi0's original **10 denoise evaluations**. The rolling update still
uses PI-R2's exact per-position start/target schedule; only the number of Euler
substeps differs from the released PI-R2 default of one. The server then seeds
the rolling buffer on the per-position flow manifold and starts PI-R2 updates.
Later `/act` calls immediately use the cached slow
features with the new proprioceptive state while a single latest-wins worker
refreshes Qwen features from the new image.

Advanced decoupled clients may call `POST /slow` to refresh image/language features
and then call `POST /fast` with fresh state. In RTC mode, each `/fast` call advances
exactly one PI-R2 cycle and returns `d` actions (`6` in the example), which is the
reference reactive cadence. The first `/fast` after `/slow` performs the required
full-flow warm start using that cache. `GET /health` reports cache identity/age,
slow-worker state, and the last fast-path timing.

The inherited SIMPLE message has no session identifier. This server therefore
supports one active client episode per process; all stateful endpoints are
serialized, and `history["reset"]` starts a new episode.

## Flow-time convention

PI-R2 uses `tau=0` for noise and `tau=noise_s` for clean actions. Psi0 uses
`sigma=1` for noise and `sigma=0` for clean actions, so the port uses:

```text
sigma = 1 - tau / noise_s
psi_timestep = sigma * train_diffusion_steps
```

For exact equivalence with PI-R2's released Euler update, Psi-R2 also converts
the action increment (Psi and PI-R2 predict opposite velocity directions):

```text
delta_x = noise_s * delta_sigma * velocity_psi
        = delta_tau * velocity_pi_r2
```

PI-R2 fixes its state token at `tau=0`; after the same conversion, Psi conditions
the observation/state sequence at `sigma=1` (timestep `train_diffusion_steps`)
throughout each fast update.

Diffusers' scalar scheduler cannot perform this rolling update. Psi-R2 therefore
uses a manual per-position Euler step and only uses the original scheduler for
the reset bootstrap.

## Inference-only caveat

The port adds no learned parameters, so a legacy Psi0 checkpoint loads with its
strict key layout. The supplied RTC checkpoint has seen its clean-prefix
per-position regime, but it was not trained on PI-R2's ramp schedule or stale image
features. It is suitable for integration and latency testing, but it is not
equivalent to a Psi-R2-finetuned checkpoint. Training/finetuning and learned
image-delay conditioning are intentionally left for the later phase.
