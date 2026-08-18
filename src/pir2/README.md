# PI-R2 (πR²: Reactive Real-time Flow Policies)

Vendored copy of [pi-r2-flow](https://github.com/pi-r2-flow/pi-r2-flow)'s
`learning/Isaac-GR00T` (branch `pir2`) — a fork of NVIDIA Isaac-GR00T pinned at the
GR00T-N1.7 release, plus the πR² additions to the flow-matching action head.

Training and evaluation instructions live in
[`baselines/pir2/README.md`](../../baselines/pir2/README.md).

## What changed relative to upstream

- The Python package was renamed `gr00t` → `pir2`. Only module paths changed; class
  names (`Gr00tN1d7`, `Gr00tN1d7Config`, …) and the `model_type` string are
  untouched, so existing checkpoints load unchanged. Upstream file names are kept so
  the tree still diffs cleanly against `pi-r2-flow`.
- `PIR2_IMAGE_DELAY_MAX` is accepted alongside upstream's `GR00T_IMAGE_DELAY_MAX` for
  the async-VLM image-delay training augmentation.
- Added `configs/modality/simple_g1.py` — the modality config for Psi0's SIMPLE G1
  whole-body datasets, ported from `src/gr00t/gr00t/configs/modality/g1_locomanip.py`.
- Added `deploy/pir2_serve_simple.py` — a `serve_psi0`-shaped HTTP server
  (`--run-dir` / `--ckpt-step` / `--action-exec-horizon` / `--rtc`) that speaks psi's
  SIMPLE `/act` protocol, so the existing eval client needs no changes.

Upstream's own ZMQ server (`eval/run_gr00t_server.py`) and decoupled VLM/DiT policy
(`policy/decoupled_policy.py`) are kept as-is; the deployment stack in
`pi-r2-flow/deployment/` (xArm6 + XHand real-robot loop) is not vendored, since Psi0
evaluates through SIMPLE.

## What πR² adds to the action head

Two modifications, both in `model/gr00t_n1d7/gr00t_n1d7.py`:

1. **Proprioception-reactive diffusion forcing** — each chunk position carries its own
   noise level, so every denoising step can read fresher proprioception. Per-position
   AdaLN in the DiT, plus a learned `delay_embedding` indexed by integer image/VLM
   staleness so the policy tolerates asynchronously stale vision.
2. **Latency-adaptive flow schedule** (`streaming_schedule_mode="pir2"`) — a
   per-position staircase/ramp schedule parameterized by the inference delay `d`: a
   clamped-clean front of `d` in-flight actions, a ramped interior emitting `d` clean
   actions per denoising step, and a `d`-length noise tail. Randomizing `d` during
   training lets one checkpoint adapt to any measured `d`.

See `CHANGES.md` in the upstream repo for the file-by-file breakdown.
