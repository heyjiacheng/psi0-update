# How inference time is measured

Review document for the timing instrumentation in `psi0_serve_simple.py` + `psi0.py`.
Read the timeline, check the "what is NOT counted" section, and tell me if the
boundaries are where you want them.

## The one-request timeline

Time flows downward. `│` marks code that runs, `┌ ┐` marks a timer's start/stop.

```
SIMPLE client                          Psi0 server (port 9000)
─────────────                          ───────────────────────────────────────────────

POST /act  ──────────────────────────▶ uvicorn receives HTTP body, parses JSON
                                       │                          ⚠ NOT counted
                                       ▼
                                    ┌─ timing.reset(); request_t0 = perf_counter()   :103
                                    │
                                    │  RequestMessage.deserialize(payload)           :104
                                    │    b64 decode + rebuild image arrays
                                    │                          ⚠ counted only in server_total
                                    │
                                    │  states → normalize_state_func → GPU           :113-125
                                    │                          ⚠ counted only in server_total
                                    │
                                    │  current_time = monotonic()                    :144
                                    │  client_loop_gap_ms = current_time             :159
                                    │                        - last_serve_time
                                    │                          (rtc path only)
                                    │
   s                                │  t(Image.fromarray(img)) for each camera       :167
   e                                │    resize + center_crop, torchvision
   r                                │                          ⚠ counted only in server_total
   v                                │                          (evaluated as a call ARGUMENT,
   e                                │                           so it lands outside the model's
   r                                │                           own timers — see note below)
   _                                │
   t                                │  ══ model.predict_action[_with_training_rtc_flow] ══
   o                                │
   t                                │  ┌─ sync ─ vlm_preprocess ──────────────  psi0.py:1667/1786
   a                                │  │    apply_chat_template
   l                                │  │    process_vision_info
                                    │  │    vlm_processor(...) → tokens, pixel_values
                                    │  │    .to(device)  ← H2D copy
                                    │  │    torch.stack × 4
                                    │  └─ sync ────────────────────────────────
                                    │
                                    │  ┌─ sync ─ vlm_forward ─────────────────  psi0.py:1704/1823
                                    │  │    Qwen3-VL-2B forward (bf16 autocast)
                                    │  │      = vision encoder + LLM, one pass
                                    │  │    take hidden_states[-1], unsqueeze
                                    │  └─ sync ────────────────────────────────
                                    │
                                    │  ┌─ sync ─ action_expert ───────────────  psi0.py:1720/1839
                                    │  │    randn(B, Tp, Da); set_timesteps(10)
                                    │  │    ┌ loop ×10 (num_inference_steps) ┐
                                    │  │    │   action_header(...)           │
                                    │  │    │   noise_scheduler.step(...)    │
                                    │  │    └────────────────────────────────┘
                                    │  └─ sync ────────────────────────────────
                                    │
                                    │     action_expert_per_step =            psi0.py:1744/1876
                                    │       action_expert / 10   (derived, not measured)
                                    │
                                    │  ══════════════════════════════════════════════════
                                    │
                                    │  .cpu().numpy(), denormalize, [:Ta]            :177-180
                                    │                          ⚠ counted only in server_total
                                    │
                                    └─ server_total = perf_counter() - request_t0     :183
                                       flush() → append JSONL row                     :185

                                       last_serve_time = monotonic()                  :190
                                       ├──────────────────── start of the gap measured
                                       │                     by the NEXT request
                                       ResponseMessage(...).serialize()               :191
                                       │                          ⚠ NOT counted
◀───────────────────────────────────── JSONResponse
executes 24 actions (Ta)
…then the next POST /act
```

## What each number means

| Field | Measures | How |
|---|---|---|
| `vlm_preprocess` | Chat template, image patching, tokenization, H2D copy | measured, CUDA-synced |
| `vlm_forward` | One Qwen3-VL-2B forward producing `hidden_states[-1]` | measured, CUDA-synced |
| `action_expert` | All 10 flow-matching denoise steps over `action_header` | measured, CUDA-synced |
| `action_expert_per_step` | Cost of one denoise step | **derived** = `action_expert / 10` |
| `client_loop_gap_ms` | Previous inference end → this inference start | measured, `rtc` rows only |
| `server_total` | Deserialize → action ready to serialize | measured, wall clock |

The VLM runs **once** per request; the action expert runs **10 times** per request.
So `action_expert` vs `vlm_forward` is the fair "which component dominates" comparison,
and `action_expert_per_step` is the "how much does one extra denoise step cost" number.

## What is NOT counted — please check these

**1. HTTP + JSON layer is outside `server_total`.**
uvicorn/FastAPI receiving the POST body and JSON-parsing it happens before the
clock starts, and `ResponseMessage.serialize()` + base64-encoding the returned
actions happens after it stops. The inbound parse scales with your camera payload
(your SIMPLE agent sends `rgb_head_stereo_left`), so it is not necessarily small.
`server_total` is therefore *not* what the client experiences as request latency —
if you need that, time it on the client side around `client.query_action`.

**2. Image resize / center-crop is not inside `vlm_preprocess`.**
The transforms are written as a call argument:

```python
self.model.predict_action(observations=[[t(Image.fromarray(img)) for img in ...]], ...)
```

Python evaluates arguments *before* entering the function, so this CPU work runs
before `vlm_preprocess`'s timer starts. It is inside `server_total` but attributed
to no stage. If you want it in `vlm_preprocess`, say so — it's a small change.

**3. Everything unmeasured shows up as a gap.** Sanity check on any row:

```
server_total − (vlm_preprocess + vlm_forward + action_expert)
  = deserialize + state normalize + image transforms + D2H copy + denormalize
```

If that gap is large, the bottleneck is CPU-side data handling, not the model.
This gap is the main thing to eyeball first.

## Two things that could make the numbers misleading

**Syncing serializes the pipeline.** Each timer calls `cuda.synchronize()` before
starting and before stopping. That is what makes the three stages individually
attributable — otherwise async GPU launches would charge the wrong stage. The cost:
CPU/GPU overlap between stages is destroyed, so the instrumented run can be
slightly slower than an uninstrumented one. `server_total` is still honest about
the run that actually happened, but if you are quoting a headline latency number,
measure it once with `PSI_PROFILE=0` (all timers become no-ops) and compare.

**`action_expert_per_step` is an average, not a measurement.** It cannot show you
that step 1 is slower than steps 2–10 (kernel autotuning, cache warmup). If you need
the per-step distribution, the loop needs a timer inside it.

## Episode segmentation

- Row field `episode` increments when the client sends `{"reset": True}`, which
  `SIMPLE/src/simple/baselines/psi0_decoupled_wbc.py:76` does on the first query of
  each episode. So 10 experiments → `episode` 1…10.
- Row field `path`: `uncond` = episode's first step, runs `predict_action` with no RTC
  conditioning. `rtc` = all later steps, runs `predict_action_with_training_rtc_flow`.
  `plain` = server started without `--rtc`.
- **All summaries drop each episode's first request by default** (it's a different code
  path, and episode 1's first request also absorbs CUDA warmup). `--keep-first` in the
  analyzer includes them. Verify this is the exclusion you want — it is the one
  judgement call in the reporting.

## Assumption: one request at a time

Timings accumulate in module-level state in `psi/utils/timing.py`, keyed by nothing.
The endpoint is a `def` (not `async def`), so FastAPI runs it in a threadpool — if two
clients ever hit `/act` concurrently, their stage timings would interleave into one
corrupted row. With a single SIMPLE client issuing sequential requests this cannot
happen. Flagging it so it isn't a surprise later.

## Reading a row

```json
{"idx": 41, "episode": 3, "episode_start": false, "path": "rtc",
 "vlm_preprocess": 18.4, "vlm_forward": 31.1, "action_expert": 62.2,
 "action_expert_per_step": 6.2, "client_loop_gap_ms": 812.3, "server_total": 115.8}
```

Request 42 overall, 3rd episode, a normal RTC step. VLM 31ms, action expert 62ms
(6.2ms × 10 steps), 116ms total on the server, and 25ms unaccounted
(116 − 18.4 − 31.1 − 62.2) in deserialize/transforms/denormalize. The client spent
812ms executing the previous 24-action chunk, so inference at 116ms is comfortably
inside the real-time budget.

## Generating a report

`scripts/make_timing_report.py` turns a log into a standalone HTML page (the loop
diagram + the per-request composition) and appends one row to a cumulative
`history.csv` so experiments can be compared:

```bash
python3 scripts/make_timing_report.py --title "baseline · 10 denoise steps"
python3 scripts/make_timing_report.py --title "5 denoise steps" --steps 5
```

With no `--logs`, it picks the newest `<run_dir>/inference_timing/*.jsonl`. Every
run writes a new timestamped html into `docs/timing_reports/`; the history table
inside each page shows every run recorded so far, so the newest report always
carries the full comparison.
