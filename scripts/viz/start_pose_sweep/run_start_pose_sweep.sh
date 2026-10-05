#!/usr/bin/env bash
# Does the Psi0 policy still do the task when the robot starts somewhere else?
#
# Closed-loop success rate of the same checkpoint on the same task as the VLM camera sweep (G1: walk to the box,
# bend, pick it up, put it on the table), with the robot's start pose moved off the recorded one: back/forward,
# sideways and turned. Every demo starts from the same spot, and the policy gets no base pose (only the head
# camera and the joints), so this asks how its locomotion generalizes from what it sees at an unseen start.
#
# One command, three stages:
#   1. serve  Psi0 venv:   serve_psi0_sonic_http with the checkpoint (port $PORT, log server.log)
#   2. eval   SIMPLE venv: SIMPLE's scripts/run_eval_wbc.sh, i.e. the lockstep SONIC controller and SIMPLE's
#             closed-loop eval, unchanged, with eval_start_pose.py as its wrapper: every start offset x every
#             recorded episode. Each run is written to results.jsonl as it finishes; a crash is resumed
#             (up to ATTEMPTS times), and re-running the same command continues where it stopped
#   3. plot   Psi0 venv:   figures + summary.json + index.html
#
# Usage:
#   scripts/viz/start_pose_sweep/run_start_pose_sweep.sh                  # 12 start poses x 5 episodes
#   PRESET=grid .../run_start_pose_sweep.sh             # 3 x 5 floor grid around the start, heading kept
#   OFFSETS="0,0,0 -0.2,0,0 0,0.3,0" .../run_start_pose_sweep.sh   # your own dx(m),dy(m),dyaw(deg) list
#   EPISODES=0,1 REPEATS=3 .../run_start_pose_sweep.sh  # which recorded scenes, runs per (pose, scene)
#   SKIP_EVAL=1 OUT=<existing dir> .../run_start_pose_sweep.sh             # re-plot only
#   REUSE_SERVER=1 .../run_start_pose_sweep.sh          # a policy server is already up on $PORT
#   .../run_start_pose_sweep.sh --tmux                  # detached tmux session
#   OPEN=1 .../run_start_pose_sweep.sh                  # open index.html at the end
# Offsets: dx forward (toward the box), dy left, dyaw turn left, all from the recorded start.
# Delete OUT to start a sweep over; otherwise finished runs are kept and skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

ENV_ID="${ENV_ID:-simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
DATA_DIR="${DATA_DIR:-$SIMPLE_DIR/data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0}"
RUN_DIR="${RUN_DIR:-$PSI0_DIR/.runs/finetune/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223}"
CKPT_STEP="${CKPT_STEP:-40000}"
PRESET="${PRESET:-axes}"         # axes | grid | axes+grid  (see eval_start_pose.py)
OFFSETS="${OFFSETS:-}"           # overrides PRESET
EPISODES="${EPISODES:-all}"      # recorded episodes = scenes (table material, lighting); all = 0..4
REPEATS="${REPEATS:-1}"
BUDGET_STEPS="${BUDGET_STEPS:-0}"  # 0 = SIMPLE's budget, max(2 x recorded length, 1500) steps
DR_LEVEL="${DR_LEVEL:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
PORT="${PORT:-8014}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
RTC="${RTC:-1}"
ATTEMPTS="${ATTEMPTS:-3}"
name="$PRESET"; [[ -n "$OFFSETS" ]] && name="custom"
OUT="${OUT:-$PSI0_DIR/.runs/start_pose_sweep/${name}_ckpt${CKPT_STEP}}"

if [[ "${1:-}" == "--tmux" ]]; then
  session="start_pose_$(date +%H%M%S)"
  env_vars=""
  for v in ENV_ID DATA_DIR RUN_DIR CKPT_STEP PRESET OFFSETS EPISODES REPEATS BUDGET_STEPS DR_LEVEL SAVE_VIDEO \
           PORT ACTION_EXEC_HORIZON RTC ATTEMPTS OUT SKIP_EVAL REUSE_SERVER OPEN CUDA_VISIBLE_DEVICES; do
    [[ -n "${!v:-}" ]] && env_vars+="$v=$(printf '%q' "${!v}") "
  done
  tmux new-session -d -s "$session" \
    "env $env_vars bash $(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "results will be in: $OUT  (figures: $OUT/figures/index.html)"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }
[[ -d "$RUN_DIR/checkpoints/ckpt_$CKPT_STEP" ]] || { echo "checkpoint missing: $RUN_DIR/checkpoints/ckpt_$CKPT_STEP" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "dataset missing: $DATA_DIR" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "output: $OUT"

health() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

SERVER_PID=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping the policy server"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "=== [1/3] policy server (log: $OUT/server.log)"
  if health; then
    [[ "${REUSE_SERVER:-0}" == "1" ]] || {
      echo "something already answers on port $PORT; stop it, pick another PORT, or set REUSE_SERVER=1" >&2; exit 1; }
    echo "reusing the server on port $PORT"
    curl -sf "http://127.0.0.1:$PORT/info" > "$OUT/server_info.json" || true
  else
    rtc_flag=""; [[ "$RTC" == "1" ]] && rtc_flag="--rtc"
    # shellcheck disable=SC2086
    (cd "$PSI0_DIR" && exec setsid .venv-psi/bin/serve_psi0_sonic_http --policy psi0 --port "$PORT" \
        --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP" --action-exec-horizon "$ACTION_EXEC_HORIZON" $rtc_flag) \
      > "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    waited=0
    until health; do
      kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited, see $OUT/server.log" >&2; tail -n 30 "$OUT/server.log" >&2; exit 1; }
      (( waited >= 600 )) && { echo "policy server not up after 600 s, see $OUT/server.log" >&2; exit 1; }
      sleep 5; waited=$((waited + 5))
    done
    echo "server up after ${waited} s"
  fi
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" \
    | sed 's/^/GPU memory used, free: /' || true

  echo "=== [2/3] closed-loop eval (log: $OUT/eval.log, SONIC controller log: ${TMPDIR:-/tmp}/eval_wbc_logs/controller.log)"
  video_flag="--save-video"; [[ "$SAVE_VIDEO" == "1" ]] || video_flag="--no-save-video"
  eval_args=(--data-dir "$DATA_DIR" --out "$OUT" --episodes "$EPISODES" --preset "$PRESET" --offsets "$OFFSETS"
             --repeats "$REPEATS" --budget-steps "$BUDGET_STEPS" --dr-level "$DR_LEVEL" "$video_flag"
             --host 127.0.0.1 --port "$PORT")
  rc=1
  for attempt in $(seq 1 "$ATTEMPTS"); do
    (( attempt > 1 )) && echo "--- attempt $attempt/$ATTEMPTS (resuming)"
    (cd "$SIMPLE_DIR" && EVAL_WBC_WRAPPER="$SCRIPT_DIR/eval_start_pose.py" HEADLESS="${HEADLESS:-1}" \
        bash scripts/run_eval_wbc.sh "$ENV_ID" psi0 "${eval_args[@]}") >> "$OUT/eval.log" 2>&1 &
    pid=$!
    # Isaac prints a lot; show only the sweep's progress lines
    tail -n 0 -f "$OUT/eval.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^\[sweep\]|^controller ready|^controller (exited|did not)|Traceback|Error:" || true
    rc=0; wait "$pid" || rc=$?
    (( rc == 0 )) && break
    echo "eval exited with $rc, see $OUT/eval.log" >&2
    tail -n 20 "$OUT/eval.log" >&2
    health || { echo "the policy server is down too, see $OUT/server.log" >&2; break; }
  done
  (( rc == 0 )) || echo "WARNING: the sweep is incomplete; plotting what finished (re-run to resume)" >&2
fi

echo "=== [3/3] plot"
(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_start_pose.py" --results "$OUT")

echo
echo "figures: $OUT/figures"
ls -1 "$OUT/figures"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/figures/index.html" >/dev/null 2>&1 || true
fi
