#!/usr/bin/env bash
# Has the Psi0 policy learnt where the target is relative to the robot, or only the motion it was shown?
#
# Closed-loop trials of a SIMPLE Teleop task's released Psi0 checkpoint with the target object (the faucet, oven,
# door, ... or the object to pick) put around the robot, inside and outside the range the training demos covered: by
# default on a 3 cm grid, every grid point REPEATS (5) times, each repeat in another base scene. Everything else is
# re-drawn by SIMPLE's domain randomization at its highest replay level (distractors, table/floor/robot/object
# materials, lighting), seeded per trial. Output: $OUT/report/index.html, a self-contained page (top-down success map,
# success rates, every trial's video; refreshed every REPORT_EVERY s while the sweep runs) and $OUT/report/results.json
# (every trial's numbers). Once every trial has finished, the report folder is all OUT keeps.
#
# One command, four stages:
#   0. prepare  Psi0 venv:   download the checkpoint and the task's train/eval data if missing; read the training
#               range of the target pose from the training episodes; draw the trial plan (prepare_target_sweep.py)
#   1. serve    Psi0 venv:   serve_psi0_simple with the checkpoint (port $PORT, log server.log)
#   2. eval     SIMPLE venv: SIMPLE's Teleop eval loop (eval_decoupled_wbc, agent psi0_decoupled_wbc), unchanged,
#               driven by eval_target_pose.py: one episode per trial, CHUNK trials per Isaac process. Each trial is
#               written to results.jsonl as it finishes; a crash is resumed (given up after ATTEMPTS crashes in a row
#               without a finished trial; a dead policy server is restarted), and re-running the command continues.
#               The report is rebuilt every REPORT_EVERY s meanwhile
#   3. report   Psi0 venv:   report/ (index.html, results.json, web-encoded videos) (plot_target_pose.py); then, if
#               every trial has finished, the rest of OUT (raw videos, traces, logs, plan, results.jsonl, figures/) is
#               deleted
#
# Usage:  run_target_pose_sweep.sh [TASK] [TRIALS] [--tmux]
#   scripts/viz/target_pose_sweep/run_target_pose_sweep.sh                     # OpenFaucet, the whole grid
#   scripts/viz/target_pose_sweep/run_target_pose_sweep.sh --tmux              # same, in a detached tmux session
#   .../run_target_pose_sweep.sh G1WholebodyOpenOvenTeleop-v0 --tmux           # another task
#   .../run_target_pose_sweep.sh G1WholebodyOpenOvenTeleop-v0 3                # only the first 3 trials of the plan
#   GRID_STEP=0.05 REPEATS=3 .../run_target_pose_sweep.sh   # grid spacing (m) and trials per grid point
#   SAMPLING=random .../run_target_pose_sweep.sh 30         # 30 targets drawn at random instead (TRIALS default 10)
#   SAMPLING=random IN_DIST=50 .../run_target_pose_sweep.sh # 50% of them inside the training range, 50% outside
#   RANGE="0.2,0.5,-0.3,0.3" .../run_target_pose_sweep.sh   # dx_lo,dx_hi,dy_lo,dy_hi (m, robot frame) to test
#   FORWARD_SHIFT=0.05 SIDE_MARGIN=0.10 .../run_target_pose_sweep.sh   # the default range (see below)
#   YAW="-20,20" .../run_target_pose_sweep.sh                 # target yaw range (deg); default: the training range
#   KEEP=1 .../run_target_pose_sweep.sh         # keep the rest of OUT too (needed to rebuild the report, to add
#                                               # trials later, or to pool sweeps with plot_target_pose.py --merge)
#   SKIP_EVAL=1 OUT=<existing dir> .../run_target_pose_sweep.sh             # rebuild the report only (KEEP=1 runs)
#   REUSE_SERVER=1 .../run_target_pose_sweep.sh # a server for this checkpoint already answers on $PORT
#   OPEN=1 .../run_target_pose_sweep.sh         # open the report at the end
# Tasks with a released checkpoint: OpenFaucet, OpenOven, OpenTrashCan, CloseDoor, PushOfficeChair, BendPick, Handover,
#   BendHandover, LocomotionPickBetweenTables, PickAndPlaceAndHugContainer (G1Wholebody<Name>Teleop-v0).
# Positions are in the robot's start frame: dx forward, dy to the robot's left. The default range is the training
# range widened by its own depth (at least 10 cm) towards the robot and SIDE_MARGIN (10 cm) to each side, then moved
# FORWARD_SHIFT (5 cm) away from the robot, into the table. Grid points are GRID_STEP apart, centred in the range.
# The plan is fixed by the settings and SEED; delete OUT to start over (KEEP=1 runs: re-running continues).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

use_tmux=0
for a in "$@"; do
  case "$a" in
    --tmux) use_tmux=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $a" >&2; exit 1 ;;
    *[!0-9]*) TASK="$a" ;;
    *) TRIALS="$a" ;;
  esac
done

TASK="${TASK:-G1WholebodyOpenFaucetTeleop-v0}"
TASK="${TASK#simple/}"
TRIALS="${TRIALS:-0}"           # 0: grid = every trial of the plan, random = 10
SAMPLING="${SAMPLING:-grid}"    # grid | random
GRID_STEP="${GRID_STEP:-0.03}"  # m between grid points
REPEATS="${REPEATS:-5}"         # trials per grid point, each in another base scene with another seed
IN_DIST="${IN_DIST:-}"          # random sampling: percentage (or fraction) of trials inside the training range
RANGE="${RANGE:-}"
FORWARD_SHIFT="${FORWARD_SHIFT:-0.05}"  # m the default range is moved away from the robot, into the table
SIDE_MARGIN="${SIDE_MARGIN:-0.10}"      # m the default range reaches left and right of the training range
YAW="${YAW:-}"
TARGET="${TARGET:-auto}"        # auto | articulated | target
DR_LEVEL="${DR_LEVEL:-2}"       # eval pack the base scenes come from
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-}"          # empty = the task's released checkpoint (downloaded if missing)
CKPT_STEP="${CKPT_STEP:-40000}"
BUDGET_STEPS="${BUDGET_STEPS:-0}"  # 0 = the task's max_episode_steps
SAVE_VIDEO="${SAVE_VIDEO:-1}"
INSTRUCTION="${INSTRUCTION:-}"  # empty = the instruction SIMPLE's eval sends (the scene's language)
PORT="${PORT:-8015}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
RTC="${RTC:-1}"
ATTEMPTS="${ATTEMPTS:-3}"       # crashes in a row without a finished trial before giving up
CHUNK="${CHUNK:-40}"            # trials per Isaac process (0 = all in one)
REPORT_EVERY="${REPORT_EVERY:-1200}"  # s between report rebuilds while the sweep runs (0 = only at the end)
if [[ "$SAMPLING" == "grid" ]]; then
  mode="grid$(awk -v s="$GRID_STEP" 'BEGIN { printf "%g", s * 100 }')cm_x${REPEATS}"
else
  mode="uniform"; [[ -n "$IN_DIST" ]] && mode="in${IN_DIST}"
fi
[[ -n "$RANGE$YAW" || "$FORWARD_SHIFT" != "0.05" || "$SIDE_MARGIN" != "0.10" ]] && mode+="_custom"
OUT="${OUT:-$PSI0_DIR/.runs/target_pose_sweep/${TASK}_${mode}_seed${SEED}}"

if [[ "$use_tmux" == "1" ]]; then
  session="target_pose_$(date +%H%M%S)"
  # every setting passed explicitly, the unset ones unset: a tmux server keeps the environment it was started from
  # (e.g. an earlier run's IN_DIST), and its sessions would inherit it
  unset_args=() set_args=()
  for v in TASK TRIALS SAMPLING GRID_STEP REPEATS IN_DIST RANGE FORWARD_SHIFT SIDE_MARGIN YAW TARGET DR_LEVEL SEED \
           RUN_DIR CKPT_STEP BUDGET_STEPS SAVE_VIDEO INSTRUCTION PORT ACTION_EXEC_HORIZON RTC ATTEMPTS CHUNK \
           REPORT_EVERY OUT SKIP_EVAL KEEP REUSE_SERVER OPEN HEADLESS SIMPLE_DIR CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!v:-}" ]]; then set_args+=("$v=${!v}"); else unset_args+=(-u "$v"); fi
  done
  tmux new-session -d -s "$session" "env $(printf '%q ' "${unset_args[@]}" "${set_args[@]}")bash \
$(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "report will be: $OUT/report/index.html (rebuilt every $((REPORT_EVERY / 60)) min while it runs)"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "task: $TASK   output: $OUT"

health() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

SERVER_PID=
REPORT_PID=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  [[ -n "$REPORT_PID" ]] && kill "$REPORT_PID" 2>/dev/null || true
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
  echo "=== [0/3] prepare"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_target_sweep.py" --task "$TASK" \
      --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$OUT" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP" \
      --trials "$TRIALS" --sampling "$SAMPLING" --grid-step "$GRID_STEP" --repeats "$REPEATS" \
      --in-dist "$IN_DIST" --range "$RANGE" --forward-shift "$FORWARD_SHIFT" --side-margin "$SIDE_MARGIN" \
      --yaw "$YAW" --target "$TARGET" --dr-level "$DR_LEVEL" --seed "$SEED")
  run_dir="$(cat "$OUT/run_dir.txt")"

  start_server() {
    rtc_flag=""; [[ "$RTC" == "1" ]] && rtc_flag="--rtc"
    # shellcheck disable=SC2086
    (cd "$PSI0_DIR" && exec setsid .venv-psi/bin/serve_psi0_simple --policy psi0 --port "$PORT" \
        --run-dir "$run_dir" --ckpt-step "$CKPT_STEP" --action-exec-horizon "$ACTION_EXEC_HORIZON" $rtc_flag) \
      >> "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    waited=0
    until health; do
      kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited, see $OUT/server.log" >&2; tail -n 30 "$OUT/server.log" >&2; return 1; }
      (( waited >= 600 )) && { echo "policy server not up after 600 s, see $OUT/server.log" >&2; return 1; }
      sleep 5; waited=$((waited + 5))
    done
    echo "server up after ${waited} s"
  }

  echo "=== [1/3] policy server (log: $OUT/server.log)"
  if health; then
    [[ "${REUSE_SERVER:-0}" == "1" ]] || {
      echo "something already answers on port $PORT; stop it, pick another PORT, or set REUSE_SERVER=1" >&2; exit 1; }
    echo "reusing the server on port $PORT"
  else
    start_server || exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/info" > "$OUT/server_info.json" || true
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" \
    | sed 's/^/GPU memory used, free: /' || true

  n_trials="$(cat "$OUT/n_trials.txt")"
  n_done() { local n; n="$(grep -c . "$OUT/results.jsonl" 2>/dev/null)" || true; echo "${n:-0}"; }
  echo "=== [2/3] closed-loop eval: $(n_done)/$n_trials done (log: $OUT/eval.log; each Isaac start takes a few minutes)"
  if (( REPORT_EVERY > 0 )); then  # a fresh report every REPORT_EVERY s, so a long sweep can be looked at as it goes
    (trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM
     while true; do
       sleep "$REPORT_EVERY" & wait $!
       [[ "$OUT/results.jsonl" -nt "$OUT/report/index.html" ]] || continue
       (cd "$PSI0_DIR" && exec nice -n 10 .venv-psi/bin/python -u "$SCRIPT_DIR/plot_target_pose.py" --results "$OUT") \
         >> "$OUT/report.log" 2>&1 &
       wait $! && echo "[report] rebuilt at $(date +%H:%M) ($(n_done)/$n_trials done): $OUT/report/index.html" || true
     done) &
    REPORT_PID=$!
  fi
  eval_args=(--out "$OUT" --trials "$n_trials" --max-new "$CHUNK" --budget-steps "$BUDGET_STEPS" --host 127.0.0.1
             --port "$PORT")
  [[ "$SAVE_VIDEO" == "1" ]] && eval_args+=(--save-video) || eval_args+=(--no-save-video)
  [[ "${HEADLESS:-1}" == "1" ]] && eval_args+=(--headless) || eval_args+=(--no-headless)
  [[ -n "$INSTRUCTION" ]] && eval_args+=(--instruction "$INSTRUCTION")
  fails=0
  while (( $(n_done) < n_trials )); do
    before=$(n_done)
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" exec .venv/bin/python -u "$SCRIPT_DIR/eval_target_pose.py" \
        "${eval_args[@]}") >> "$OUT/eval.log" 2>&1 &
    pid=$!
    # Isaac prints a lot; show only the sweep's progress lines
    tail -n 0 -f "$OUT/eval.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^\[sweep\]|Traceback|Error:" || true
    rc=0; wait "$pid" || rc=$?
    if (( $(n_done) > before )); then fails=0; else fails=$((fails + 1)); fi
    if (( rc != 0 )); then
      echo "eval exited with $rc at $(date '+%F %H:%M'), see $OUT/eval.log" >&2
      tail -n 20 "$OUT/eval.log" >&2
    fi
    (( fails >= ATTEMPTS )) && { echo "no trial finished in $ATTEMPTS attempts in a row; giving up" >&2; break; }
    if ! health; then
      echo "the policy server is down, see $OUT/server.log" >&2
      [[ -n "$SERVER_PID" ]] || break  # not ours (REUSE_SERVER) to restart
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
      echo "restarting it"
      start_server || break
    fi
  done
  [[ -n "$REPORT_PID" ]] && { kill "$REPORT_PID" 2>/dev/null || true; wait "$REPORT_PID" 2>/dev/null || true; }
  REPORT_PID=
  (( $(n_done) >= n_trials )) || echo "WARNING: the sweep is incomplete ($(n_done)/$n_trials); plotting what finished (re-run to resume)" >&2
fi

echo "=== [3/3] report"
(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_target_pose.py" --results "$OUT")

# The report (with results.json) is self-contained; the rest is only needed to resume, so it goes once every trial is in
n_done="$(grep -c . "$OUT/results.jsonl" 2>/dev/null || true)"
n_trials="$(cat "$OUT/n_trials.txt" 2>/dev/null || echo 1000000)"
if [[ "${KEEP:-0}" != "1" && -f "$OUT/report/index.html" && -f "$OUT/report/results.json" ]] \
    && (( ${n_done:-0} >= n_trials )); then
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name report -exec rm -rf {} +
  echo "kept only the report ($(du -sh "$OUT/report" | cut -f1)); KEEP=1 keeps everything"
fi

echo
echo "report: $OUT/report/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/report/index.html" >/dev/null 2>&1 || true
fi
