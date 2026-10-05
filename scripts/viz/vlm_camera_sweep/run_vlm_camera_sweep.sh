#!/usr/bin/env bash
# How does the Psi0 VLM latent react when the head camera moves?
#
# One command (no policy server and no SONIC controller needed):
#   0. check      the anchor frame is a training sample of the checkpoint: pixel-identical image and identical
#                 state in the pack the checkpoint was fine-tuned on (stops the run otherwise)    (<1 min)
#   experiment 1  camera sweep in the training scene: the head camera moved one DoF at a time around the
#                 anchor, plus the floor grid and the re-rendered episodes (figures 1-8)         (~12 min)
#   experiment 2  the same camera sweeps in a scene that is not in the training set (lighting +
#                 materials re-drawn, a table no training episode uses), figure 1b               (~6 min)
#   overview      <OUT>/index.html: the anchor check and both experiments on one page
# Each experiment runs three stages:
#   render   SIMPLE venv: reset the episode's recorded scene, put the robot at the anchor frame, re-render
#            the head camera under pose offsets
#   extract  Psi0 venv:   the served checkpoint's frozen VLM (exactly like /act) -> latents, and the action
#            expert's predicted chunk for each render
#   plot     Psi0 venv:   figures + summary.json + index.html
#
# Data: the checkpoint was fine-tuned on G1WholebodyXMoveBendCarryBoxSonic-v0 (50 episodes; TRAIN_PACK), which is
# G1WholebodyXMoveBendPickVariant1Teleop-v0 post-processed: same images and states, but only the source keeps
# each episode's scene (table material, lighting) in meta/episodes.jsonl. So DATA_DIR is the source and stage 0
# proves the anchor is the same sample in TRAIN_PACK. Both are downloaded from USC-PSI-Lab/psi-data on first use.
# .../G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 is the 5-episode *eval* pack (held out; fails stage 0).
#
# Usage:
#   scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh                 # defaults below
#   EPISODE=2 ANCHOR_FRAME=250 scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh   # another training frame
#   MODE=camera scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh     # move only the camera
#   EXPERIMENTS=1 .../run_vlm_camera_sweep.sh      # only the camera sweep in the training scene
#   SCENE2_SEED=7 .../run_vlm_camera_sweep.sh      # another second scene for experiment 2
#   RENDER_ARGS="--steps 25 --yaw-range -60 60" .../run_vlm_camera_sweep.sh
#   SKIP_RENDER=1 SKIP_EXTRACT=1 OUT=<existing dir> .../run_vlm_camera_sweep.sh   # re-plot only
#   CHECK_ANCHOR=0 DATA_DIR=<other dataset> .../run_vlm_camera_sweep.sh   # anchor outside the training set
#   .../run_vlm_camera_sweep.sh --tmux                                   # detached tmux session
#   OPEN=1 .../run_vlm_camera_sweep.sh                                   # open index.html at the end
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

ENV_ID="${ENV_ID:-simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
TRAIN_DATA="$SIMPLE_DIR/data/simple/G1WholebodyXMoveBendPickVariant1Teleop-v0/level-0"
TRAIN_DATA_URL="https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main/G1WholebodyXMoveBendPickVariant1Teleop-v0.zip"
TRAIN_PACK="${TRAIN_PACK:-$SIMPLE_DIR/data/psi-data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
TRAIN_PACK_URL="https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main/simple/G1WholebodyXMoveBendCarryBoxSonic-v0.zip"
DATA_DIR="${DATA_DIR:-$TRAIN_DATA}"
DATA_LABEL="${DATA_LABEL:-$([[ "$DATA_DIR" == "$TRAIN_DATA" ]] && echo "training set" || basename "$(dirname "$DATA_DIR")")}"
DATA_TAG="${DATA_TAG:-$([[ "$DATA_DIR" == "$TRAIN_DATA" ]] && echo train || echo data)}"
RUN_DIR="${RUN_DIR:-$PSI0_DIR/.runs/finetune/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223}"
CKPT_STEP="${CKPT_STEP:-40000}"
EPISODE="${EPISODE:-0}"
ANCHOR_FRAME="${ANCHOR_FRAME:-100}"  # training renders settle after ~10 frames; 100 is still the initial stand
MODE="${MODE:-rigid}"            # rigid: robot moves with the camera | camera: only the camera moves
SEED="${SEED:-0}"                # table-material draw, only used with DR_LEVEL=0
DR_LEVEL="${DR_LEVEL:--1}"       # -1: each episode's recorded scene (as trained on) | 0: re-draw the table (eval)
GI="${GI:-on}"                   # RTX indirect diffuse: on like the training renderer
TRAJ_EPISODES="${TRAJ_EPISODES:-0,1,2,3,4}"  # re-rendered episodes (probes); the anchor episode is always in
EXPERIMENTS="${EXPERIMENTS-1 2}"         # 1: camera sweep, training scene | 2: + scene not in the training set
CHECK_ANCHOR="${CHECK_ANCHOR:-1}"        # stage 0: stop unless the anchor is a training sample
SCENE2_SEED="${SCENE2_SEED:-100}"        # experiment 2's scene; bumped past every training table material
SCENE2_DR_LEVEL="${SCENE2_DR_LEVEL:-1}"  # 1: re-draw lighting + table/robot materials, layout kept
RENDER_ARGS="${RENDER_ARGS:-}"   # extra flags for render_camera_sweep.py (see its --help)
OUT="${OUT:-$PSI0_DIR/.runs/vlm_camera_sweep/${DATA_TAG}_ep${EPISODE}_f${ANCHOR_FRAME}_${MODE}}"

if [[ "${1:-}" == "--tmux" ]]; then
  session="vlm_sweep_$(date +%H%M%S)"
  env_vars=""
  for v in ENV_ID DATA_DIR DATA_LABEL DATA_TAG TRAIN_PACK RUN_DIR CKPT_STEP EPISODE ANCHOR_FRAME MODE SEED DR_LEVEL \
           GI TRAJ_EPISODES EXPERIMENTS CHECK_ANCHOR SCENE2_SEED SCENE2_DR_LEVEL RENDER_ARGS OUT \
           SKIP_RENDER SKIP_EXTRACT OPEN CUDA_VISIBLE_DEVICES; do
    [[ -n "${!v:-}" ]] && env_vars+="$v=$(printf '%q' "${!v}") "
  done
  tmux new-session -d -s "$session" \
    "env $env_vars bash $(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "results will be in: $OUT/index.html"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }
[[ -d "$RUN_DIR" ]] || { echo "checkpoint run dir missing: $RUN_DIR" >&2; exit 1; }
fetch() {  # <url> <unzip into> <what>
  echo "downloading $3 -> $2"
  local zip; zip="$(mktemp --suffix=.zip)"
  curl -fL --retry 3 -o "$zip" "$1" && mkdir -p "$2" && unzip -q -o "$zip" -d "$2"
  rm -f "$zip"
}
[[ -d "$DATA_DIR" || "$DATA_DIR" != "$TRAIN_DATA" ]] || fetch "$TRAIN_DATA_URL" "$SIMPLE_DIR/data/simple" "the training-set source (180 MB)"
[[ -d "$DATA_DIR" ]] || { echo "dataset missing: $DATA_DIR" >&2; exit 1; }
[[ "$CHECK_ANCHOR" != "1" || -d "$TRAIN_PACK" ]] || fetch "$TRAIN_PACK_URL" "$(dirname "$TRAIN_PACK")" "the training pack (178 MB)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "output: $OUT"

if [[ "$CHECK_ANCHOR" == "1" ]]; then
  echo "=== [0] is the anchor a training sample of the checkpoint?"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/check_anchor.py" --data-dir "$DATA_DIR" \
      --train-pack "$TRAIN_PACK" --run-dir "$RUN_DIR" --episode "$EPISODE" --frame "$ANCHOR_FRAME" --out "$OUT") \
    || { echo "stopping: episode $EPISODE frame $ANCHOR_FRAME is not a training sample ($OUT/anchor_check.json)" >&2; exit 1; }
else
  rm -f "$OUT/anchor_check.json"
fi

run_experiment() {  # <dir> <title> <scene2 seed> <extra render args>
  local dir="$1" title="$2" scene2="$3" extra="$4" pid
  mkdir -p "$dir"
  echo
  echo "##### $title -> $dir"
  if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
    echo "=== render (Isaac Sim, log: $dir/render.log)"
    rm -rf "$dir/img" "$dir/index.jsonl" "$dir/meta.json"
    # shellcheck disable=SC2086
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python -u "$SCRIPT_DIR/render_camera_sweep.py" \
        --env-id "$ENV_ID" --data-dir "$DATA_DIR" --out "$dir" --episode "$EPISODE" \
        --anchor-frame "$ANCHOR_FRAME" --mode "$MODE" --seed "$SEED" --dr-level "$DR_LEVEL" --gi "$GI" \
        --data-label "$DATA_LABEL" --traj-episodes "$TRAJ_EPISODES" \
        --scene2-seed "$scene2" --scene2-dr-level "$SCENE2_DR_LEVEL" $RENDER_ARGS $extra) \
      > "$dir/render.log" 2>&1 &
    pid=$!
    # Isaac prints a lot; show only this script's progress lines
    tail -n +1 -f "$dir/render.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^(anchor|sweep|grid|traj|scene2|wrote|WARNING:)|Traceback|Error:" || true
    wait "$pid" || { echo "render failed, see $dir/render.log" >&2; tail -n 30 "$dir/render.log" >&2; exit 1; }
    grep -q "^wrote " "$dir/render.log" || { echo "render did not finish, see $dir/render.log" >&2; exit 1; }
  fi
  if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
    echo "=== extract VLM latents (log: $dir/extract.log)"
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || echo 99999)
    if (( free_mb < 9000 )); then
      echo "WARNING: only ${free_mb} MiB GPU memory free; the model needs ~8 GB (is a policy server running?)" >&2
    fi
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/extract_sweep_latents.py" \
        --renders "$dir" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP") > "$dir/extract.log" 2>&1 &
    pid=$!
    tail -n +1 -f "$dir/extract.log" --pid "$pid" 2>/dev/null | grep --line-buffered -E "^\[|^saved|Traceback|Error" || true
    wait "$pid" || { echo "extraction failed, see $dir/extract.log" >&2; tail -n 30 "$dir/extract.log" >&2; exit 1; }
  fi
  echo "=== plot"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_sweep.py" --renders "$dir" --data-dir "$DATA_DIR")
}

for e in $EXPERIMENTS; do
  case "$e" in
    1) run_experiment "$OUT/exp1_camera_sweep" "experiment 1: camera sweep in the training scene" -1 "" ;;
    # the training-scene sweeps again (same anchor, the blue curves) + the same sweeps in scene 2; no grid/episodes
    2) run_experiment "$OUT/exp2_scene_change" "experiment 2: the same sweeps in a scene not in the training set" \
         "$SCENE2_SEED" "--grid-steps 0 --traj-episodes none" ;;
    *) echo "unknown experiment '$e' (EXPERIMENTS takes 1 and/or 2)" >&2; exit 1 ;;
  esac
done

(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/overview.py" --out "$OUT")
echo
echo "results: $OUT/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/index.html" >/dev/null 2>&1 || true
fi
