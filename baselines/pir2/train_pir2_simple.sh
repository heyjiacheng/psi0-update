#!/bin/bash
# Finetune PI-R2 (GR00T-N1.7 action head) on a Psi0 SIMPLE task.
#
# Usage:
#   bash baselines/pir2/train_pir2_simple.sh <task> [variant] [exp]
#
#   task     SIMPLE dataset name under $DATA_ROOT, e.g. G1WholebodyXMovePick-v0
#   variant  pir2 (default) | rtc | plain_flow
#   exp      run-name suffix; defaults to a slug of the task
#
# The three variants are the three released PI-R2 configurations; they share this
# one entrypoint and differ only in the streaming flags below.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# launch_finetune.py has no load_dotenv of its own, so HF_HOME / WANDB_API_KEY /
# DATA_HOME have to be in the environment before torchrun starts. Sourced first so
# the explicit settings below still win.
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
# .env ships CUDA_LAUNCH_BLOCKING=true for debugging; leaving it on serializes every
# kernel launch and costs most of the training throughput.
unset CUDA_LAUNCH_BLOCKING

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# shellcheck disable=SC1091
source "${VENV:-.venv-pir2}/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [variant: pir2|rtc|plain_flow] [exp]"
    echo "Example: $0 G1WholebodyXMovePick-v0 pir2"
    exit 1
fi

task="$1"
variant="${2:-pir2}"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
exp="${3:-$default_exp}"

DATA_ROOT="${DATA_ROOT:-${DATA_HOME:-/hfm/data}/simple}"
# The modality config reads the dataset's own meta/modality.json.
export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/$task}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"

# Chunk length T the flow head predicts. Read by the modality config; the deploy
# --action-exec-horizon must divide it for the pir2 streaming path.
export PIR2_ACTION_HORIZON="${PIR2_ACTION_HORIZON:-48}"

MAX_STEPS="${MAX_STEPS:-40000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
STAMP="$(date +%y%m%d%H%M)"
OUTPUT_DIR="${OUTPUT_DIR:-.runs/finetune/pir2.$exp.$variant.T$PIR2_ACTION_HORIZON.b$GLOBAL_BATCH_SIZE.gpus$NPROC_PER_NODE.$STAMP}"

case "$variant" in
  plain_flow)
    # Standard GR00T flow matching: no streaming, no per-position tau.
    variant_args=()
    ;;
  rtc)
    # Train-Time RTC: front d positions clamped clean, rest at one scalar tau.
    variant_args=(
      --streaming
      --streaming-constant-weight 0.0
      --streaming-chunk-wise-weight 0.0
      --streaming-rtc-weight 1.0
      --streaming-rtc-d-max 10
      --streaming-mask-clean-end
    )
    ;;
  pir2)
    # PI-R2: proprio-reactive diffusion forcing + latency-adaptive schedule, with
    # the async-VLM image-delay channel.
    variant_args=(
      --streaming
      --streaming-constant-weight 0.2
      --streaming-chunk-wise-weight 0.8
      --streaming-schedule-mode pir2
      --streaming-chunk-size-max 5
      --streaming-mask-clean-end
      --image-delay-max 5
      --image-delay-embed-dim 64
    )
    # Tells the dataloader to sample image staleness d ~ U[0, 5] per step.
    export PIR2_IMAGE_DELAY_MAX=5
    ;;
  *)
    echo "Unknown variant '$variant' (expected pir2, rtc, or plain_flow)"
    exit 1
    ;;
esac

wandb_args=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    wandb_args=(--use-wandb --wandb-project "${WANDB_PROJECT:-psi-pir2}")
fi

echo "Task:      $task"
echo "Variant:   $variant"
echo "Dataset:   $DATASET_PATH"
echo "Output:    $OUTPUT_DIR"
echo "Horizon T: $PIR2_ACTION_HORIZON"

# --experiment-name is deliberately left unset: the trainer then treats OUTPUT_DIR
# itself as the run dir (instead of nesting one level deeper), so OUTPUT_DIR is what
# you pass to serve_pir2 --run-dir.
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
    src/pir2/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path src/pir2/configs/modality/simple_g1.py \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus "$NPROC_PER_NODE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 5 \
    --output-dir "$OUTPUT_DIR" \
    "${wandb_args[@]}" \
    "${variant_args[@]}" \
    "${@:4}"

echo
echo "Done. Serve it with:"
echo "  bash baselines/pir2/serve_pir2_simple.sh $OUTPUT_DIR $MAX_STEPS"
