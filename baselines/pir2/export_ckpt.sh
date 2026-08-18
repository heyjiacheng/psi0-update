#!/bin/bash
# Copy the inference-only slice of a trained PI-R2 checkpoint.
#
# A training checkpoint also carries optimizer / scheduler / RNG state for resuming
# (roughly 3x the model weights). serve_pir2 never reads those, so this drops them.
#
# Usage:
#   bash baselines/pir2/export_ckpt.sh <src_run_dir> <ckpt_step> <dst_run_dir>
#
# <src_run_dir> may be a remote rsync path (user@host:/path/to/run_dir).
# Example:
#   bash baselines/pir2/export_ckpt.sh \
#       gpubox:/data/psi/.runs/finetune/pir2.xmove-pick.pir2.T48.b512.gpus8.2608171200 \
#       40000 \
#       .runs/finetune/pir2.xmove-pick

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 SRC_RUN_DIR CKPT_STEP DST_RUN_DIR"
    exit 1
fi

SRC_RUN_DIR="${1%/}"
CKPT_STEP=$2
DST_RUN_DIR="${3%/}"

mkdir -p "$DST_RUN_DIR/checkpoint-$CKPT_STEP"

# Everything serve_pir2 needs and nothing else:
#   config.json                     model architecture + the streaming/pir2 flags
#                                   serve_pir2 auto-detects the ckpt type from
#   model*.safetensors[.index.json] weights
#   processor_config.json           modality configs, image/state/action dims
#   statistics.json                 normalization stats
#   embodiment_id.json              embodiment tag -> id mapping
rsync -avh --progress \
    --include='config.json' \
    --include='model*.safetensors' \
    --include='model.safetensors.index.json' \
    --include='processor_config.json' \
    --include='statistics.json' \
    --include='embodiment_id.json' \
    --exclude='*' \
    "$SRC_RUN_DIR/checkpoint-$CKPT_STEP/" \
    "$DST_RUN_DIR/checkpoint-$CKPT_STEP/"

echo
echo "Exported to $DST_RUN_DIR/checkpoint-$CKPT_STEP"
du -sh "$DST_RUN_DIR/checkpoint-$CKPT_STEP"
echo
echo "Serve it with:"
echo "  uv run --active --group pir2 --group serve serve_pir2 \\"
echo "    --run-dir=$DST_RUN_DIR --ckpt-step=$CKPT_STEP --action-exec-horizon=24 --rtc"
