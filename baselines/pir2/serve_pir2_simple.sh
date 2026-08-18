#!/bin/bash
# Serve a trained PI-R2 checkpoint on psi's SIMPLE /act protocol.
#
# Usage:
#   bash baselines/pir2/serve_pir2_simple.sh <run_dir> <ckpt_step> [extra serve_pir2 args...]
#
# Env knobs: PORT, HOST, ACTION_EXEC_HORIZON, RTC (1/0), NFE, CUDA_VISIBLE_DEVICES.
#
# This is the shell wrapper; the equivalent one-liner is
#   uv run --active --group pir2 --group serve serve_pir2 --run-dir=... --ckpt-step=...

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "${VENV:-.venv-pir2}/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 RUN_DIR CKPT_STEP [extra args...]"
    exit 1
fi

RUN_DIR=$1
CKPT_STEP=$2
shift 2

rtc_args=()
if [ "${RTC:-1}" = "1" ]; then
    rtc_args=(--rtc)
fi

python src/pir2/deploy/pir2_serve_simple.py \
    --host="${HOST:-0.0.0.0}" \
    --port="${PORT:-9000}" \
    --run-dir="$RUN_DIR" \
    --ckpt-step="$CKPT_STEP" \
    --action-exec-horizon="${ACTION_EXEC_HORIZON:-24}" \
    --nfe="${NFE:-10}" \
    "${rtc_args[@]}" \
    "$@"
