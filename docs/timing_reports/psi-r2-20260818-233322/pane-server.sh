#!/usr/bin/env bash
# pane 0 -- Psi0 policy server
set -uo pipefail
cd "/home/jc/jc_workspace/humanoid/Psi0"
source .venv/bin/activate
export run_dir=".runs/finetune/simple/Efficient/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205"
export ckpt_step="40000"
echo "[server] $(date -Is) run_dir=$run_dir ckpt_step=$ckpt_step port=9000"
uv run --active --group psi-r2 --group serve serve_psi_r2 \
    --host 0.0.0.0 \
    --port 9000 \
    --run-dir="$run_dir" \
    --ckpt-step="$ckpt_step" \
    --action-exec-horizon=24 \
    --rtc 2>&1 | tee "/home/jc/jc_workspace/humanoid/Psi0/docs/timing_reports/psi-r2-20260818-233322/server.log"
echo "[server] exited with ${PIPESTATUS[0]} at $(date -Is)"
