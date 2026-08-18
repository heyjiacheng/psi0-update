#!/usr/bin/env bash
# pane 1 -- SIMPLE eval client (waits for the server port, then runs the policy)
set -uo pipefail
cd "/home/jc/jc_workspace/humanoid/SIMPLE"

echo "[client] waiting up to 600s for the policy server on localhost:9000 ..."
deadline=$(( $(date +%s) + 600 ))
until (exec 3<>/dev/tcp/127.0.0.1/9000) 2>/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "[client] server never came up on port 9000 -- giving up"
    echo "timeout-waiting-for-server" > "/home/jc/jc_workspace/humanoid/Psi0/docs/timing_reports/psi-r2-20260818-233322/.client-done"
    exit 1
  fi
  sleep 5
done
echo "[client] server is up."

if [[ "1" == "1" ]]; then
  target="/home/jc/jc_workspace/humanoid/SIMPLE/data/evals/efficient"
  case "$target" in
    "/home/jc/jc_workspace/humanoid/SIMPLE"/data/evals/?*) rm -rf "$target" && echo "[client] removed $target" ;;
    *) echo "[client] refusing to rm outside SIMPLE/data/evals: $target"; exit 1 ;;
  esac
fi

source .venv/bin/activate
export entry="eval_decoupled_wbc.py" task="G1WholebodyXMovePickTeleop-v0" agent="psi0_decoupled_wbc" dr="level-0"
echo "[client] $(date -Is) entry=$entry task=$task agent=$agent dr=$dr"
MUJOCO_GL=egl python src/simple/cli/$entry \
    simple/$task \
    $agent \
    $dr \
    --host=localhost \
    --port=9000 \
    --sim-mode=mujoco_isaac \
    --headless \
    --data-format=lerobot \
    --data-dir=data/evals/simple-eval/$task/$dr \
    --eval-dir=data/evals/efficient 2>&1 | tee "/home/jc/jc_workspace/humanoid/Psi0/docs/timing_reports/psi-r2-20260818-233322/client.log"
rc=${PIPESTATUS[0]}
echo "[client] exited with $rc at $(date -Is)"
echo "$rc" > "/home/jc/jc_workspace/humanoid/Psi0/docs/timing_reports/psi-r2-20260818-233322/.client-done"
