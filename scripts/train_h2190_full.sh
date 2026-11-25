#!/usr/bin/env bash
# DEP-RL training on sconewalk_h2190-v1 (3-D, 90 muscles), 1e8 steps (~16 h).
# (Uses the full config; the config itself sets 5e8 steps.)
# Starts detached (survives closing the terminal). Re-running RESUMES the newest run.
#   ./scripts/train_h2190_full.sh                      # start / resume
#   tail -f logs/deprl_h2190_full.log                   # watch progress
#   pkill -f "deprl_train.py configs/deprl/sconewalk_h2190_v1.yaml"   # stop (resume later)
# Checkpoints: results/deprl/sconewalk_h2190_v1/<date>.<model>/checkpoints/step_*.pt
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null
mkdir -p logs
# 1e8 steps total, checkpoint every 1e7 (~1.5 h; 27 MB each + the ~6 GB replay buffer, overwritten each save)
# add --no-buffer to skip saving the buffer; pass --steps 2e8 to continue further later
# override with e.g. ./scripts/train_h2190_full.sh --save-steps 5e6
setsid nohup .venv/bin/python scripts/deprl_train.py configs/deprl/sconewalk_h2190_v1.yaml \
    --steps 1e8 --save-steps 1e7 "$@" >> logs/deprl_h2190_full.log 2>&1 < /dev/null &
echo "started PID $!  - log: logs/deprl_h2190_full.log"
