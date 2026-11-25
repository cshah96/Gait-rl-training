#!/usr/bin/env bash
# 10-minute DEP-RL training run on sconewalk_h2190-v1 (3-D, 90 muscles).
#   ./scripts/train_h2190_10min.sh            # 10 minutes
#   MINUTES=30 ./scripts/train_h2190_10min.sh # any length
# Result: results/deprl/sconewalk_h2190_v1_10min/<date>.<model>/checkpoints/step_*.pt
# Open in SCONE Studio: ~/SCONE/results/scone-rl/deprl/sconewalk_h2190_v1_10min/...
# Note: 10 min is far too short to learn walking (the full run is 5e8 steps, days);
# expect the model to fall within ~1 s. Resume later by re-running without --fresh.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null
mkdir -p logs
.venv/bin/python scripts/deprl_train.py configs/deprl/sconewalk_h2190_v1.yaml \
    --minutes "${MINUTES:-10}" --save-steps 2e5 --name sconewalk_h2190_v1_10min --fresh \
    2>&1 | tee logs/deprl_h2190_10min.log
