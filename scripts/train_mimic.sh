#!/usr/bin/env bash
# Long gait-IMITATION training (DEP-RL, H2190 scaled to the subject).
# Measured on this machine (2026-11-23, full config, 195 envs): ~1,425 env steps/s while learning
#   -> 1e8 steps ~ 19-20 h.  RAM ~12 GB extra, GPU ~1.4 GB.
# Starts detached (survives closing the terminal / logging out). Re-running RESUMES the newest run.
#
# Disk-light by default: a checkpoint every 2e7 steps (5 x 29 MB for 1e8) and NO replay-buffer save
# (the buffer is ~6 GB; without it a resumed run just refills it). BUFFER=1 saves the buffer too.
#
#   ./scripts/train_mimic.sh                                   # AB07 config, 1e8 steps
#   ./scripts/train_mimic.sh configs/deprl/sconemimic_h2190_ab07.yaml --steps 2e8
#   BUFFER=1 ./scripts/train_mimic.sh                          # also save the replay buffer
#   tail -f logs/mimic_train.log                               # progress
#   pkill -f "deprl_train.py configs/deprl/sconemimic"         # stop (resume later)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null
mkdir -p logs
CONFIG=${1:-configs/deprl/sconemimic_h2190_ab07.yaml}
shift || true
EXTRA=()
[ "${BUFFER:-0}" = "1" ] || EXTRA+=(--no-buffer)
if pgrep -f "deprl_train.py $CONFIG" >/dev/null; then
    echo "already running: $CONFIG (see logs/mimic_train.log)"; exit 1
fi
# later args override these defaults
setsid nohup .venv/bin/python scripts/deprl_train.py "$CONFIG" --steps 1e8 --save-steps 2e7 "${EXTRA[@]}" "$@" \
    >> logs/mimic_train.log 2>&1 < /dev/null &
echo "started PID $!  config $CONFIG  - log: logs/mimic_train.log"
echo "expected: ~1,400 steps/s -> 1e8 steps in ~20 h; checkpoints every 2e7 in results/deprl/sconemimic_h2190_ab07/"
