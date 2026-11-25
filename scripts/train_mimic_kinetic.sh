#!/usr/bin/env bash
# KINETIC FINE-TUNE of the AB07 imitation policy: continues the finished 1e8-step run with a
# stronger joint-moment / ground-force / centre-of-pressure reward (configs/deprl/sconemimic_h2190_ab07_kinetic.yaml)
#
# First call: copies the source run (checkpoints + log.csv) to results/deprl/sconemimic_h2190_ab07_kinetic/<same run>
# and starts there, so depRL resumes at step 1e8 (step count and learning curve continue) and trains
# to 1.3e8 (+30M steps, ~5 h). The source run is not modified. Later calls just resume the fine-tune.
#
#   ./scripts/train_mimic_kinetic.sh                    # detached; log: logs/mimic_kinetic.log
#   ./scripts/train_mimic_kinetic.sh --steps 1.5e8       # longer
#   SRC=results/deprl/sconemimic_h2190_ab07/<run> ./scripts/train_mimic_kinetic.sh   # other source run
#   pkill -f "deprl_train.py configs/deprl/sconemimic_h2190_ab07_kinetic"   # stop (resumable)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null
CONFIG=configs/deprl/sconemimic_h2190_ab07_kinetic.yaml
SRC=${SRC:-$(ls -d results/deprl/sconemimic_h2190_ab07/*/ | sort | tail -1)}
SRC=${SRC%/}
DST_ROOT=results/deprl/sconemimic_h2190_ab07_kinetic
mkdir -p logs
if pgrep -f "deprl_train.py $CONFIG" >/dev/null; then
    echo "already running: $CONFIG (see logs/mimic_kinetic.log)"; exit 1
fi
if [ -z "$(ls -d $DST_ROOT/*/ 2>/dev/null)" ]; then
    CK=$(ls "$SRC"/checkpoints/step_*.pt | sort -V | tail -1)
    DST="$DST_ROOT/$(basename "$SRC")"
    echo "seeding $DST from $CK"
    mkdir -p "$DST/checkpoints"
    cp "$CK" "$DST/checkpoints/"
    for f in actor critic dual logger obs_norm time; do cp "$SRC/checkpoints/$f.pt" "$DST/checkpoints/"; done
    cp "$SRC/log.csv" "$DST/log.csv"
fi
# checkpoint every 1e7 (3 x 29 MB), no 6 GB replay buffer unless BUFFER=1
EXTRA=()
[ "${BUFFER:-0}" = "1" ] || EXTRA+=(--no-buffer)
setsid nohup .venv/bin/python scripts/deprl_train.py "$CONFIG" --steps 1.3e8 --save-steps 1e7 "${EXTRA[@]}" "$@" \
    >> logs/mimic_kinetic.log 2>&1 < /dev/null &
echo "started PID $!  - log: logs/mimic_kinetic.log  (~1,650 steps/s: +3e7 steps in ~5 h)"
