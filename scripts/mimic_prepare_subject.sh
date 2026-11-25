#!/usr/bin/env bash
# Step 1 of the gait-imitation pipeline for one subject:
#   .b3d (AddBiomechanics treadmill data) -> scaled H2190 model + reference motion + Studio playback
#
#   ./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07          # 1.2 m/s
#   SPEED=1.0 ./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07
#   SCALING=height ./scripts/mimic_prepare_subject.sh ...                        # uniform scaling
set -euo pipefail
cd "$(dirname "$0")/.."
B3D=$1; SUBJECT=$2
[ -x .venv-data/bin/python ] || { .tools/bin/uv venv --python 3.11 .venv-data && .tools/bin/uv pip install --python .venv-data/bin/python nimblephysics numpy; }
.venv-data/bin/python -u scripts/mimic_extract_b3d.py "$B3D" --subject "$SUBJECT" --speed "${SPEED:-1.2}" | grep -v "^  skip"
source env.sh >/dev/null
.venv/bin/python scripts/mimic_build_subject.py --subject "$SUBJECT" --scaling "${SCALING:-segments}" 2>&1 | grep -v "^[0-9][0-9]:[0-9][0-9]:\|^Found sconepy"
