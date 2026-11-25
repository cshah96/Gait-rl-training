# Source this file to activate the training environment:
#     source env.sh
# It activates the Python 3.9 virtualenv and makes sure sconepy can be found.

_SCONE_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ ! -f "$_SCONE_RL_ROOT/.venv/bin/activate" ]; then
    echo "[scone-rl] .venv not found - run ./setup.sh first" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$_SCONE_RL_ROOT/.venv/bin/activate"

# sconegym's sconetools.py looks in $SCONE_PATH/lib for sconepy*.so
export SCONE_PATH="${SCONE_PATH:-/opt/scone}"
export SCONE_RL_ROOT="$_SCONE_RL_ROOT"

echo "[scone-rl] python : $(command -v python) ($(python --version 2>&1))"
echo "[scone-rl] SCONE  : $SCONE_PATH"
