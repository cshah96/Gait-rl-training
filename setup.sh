#!/usr/bin/env bash
# =============================================================================
# One-shot installer for the scone-rl training environment.
#
#   ./setup.sh                 # default: GPU (CUDA 12.8) PyTorch
#   TORCH=cpu ./setup.sh       # CPU-only PyTorch
#   SCONEGYM_DIR=/path/to/sconegym ./setup.sh
#
# What it does (idempotent, safe to re-run):
#   1. installs `uv` into ./.tools (from PyPI, using the system python3)
#   2. creates ./.venv with Python 3.9 (sconepy only ships a cp39 build)
#   3. installs PyTorch (cu128 build -> supports RTX 50xx / Blackwell)
#   4. installs depRL (tgeijten/depRL @ sconegym branch, git submodule) editable
#   5. installs sconegym (sibling clone ../sconegym, or clones it) editable
#   6. points SCONE Studio's "Sconegym python" preference at ./bin/studio-python
#      (runs .venv python; double-clicked PPO checkpoints -> ppo_evaluate.py,
#       depRL checkpoints -> deprl.play)
#   7. creates ./results/{ppo,deprl} + the link ~/SCONE/results/scone-rl -> ./results,
#      and runs the install check
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SCONE_PATH="${SCONE_PATH:-/opt/scone}"
SCONEGYM_DIR="${SCONEGYM_DIR:-$ROOT/../sconegym}"
TORCH="${TORCH:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- 0. sanity -------------------------------------------------------------
if ! ls "$SCONE_PATH"/lib/sconepy*.so >/dev/null 2>&1; then
    echo "sconepy not found in $SCONE_PATH/lib - install SCONE first (https://scone.software)" >&2
    exit 1
fi

# --- 1. uv -----------------------------------------------------------------
if [ ! -x .tools/bin/uv ]; then
    say "Installing uv into .tools/"
    python3 -m venv .tools
    .tools/bin/pip install -q uv
fi
UV=.tools/bin/uv

# --- 2. Python 3.9 venv ----------------------------------------------------
if [ ! -x .venv/bin/python ]; then
    say "Creating Python 3.9 virtualenv in .venv/"
    "$UV" venv --python 3.9 .venv
fi
PIP=("$UV" pip install --python .venv/bin/python)

# --- 3. PyTorch ------------------------------------------------------------
say "Installing PyTorch $TORCH_VERSION ($TORCH)"
"${PIP[@]}" "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH"

# --- 4. depRL --------------------------------------------------------------
say "Installing depRL (sconegym branch)"
if [ ! -f third_party/depRL/pyproject.toml ]; then
    if [ -d .git ]; then
        git submodule update --init --recursive
    else
        git clone -b sconegym https://github.com/tgeijten/depRL.git third_party/depRL
    fi
fi
"${PIP[@]}" -r requirements.txt
# --no-deps: depRL pins torch==2.3.1 which has no RTX 50xx kernels
"${PIP[@]}" --no-deps -e third_party/depRL

# --- 5. sconegym -----------------------------------------------------------
if [ ! -f "$SCONEGYM_DIR/setup.py" ]; then
    say "Cloning sconegym into third_party/sconegym"
    SCONEGYM_DIR="$ROOT/third_party/sconegym"
    [ -d "$SCONEGYM_DIR" ] || git clone https://github.com/tgeijten/sconegym.git "$SCONEGYM_DIR"
fi
say "Installing sconegym from $SCONEGYM_DIR"
"${PIP[@]}" --no-deps -e "$SCONEGYM_DIR"

# --- 5b. make the repo packages (scone_ppo, scone_mimic) importable everywhere,
#          including in depRL worker processes and SCONE Studio playback
echo "$ROOT" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/scone_rl_root.pth"

# --- 6. SCONE Studio -------------------------------------------------------
say "Configuring SCONE Studio to evaluate .pt checkpoints with bin/studio-python"
chmod +x bin/studio-python
.venv/bin/python scripts/configure_studio.py

# --- 7. results folder + check --------------------------------------------
# All output lives in ./results/ppo/ and ./results/deprl/. sconepy can only write
# into the SCONE results dir, so ~/SCONE/results/scone-rl is linked to ./results.
[ -L results ] && rm results          # older versions of this script made a symlink here
mkdir -p results/ppo results/deprl logs
(cd scripts && SCONE_PATH=$SCONE_PATH ../.venv/bin/python -c 'from _paths import ensure_scone_link; ensure_scone_link()' 2>&1 | grep -v "^[0-9:]* \|^Found" || true)

say "Running install check"
SCONE_PATH=$SCONE_PATH .venv/bin/python scripts/check_install.py

say "Done. Activate with:  source env.sh"
