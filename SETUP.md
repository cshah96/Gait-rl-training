# Setup

How to get **scone-rl** running on a new machine, from a fresh clone up to training and viewing a
policy in SCONE Studio. For the full user guide see [README.md](README.md); for the gait-imitation
pipeline see [docs/IMITATION_TRAINING.md](docs/IMITATION_TRAINING.md).

## 1. Prerequisites

| What | Notes |
|---|---|
| Linux (tested on Ubuntu, x86-64) | |
| [SCONE](https://scone.software) installed in `/opt/scone` | Provides `sconepy` (Python 3.9 build) and SCONE Studio. Another location works: set `SCONE_PATH`. |
| Hyfydy licence activated in SCONE Studio | The H0918 / H1622 / H2190 models are Hyfydy models. Without a licence they will not load. |
| NVIDIA GPU + driver (optional) | Default PyTorch build is CUDA 12.8 (supports RTX 50xx). Without a GPU use `TORCH=cpu`. |
| `git`, `python3` (any 3.x, only used to bootstrap `uv`) | `setup.sh` downloads Python 3.9 / 3.11 itself via `uv`. |

## 2. Clone

```bash
mkdir -p ~/workspaces/projects/nms-simulation && cd ~/workspaces/projects/nms-simulation
git clone --recurse-submodules git@github.com:cshah96/Gait-rl-training.git scone-rl
cd scone-rl
git checkout dev
```

`third_party/depRL` is a git submodule (tgeijten/depRL, `sconegym` branch). If you cloned without
`--recurse-submodules`, run `git submodule update --init --recursive`.

sconegym is expected as a sibling clone `../sconegym`; if it is not there, `setup.sh` clones
tgeijten/sconegym into `third_party/sconegym` instead.

## 3. Install

```bash
./setup.sh                 # GPU (CUDA 12.8) PyTorch
# or: TORCH=cpu ./setup.sh
# or: SCONE_PATH=/path/to/scone SCONEGYM_DIR=/path/to/sconegym ./setup.sh
```

`setup.sh` is idempotent (safe to re-run). It:

1. installs `uv` into `.tools/`
2. creates `.venv/` with Python 3.9 (sconepy only ships a cp39 build)
3. installs PyTorch, `requirements.txt`, depRL and sconegym (editable)
4. makes the repo packages (`scone_ppo`, `scone_mimic`) importable from the venv
5. points SCONE Studio's "Sconegym python" preference at `bin/studio-python`, so double-clicking a
   checkpoint in Studio plays the policy
6. creates `results/` and `logs/`, links `~/SCONE/results/scone-rl -> ./results`
7. runs `scripts/check_install.py`

Every session afterwards:

```bash
source env.sh              # activates .venv and sets SCONE_PATH
python scripts/check_install.py
```

## 4. Quick test: walking policy

```bash
# PPO walking on the small model (default pipeline)
python scripts/ppo_train.py
# DEP-RL walking on H2190, 10-minute demo
./scripts/train_h2190_10min.sh
```

Results go to `results/ppo/...` and `results/deprl/...`; open them in SCONE Studio
(`~/SCONE/results/scone-rl/`) or see README sections 4-7.

## 5. Gait imitation data (Camargo / AddBiomechanics)

The derived reference files (`data/mimic/<subject>/*.npz`) are **not in git**, because they come from
an external dataset. Regenerate them:

1. Download the Camargo 2021 treadmill `.b3d` file for the subject from
   [AddBiomechanics](https://addbiomechanics.org) and put it in `../Data/`
   (e.g. `../Data/AB07_split5-001.b3d`).
2. Build the scaled model and reference motion:

   ```bash
   ./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07
   ```

   The first run creates `.venv-data/` (Python 3.11 + nimblephysics, used only to read `.b3d`).
   The output is `data/mimic/AB07/AB07_reference.npz`, plus the scaled model in
   `data/mimic/AB07/model/` (which is in git).

3. Smoke test, then train:

   ```bash
   python scripts/deprl_train.py configs/deprl/sconemimic_h2190_ab07_smoke.yaml
   ./scripts/train_mimic.sh            # 1e8 steps, detached
   ./scripts/train_mimic_kinetic.sh    # optional kinetic fine-tune from the finished run
   ```

4. Evaluate:

   ```bash
   python scripts/mimic_evaluate.py --experiment sconemimic_h2190_ab07 -n 5
   ```

## 6. What is not in git

| Path | Why | How to get it back |
|---|---|---|
| `.venv/`, `.venv-data/`, `.tools/` | environments | `./setup.sh`, `mimic_prepare_subject.sh` |
| `results/` | trained policies, rollouts (large binaries) | train, or copy from another machine |
| `logs/` | training logs | created by the training scripts |
| `data/mimic/*/*.npz` | derived from the Camargo dataset | `mimic_prepare_subject.sh` |
| `configs/.resolved/` | generated config copies | created at training time |

## 7. Troubleshooting

- **`ModuleNotFoundError: deprl`**: the venv is not active. Run `source env.sh`, or call
  `.venv/bin/python` directly.
- **`sconepy not found`**: SCONE is not in `/opt/scone`. Set `SCONE_PATH` and re-run `./setup.sh`.
- **Model fails to load / Hyfydy error**: activate the Hyfydy licence in SCONE Studio.
- **Segfault when reading a `.b3d`**: known nimblephysics teardown bug, handled in the repo scripts
  (see docs/IMITATION_TRAINING.md).
- More in README section 10.
