# scone-rl: Reinforcement Learning for Musculoskeletal Walking (SCONE + Hyfydy)

This repository trains neural-network controllers that make **muscle-driven human models walk**. The controllers command individual muscles; they never set joint torques directly. Trained policies can be **watched in 3-D in SCONE Studio**.

It contains two training pipelines on the same simulator:

| Pipeline | Algorithm | Use it for | Status on this machine |
|---|---|---|---|
| **PPO** (default) | Proximal Policy Optimization, ~300 readable lines in `scone_ppo/` (formerly the `gait-training` repo) | The 2-D H0918 model (`sconewalk_h0918-v1`): fast, simple, proven | ✅ **walks**: a stable gait in under an hour (see `docs/VERIFICATION.md`) |
| **DEP-RL** | MPO + DEP exploration (`third_party/depRL`, the method used by the sconegym authors) | Harder models: 3-D H1622 / H2190 (22–90 muscles), running, delayed sensors | ✅ **H2190 (90 muscles) walks** after 1e8 steps (~16 h, `docs/VERIFICATION.md` §7) |
| **Gait imitation** | DEP-RL + tracking reward (`scone_mimic/`) | Make H2190, scaled to a real person, reproduce that person's measured kinematics + kinetics (Camargo / AddBiomechanics data) | ✅ AB07: 1e8-step run + kinetic fine-tune done (walks 7.5 s, tracks joint angles, moments, GRF, centre of pressure). **Guide: [`docs/IMITATION_TRAINING.md`](docs/IMITATION_TRAINING.md)** |

> **New machine?** Start with [`SETUP.md`](SETUP.md): prerequisites, clone, install, data preparation.

> **How does it all work?** [`Readme_process_explained.md`](Readme_process_explained.md) explains what is trained, the inputs and outputs, the reward, what happens in each cycle, what Hyfydy does, and the math of the muscle model, PPO, MPO and DEP.

---

## Table of contents

1. [Quick start: train a walking policy and watch it](#1-quick-start-train-a-walking-policy-and-watch-it)
2. [What is installed and why](#2-what-is-installed-and-why)
3. [Repository layout](#3-repository-layout)
4. [Where results are stored](#4-where-results-are-stored)
5. [PPO pipeline (default)](#5-ppo-pipeline-default)
6. [Viewing a policy in SCONE Studio](#6-viewing-a-policy-in-scone-studio)
7. [Playing back a saved policy from the terminal](#7-playing-back-a-saved-policy-from-the-terminal)
8. [DEP-RL pipeline (for complex models)](#8-dep-rl-pipeline-for-complex-models)
9. [Moving to more complex models](#9-moving-to-more-complex-models)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

---

## 1. Quick start: train a walking policy and watch it

```bash
cd ~/workspaces/projects/nms-simulation/scone-rl
./setup.sh                    # one-time install (idempotent, safe to re-run)
source env.sh                 # activate the Python 3.9 env (every new shell)
python scripts/check_install.py

# --- train (PPO, proven settings: 20M steps, under an hour) ---
tmux new -s train             # optional, so it survives closing the terminal
python scripts/ppo_train.py 2>&1 | tee logs/ppo_train.log

# --- check and watch it ---
python scripts/ppo_plot.py --latest                         # learning curves -> training.png
python scripts/ppo_evaluate.py --latest --episodes 3        # replay -> .sto files + gait stats
sconestudio                                                 # then see section 6
```

Trained policies are not in git (`results/` is ignored). On the original machine they are in `results/ppo/`: the run trained here on 2026-11-18 (`20261118_114942_const_1p2`) and the run imported from `gait-training` (`20261105_101425_const_1p2_full`). Both final checkpoints walk in 19 of 20 test episodes (`docs/VERIFICATION.md`). Rollouts are already generated, so you can watch one right away:

```
SCONE Studio → Optimization Results → scone-rl → ppo → 20261118_114942_const_1p2
             → run_checkpoint_19996672 → double-click 00000_9325.863.sto → Play
```

> **Use the final checkpoint (`latest.pt` / the largest `step_<N>.pt`), not `best.pt`.** `best.pt` marks the peak of a noisy 50-episode training average. In 20 test episodes it walked only 10/20 (this run) and 16/20 (old run), while the final checkpoints walked 19/20 in both runs.

---

## 2. What is installed and why

### 2.1 System prerequisites (already present on this machine)

| Item | Version | Notes |
|---|---|---|
| SCONE | 2.4.5 at `/opt/scone` | `sconestudio`, `sconecmd`, `lib/sconepy.cpython-39-x86_64-linux-gnu.so` |
| Hyfydy | 1.13.2 | trial license in `~/.config/SCONE/scone-settings.zml`. **Trial expires 2026-12-04.** |
| GPU / CPU | RTX 5080 (16 GB), 16 logical cores, 46 GB RAM | PPO runs on CPU; DEP-RL uses the GPU |

### 2.2 The Python 3.9 environment (`.venv/`)

`sconepy` (SCONE's Python API) ships **only as a CPython 3.9 module**. `setup.sh` therefore:
1. creates `.tools/`, a tiny venv on the system Python that only holds [`uv`](https://github.com/astral-sh/uv), installed from PyPI;
2. uses `uv` to create **`.venv/`** with a standalone **Python 3.9.25**.

`env.sh` activates it and sets `SCONE_PATH=/opt/scone`. sconegym's `sconetools.py` uses that variable to find `sconepy` at runtime.

> **Why not the shared `~/workspaces/.venv`?** It is Python 3.12 (numpy 2.5, pandas 3, torch 2.13). sconepy cannot be imported there, because its only build is cp39. depRL also needs Python ≤ 3.11.5, `gym==0.13` and numpy < 2, which would break other projects. So the SCONE env lives in this repo, and the shared env is untouched. (The old Windows `gait-training` setup used a separate `workspaces\.venv-scone` for the same reason.)

| Package | Version | Why |
|---|---|---|
| `torch` | **2.7.1+cu128** | depRL pins 2.3.1, which has no RTX 50xx (sm_120) kernels. 2.7.1 is the newest torch with Python 3.9 wheels and sm_120 support. |
| `deprl` | 0.4.0, editable, `third_party/depRL` | the `sconegym` branch of `tgeijten/depRL` (what the SCONE docs require). Git submodule pinned at `8f5e509`, installed `--no-deps` so torch is not downgraded. |
| `sconegym` | 0.2.0, editable, `../sconegym` | your clone (environments + reward). `--no-deps`. |
| `gym` 0.13.0, `numpy` 1.26.4 | | old gym API used by sconegym; numpy < 2 |
| `pandas`, `pyyaml`, `termcolor`, `gdown`, `matplotlib` | | logging, configs, plots |

`scone_ppo/` (PPO) needs nothing beyond `torch` + `numpy` + sconegym.

### 2.3 SCONE Studio integration

When you double-click a `step_<N>.pt` checkpoint, SCONE Studio runs `<python> -m deprl.play --checkpoint_file <file> --num_episodes <n>`. `<python>` is set in **Tools → Preferences → Sconegym**. `setup.sh` (via `scripts/configure_studio.py`) sets it to **`bin/studio-python`**, a small wrapper that:
* sends **PPO** checkpoints (run folder contains `config.json`) to `scripts/ppo_evaluate.py`;
* passes **depRL** checkpoints (run folder contains `config.yaml`) through to `deprl.play` unchanged.

Either way the rollouts appear as `.sto` files in `<run>/run_checkpoint_<N>/`. The original Studio settings are backed up at `~/.config/SCONE/studio-settings.zml.bak`.

---

## 3. Repository layout

```
scone-rl/
├── README.md                     this file: install / train / view
├── Readme_process_explained.md   how training works: inputs, outputs, reward, cycle, Hyfydy, math (PPO + DEP-RL)
├── setup.sh                      one-shot installer
├── env.sh                        `source env.sh` = activate .venv + SCONE_PATH
├── requirements.txt              pinned Python deps (torch installed separately by setup.sh)
├── bin/
│   └── studio-python             SCONE Studio's Python: routes PPO / depRL checkpoints (§2.3)
├── scone_ppo/                    ── PPO library (from gait-training) ──
│   ├── config.py                 every hyperparameter + CLI overrides (defaults = the proven walking run)
│   ├── envs.py                   EnergyWalkEnv: sconegym + effort cost + constant-speed reward + time limit + stats
│   ├── vec_env.py                64 simulations in 16 worker processes
│   ├── normalization.py          running observation normalisation, reward scaling
│   ├── networks.py               actor (policy) + critic (value) networks
│   └── ppo.py                    rollout buffer, GAE, clipped PPO update
├── configs/deprl/                ── DEP-RL configs ──
│   ├── sconewalk_h0918_v1.yaml        2-D, 18 muscles (published hyperparameters)
│   ├── sconewalk_h0918_v1_smoke.yaml  tiny pipeline test
│   ├── sconewalk_h1622_v1.yaml        3-D, 22 muscles
│   └── sconewalk_h2190_v1.yaml        3-D, 90 muscles (bigger networks)
├── scripts/
│   ├── check_install.py          verify python / sconepy / Hyfydy / envs / PPO / depRL / CUDA
│   ├── ppo_check_env.py          print obs layout, reward terms and sim speed of the PPO env
│   ├── ppo_train.py              PPO training
│   ├── ppo_evaluate.py           replay a PPO checkpoint → gait stats + .sto
│   ├── ppo_plot.py               progress.csv → training.png + reward_terms.png
│   ├── deprl_train.py            DEP-RL training (wraps deprl.main)
│   ├── deprl_evaluate.py         replay DEP-RL checkpoints → .sto (wraps deprl.play)
│   ├── deprl_plot.py             DEP-RL log.csv → training_curves.png
│   ├── list_runs.py              every run (PPO + DEP-RL) with checkpoints and rollouts
│   ├── configure_studio.py       set SCONE Studio's sconegym python
│   └── _paths.py                 results location helpers (§4)
├── results/                      ALL training output (git-ignored), see §4
│   ├── ppo/
│   └── deprl/
├── docs/
│   ├── VERIFICATION.md           record of the tests and benchmarks run on this machine
│   └── ppo_validation_5M.png     curves of the early 5M-step PPO validation run
├── third_party/depRL/            git submodule: tgeijten/depRL @ sconegym
├── logs/                         stdout logs (git-ignored)
├── .venv/  .tools/               Python env (git-ignored)
```

---

## 4. Where results are stored

**Everything is inside this repo, in `results/`.**

There is one catch. sconepy can only write `.sto` files into the SCONE results directory (`~/SCONE/results`); it has no API to change that and treats even absolute paths as relative to it. depRL does the same for its training runs. So there is **one symlink**:

```
~/SCONE/results/scone-rl   ->   scone-rl/results
```

Scripts write to `scone-rl/ppo/...` or `scone-rl/deprl/...` relative to the SCONE results dir, and the bytes land in `scone-rl/results/...`. SCONE Studio shows everything under a single **`scone-rl`** folder in its *Optimization Results* pane. Nothing else in `~/SCONE/results` is touched. `setup.sh` and the scripts create the link automatically.

```
results/
├── ppo/
│   └── 20261118_114942_const_1p2/           <YYYYMMDD_HHMMSS>_<run_name>
│       ├── config.json                      exact settings
│       ├── progress.csv                     one row per PPO update (8,192 steps)
│       ├── training.png, reward_terms.png   from ppo_plot.py
│       ├── latest.pt                        newest policy (overwritten every update)
│       ├── best.pt                          highest average return so far
│       ├── checkpoints/step_<N>.pt          snapshot every 25 updates (204,800 steps); N = env steps
│       ├── run_checkpoint_<N>/*.sto         rollouts of step_<N>.pt (Studio double-click or ppo_evaluate.py)
│       └── run_best/*.sto, run_latest/*.sto rollouts of best.pt / latest.pt
└── deprl/
    └── sconewalk_h0918_v1/                  experiment name
        └── 261118.142501.H0918v2j/          <YYMMDD.HHMMSS>.<model>, one per run
            ├── config.yaml, log.csv
            ├── checkpoints/step_<N>.pt      every 1e6 steps + resume state (replay buffer ~1.4 GB)
            └── run_checkpoint_<N>/*.sto
```

**Checkpoint sizes.** PPO is ~0.8 MB per checkpoint, about 80 MB for a 20M-step run. DEP-RL is 1.7 MB per `step_*.pt` plus a ~1.4 GB replay buffer that is overwritten at each save. `python scripts/list_runs.py` lists every run, its newest checkpoint, and its rollouts. `results/` is git-ignored, so back it up separately (`rsync -a results/ <backup>/`).

---

## 5. PPO pipeline (default)

### 5.1 What it trains

`sconewalk_h0918-v1` (2-D, 9 DOFs, 18 muscles) wrapped in `EnergyWalkEnv` (`scone_ppo/envs.py`). On top of sconegym's reward, the wrapper adds:
* **constant-speed walking**: a bell-shaped speed reward peaking **at** 1.2 m/s (averaged over 0.5 s), instead of sconegym's "at least 1.2 m/s";
* an **energy cost**: −10 × Σ(muscle volume weight × activation²);
* a 10 s **time limit** (1000 steps × 10 ms) that is handled as truncation, not failure;
* per-episode **gait statistics**: speed, distance, speed error, effort, effort per metre, fell.

The defaults in `scone_ppo/config.py` are exactly the settings of the run that learned to walk. Details and math are in `Readme_process_explained.md`, Part B.

### 5.2 Train

```bash
source env.sh
python scripts/ppo_train.py 2>&1 | tee logs/ppo_train.log         # 20M steps, proven settings
```

**How long?** Measured on this machine, about **7,000–9,500 env steps/s** (CPU only; the GPU is not needed), so the full 20M steps take **under an hour**. The exact timing of the setup run and when it started walking are in [`docs/VERIFICATION.md`](docs/VERIFICATION.md). The previous Windows run walked reliably (falls < 5 %) after about **13M steps (~28 min)**.

Common variations (any field of `scone_ppo/config.py` works as `--<field> <value>`):

```bash
python scripts/ppo_train.py --total_steps 2000000 --run_name quick          # ~4 min smoke test (won't walk yet)
python scripts/ppo_train.py --speed_mode min --run_name atleast_1p2         # sconegym's "at least 1.2 m/s"
python scripts/ppo_train.py --target_speed 1.0 --speed_sigma 0.15 --run_name const_1p0
python scripts/ppo_train.py --effort_coeff 30 --run_name effort30           # more economical gait, harder to learn
python scripts/ppo_train.py --effort_type cubed --effort_coeff 50 --run_name cubed
python scripts/ppo_train.py --use_delayed_sensors true --run_name delayed   # biologically delayed sensors (harder)
python scripts/ppo_train.py --seed 1 --run_name const_1p2_seed1             # always try 2-3 seeds
python scripts/ppo_train.py --num_workers 8 --envs_per_worker 8             # same 64 envs, fewer processes
```

Press **Ctrl+C** at any time; `latest.pt` is saved before exiting. PPO runs cannot be resumed (start a new one). They are short.

### 5.3 How often checkpoints are saved

| File | When |
|---|---|
| `latest.pt` | after **every** update (8,192 steps, ~1 s) |
| `best.pt` | whenever the average return of the last 50 episodes beats the previous best. This is noisy: test the final checkpoint first (§1). |
| `checkpoints/step_<N>.pt` | every `save_interval = 25` updates = **204,800 steps** (~25 s), plus the final update. That is about 98 snapshots over 20M steps. |

### 5.4 Reading the training log

Each update prints one line:

```
upd 1908/2441 | steps 15,630,336 | fps 7483 | return 8772.3 | len 1000.0 | speed 1.20+-0.02 m/s | effort 0.0394 | fell 0.00 | kl 0.0120 | ev 0.90 | t 0.8+0.4s
```

| Field | Meaning | What you want to see |
|---|---|---|
| `return` | total reward per episode (average of the last 50 episodes) | goes **up** (~8,000–9,000 when walking) |
| `len` | steps survived per episode (1000 = the full 10 s) | goes up to **1000** |
| `speed` | average forward speed ± mean speed error | settles near the **target** (1.2 m/s); error goes **down** |
| `effort` | mean effort per step | goes **down** once walking is stable |
| `fell` | fraction of episodes ending in a fall | goes **down** toward 0 |
| `kl` | how much the policy changed this update | small (≈ 0.005–0.03) |
| `ev` | critic explained variance | close to 1 |
| `t` | seconds for rollout + update | for diagnosing speed |

**Typical progression:** the model first learns not to collapse (`len` grows slowly from ~65 to ~200). Then it suddenly "gets" walking. In the previous run `len` jumped to ~850 and `fell` dropped from 1.0 to 0.2 between ~11M and ~13M steps; in the 2026-11-18 run the jump came between 13.1M and 13.8M steps. **Don't stop a run early because nothing seems to happen before ~13M steps.** Finally it refines the gait (`fell` ≈ 0–0.1, speed error ~0.03 m/s, effort drifting down).

```bash
python scripts/ppo_plot.py --latest              # or: python scripts/ppo_plot.py results/ppo/<run>
```
This writes `training.png` (return, length, speed, speed error, falls, effort, effort/m, explained variance, exploration std) and `reward_terms.png` into the run folder.

### 5.5 Tuning tips

* **Not learning to stay up?** Lower `--effort_coeff` (0–3) so falling isn't attractive.
* **Walking but wasteful?** Raise `--effort_coeff` gradually (10 → 20 → 40) and compare `effort_per_m`.
* **Unstable** (`kl` often > 0.03, return collapsing)? Use `--learning_rate 1e-4` or `--clip_coef 0.1`.
* **Exploration too timid or wild?** Change `--init_log_std` (−1.0 = more noise, −2.0 = less).
* **Reward hacks to look for in Studio:** diving forward (short, fast episodes), shuffling or knee-walking, hopping on one leg (check left/right symmetry).

---

## 6. Viewing a policy in SCONE Studio

Start Studio with `sconestudio` (or from the app menu). The **Optimization Results** pane on the left browses `~/SCONE/results`, where your data appears under **`scone-rl`**.

### 6.1 Double-click a checkpoint (Studio generates the animation)

1. Expand **`scone-rl` → `ppo` → `<run>` → `checkpoints`**. For DEP-RL: `scone-rl` → `deprl` → `<experiment>` → `<date>.<model>` → `checkpoints`.
2. **Double-click a `step_<N>.pt`**. The largest N is the latest. You can also select several, right-click, and choose **Evaluate**.
3. Studio runs `bin/studio-python` in the background (progress in the status bar). It rolls the policy out for *Number of episodes to evaluate* (default 5, set in Tools → Preferences → Sconegym).
4. A new folder **`run_checkpoint_<N>`** appears next to `checkpoints/`. Expand it. Press *Refresh* in the pane if it doesn't show up.
5. **Double-click a `.sto` file**, named `<episode>_<return>.sto`. For PPO the number is sconegym's own return, without the effort term.
6. Press **Play** (spacebar) on the timeline at the bottom. Drag to scrub; change the playback speed in the toolbar.

### 6.2 Or open already-generated rollouts

If you ran `ppo_evaluate.py` / `deprl_evaluate.py` (§7), the `.sto` files already exist in `run_best/`, `run_latest/` or `run_checkpoint_<N>/`. Browse to them and double-click. You can also open one directly:
```bash
sconestudio results/ppo/20261118_114942_const_1p2/run_checkpoint_19996672/00000_9325.863.sto
```

### 6.3 Analysing the gait

* **Analysis window** (*Window → Analysis*): plot any channel over time, such as muscle excitations and activations, joint angles, ground reaction forces, and COM velocity.
* **Gait analysis** (*Tools → Gait Analysis*): stride-normalised joint angles, GRFs, and muscle activity compared with normative human data.
* **Video export** (*File → Export Video*, uses `/usr/bin/ffmpeg`).

---

## 7. Playing back a saved policy from the terminal

```bash
source env.sh
python scripts/list_runs.py                                     # what is there?

# PPO
python scripts/ppo_evaluate.py --latest                         # newest run, latest.pt, 3 episodes + .sto
python scripts/ppo_evaluate.py results/ppo/<run>/latest.pt --episodes 5
python scripts/ppo_evaluate.py results/ppo/<run>/checkpoints/step_13107200.pt   # an earlier stage of learning
python scripts/ppo_evaluate.py results/ppo/<run>/latest.pt --seconds 60 --no_store # 60 s stability test, no files
python scripts/ppo_evaluate.py results/ppo/<run>/latest.pt --episodes 20 --no_store # reliability: how many of 20 walk
python scripts/ppo_evaluate.py results/ppo/<run>/latest.pt --stochastic           # with exploration noise

# DEP-RL
python scripts/deprl_evaluate.py --experiment sconewalk_h0918_v1                  # newest run, newest checkpoint
python scripts/deprl_evaluate.py --run results/deprl/<exp>/<run> --all            # every checkpoint
```

Each PPO episode prints `ok`/`FELL`, duration, distance, speed (and error), mean effort, effort per metre, and return. The `.sto` files are written as described in §4, ready for Studio (§6.2).

---

## 8. DEP-RL pipeline (for complex models)

**What it is.** DEP-RL (Schumacher et al., ICLR 2023) has two parts. **MPO** is an off-policy actor-critic with a replay buffer. **DEP** is a self-organising controller that replaces random exploration noise with coordinated, rhythmic muscle activity. It was designed for bodies with many muscles, where random noise makes muscles fight each other and nothing useful happens. Readme_process_explained.md Part C has the details and math.

**When to use it.** For the 2-D H0918 model, PPO is enough and much faster (it starts mid-stride, has only 18 muscles, and cannot fall sideways). Switch to DEP-RL for **3-D models (H1622, H2190 with 90 muscles), running, delayed sensors**, or to compare against the published DEP-RL results. It is installed and verified here, ready to use.

### 8.1 Train

```bash
source env.sh
python scripts/deprl_train.py configs/deprl/sconewalk_h0918_v1_smoke.yaml   # ~2 min pipeline test
tmux new -s deprl
python scripts/deprl_train.py configs/deprl/sconewalk_h0918_v1.yaml --steps 5e7 2>&1 | tee logs/deprl_h0918.log
```

| Option | Effect |
|---|---|
| `--steps 5e7` | total env steps (the config default is 5e8) |
| `--save-steps 5e6` | checkpoint interval (default 1e6) |
| `--name <n> --seed <s> --fresh` | separate experiment folder / new seed / don't resume |
| `--parallel 8 --sequential 10` | worker processes × envs per process (default 15 × 13) |
| `--cpu`, `--dry-run` | force CPU / print the resolved config only |

* **Resume:** rerun the same command. With `resume: true` depRL continues the newest run of that experiment, including its replay buffer.
* **Speed (measured):** ~8,600 steps/s during warm-up, **~2,600 steps/s while learning**, limited by the 30 GPU updates per 1,000 steps (each ~9 ms). That is **~5.5 h per 5e7 steps**. The **published DEP-RL walking baseline was trained for 5e7 steps**, so `--steps 5e7` is a sensible target. The config's 5e8 (~53 h) is the upstream default for long runs.
* **Checkpoints:** `step_<N>.pt` every 1e6 steps (~6.5 min). Each save also overwrites the ~1.4 GB resume state.
* **Curves:** `python scripts/deprl_plot.py --experiment sconewalk_h0918_v1`. Watch `test/episode_length/mean` → 1000 (25 s episodes) and `test/episode_score/mean`.

### 8.2 Pre-trained DEP-RL baseline

depRL can download the authors' trained H0918 walking policy (`deprl.load_baseline(env)`, 50M steps, from Google Drive; see `../sconegym/example_deprl.py`). It has not been downloaded here.

### 8.3 Differences from the PPO setup

| | PPO (`scone_ppo`) | DEP-RL (`configs/deprl`) |
|---|---|---|
| Control step / episode | 10 ms / 10 s | 25 ms / 25 s |
| Speed reward | bell curve **at** 1.2 m/s | sconegym plateau, **≥** 1.2 m/s |
| Effort | always on: −10 Σ wᵢaᵢ² | adaptive −c Σ aᵢ³, switched on once return > 1000 |
| Initial activations | random (0.3 ± 0.1) | fixed 0.01 |
| GRF penalty | −0.073 | −0.173 |
| Hardware | CPU (64 envs) | CPU workers (195 envs) + GPU learner |

---

## 9. Moving to more complex models

sconegym registers these Hyfydy environments (sizes measured here):

| Env id | Model | DOFs | Muscles | Obs size | PPO | DEP-RL config |
|---|---|---|---|---|---|---|
| `sconewalk_h0918-v1` | 2-D | 9 | 18 | 121 | ✅ proven | `configs/deprl/sconewalk_h0918_v1.yaml` |
| `sconewalk_h1622-v1` | 3-D | 19 | 22 | 161 | runs (untested for learning) | `configs/deprl/sconewalk_h1622_v1.yaml` |
| `sconewalk_h2190-v1` | 3-D | 21 | 90 | 505 | runs (untested for learning) | `configs/deprl/sconewalk_h2190_v1.yaml` (✅ walks after 1e8 steps) |
| `sconemimic_h2190-v1` | 3-D, scaled to a subject | 21 | 90 | 611 | | `configs/deprl/sconemimic_h2190_ab07.yaml` (imitation, see `docs/IMITATION_TRAINING.md`) |
| `sconerun_h0918/h1622/h2190-v1` | running | | | | | upstream: `third_party/depRL/experiments/hyfydy/scone_run_*.yaml` |

```bash
# PPO on another model: only the env id changes (networks size themselves to the obs/action dims)
python scripts/ppo_train.py --env_id sconewalk_h2190-v1 --total_steps 100000000 --run_name h2190
# DEP-RL on another model
python scripts/deprl_train.py configs/deprl/sconewalk_h2190_v1.yaml --steps 5e7
```

To write a DEP-RL config for a new model, copy an existing one and change: the `environment` id, `name`, `num_acts=` inside the `AdaptiveEnergyBuffer` (**must equal the number of muscles**), and optionally `mpo_args` (bigger networks for many muscles, as in the H2190 config). For PPO, note that `scripts/ppo_check_env.py` prints an observation layout that assumes the 2-D model.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not find sconepy` | `source env.sh`, and use `.venv/bin/python` (3.9) |
| Hyfydy not supported in `check_install.py` | license problem: SCONE Studio → Tools → Preferences → Hyfydy. **The trial expires 2026-12-04**; after that, training and playback of Hyfydy models fail. |
| Double-clicking a `.pt` in Studio does nothing / errors | Check Tools → Preferences → Sconegym → python = `<repo>/bin/studio-python`. Rerun `python scripts/configure_studio.py`, or run the same command by hand to see the error: `bin/studio-python -m deprl.play --checkpoint_file <file> --num_episodes 1` |
| `scone-rl` folder not visible in Studio | `ls -l ~/SCONE/results/scone-rl` should be a link to `<repo>/results`. Rerun `./setup.sh`, then press Refresh in the pane. |
| `best.pt` / `latest.pt` not clickable in Studio | Studio only lists `step_*.pt`. Use `ppo_evaluate.py` (§7) and open the `.sto` files. |
| `CUDA error: no kernel image` | wrong torch build. Rerun `./setup.sh` (2.7.1+cu128). Never `pip install -r third_party/depRL/requirements.txt`, because it downgrades torch. |
| DEP-RL resumes when you wanted a new run | use `--fresh` or a new `--name` |
| Desktop laggy during training | fewer workers: PPO `--num_workers 8 --envs_per_worker 8`; DEP-RL `--parallel 8` |

---

## 11. References

* SCONE / sconegym docs: <https://scone.software/doku.php?id=doc:sconegym>, SconePy API: <https://scone.software/doku.php?id=doc:sconepy>
* sconegym: <https://github.com/tgeijten/sconegym>; depRL: <https://github.com/tgeijten/depRL/tree/sconegym> (upstream <https://github.com/martius-lab/depRL>)
* Hyfydy user manual: `../hyfydy-user-manual.pdf`, <https://hyfydy.com>
* J. Schulman et al. *Proximal Policy Optimization Algorithms.* 2017.
* P. Schumacher, D. Häufle, D. Büchler, S. Schmitt, G. Martius. *DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems.* ICLR 2023.
* T. Geijtenbeek. *SCONE: Open Source Software for Predictive Simulation of Biological Motion.* JOSS 2019.
