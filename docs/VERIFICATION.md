# Verification record (2026-11-18)

This is a record of the checks, training runs and benchmarks done when the repo was set up on this workstation (RTX 5080, 16 logical cores, 46 GB RAM). It includes the merge of `gait-training` (PPO) into `scone-rl`.

## 1. Install check

```
$ python scripts/check_install.py
[OK] Python 3.9: 3.9.25
[OK] sconepy + Hyfydy: SCONE 2.4.5.3372, Hyfydy supported, results dir = /home/cshah/SCONE/results
[OK] gym env sconewalk_h0918-v1: obs (121,), act (18,), 18 muscles, ~3,008 steps/s (1 core)
[OK] PPO package (scone_ppo): scone_ppo imports, EnergyWalkEnv steps OK
[OK] depRL: from third_party/depRL
[OK] PyTorch: torch 2.7.1+cu128, CUDA OK on NVIDIA GeForce RTX 5080
All checks passed.
```

## 2. PPO: full training run on this machine

`python scripts/ppo_train.py` with the defaults (20M steps, constant speed 1.2 m/s, effort coefficient 10).

* Run folder: `results/ppo/20261118_114942_const_1p2/`
* Started 11:49:42 and finished 12:29:05, so **39 min 22 s** wall-clock at ~8,500 env steps/s on average (9,500 early, 8,500 late, CPU only). 98 checkpoints.
* Log: `logs/ppo_train_const_1p2.log`. Curves: `training.png`, `reward_terms.png` in the run folder.

| Env steps | Avg. episode length | Falls | Note |
|---|---|---|---|
| 0.5M | 69 | 100 % | |
| 7.3M | 170 | 100 % | |
| 13.1M | 204 | 100 % | |
| **13.4M → 13.8M** | **362 → 768** | **100 % → 32 %** | breakthrough |
| 15.0M | 897 | 12 % | |
| 19.4M | 1000 | 0 % | `best.pt` saved (update 2370) |
| 20.0M (final) | 892 | 12 % | training numbers include exploration noise |

**Evaluation** (deterministic policy, `scripts/ppo_evaluate.py`):

| Policy | 10 s episodes walked (20 starts, seed 7) | 60 s test |
|---|---|---|
| **this run, final `step_19996672.pt` = `latest.pt`** | **19/20**, 1.21 m/s, speed error 0.02–0.03, effort/m 0.034 | 2/3 walked 60 s / 72.5 m (the third fell at 2 s) |
| this run, `best.pt` | 10/20 | |
| old gait-training run, `latest.pt` | 19/20 | 4/4 (earlier test) |
| old gait-training run, `best.pt` | 16/20 | |

Conclusions:
* The setup reproduces the old result: same settings, same final reliability. The breakthrough came about 2M steps later (seed / thread-timing variation), which is still well within 20M steps.
* **`best.pt` is not the best policy.** It is chosen from a noisy training average. Use the final checkpoint.
* Remaining falls happen within 2–3 s, from a few randomised start states (noise on the initial pose and velocities, random leg swap).

Rollouts for SCONE Studio were generated in `run_checkpoint_19996672/` (3 episodes, all walked 10 s) and `run_best/`.

## 3. PPO: runs imported from gait-training

| Run | What |
|---|---|
| `results/ppo/20261105_101425_const_1p2_full/` | the original 20M-step run (Windows). Walks. Its checkpoints were renamed from `ckpt_<update>.pt` to `checkpoints/step_<env steps>.pt` so SCONE Studio lists them. |
| `results/ppo/20261104_190009_validation/` | 5M-step validation run. **Does not walk**: the learning rate anneals to zero before the ~11–14M-step breakthrough. Curves in `docs/ppo_validation_5M.png` (local only, not in git; regenerate with `scripts/ppo_plot.py`). |

All checkpoints, `progress.csv`, `config.json` and the stdout log were compared with the originals (identical) before `../gait-training` was deleted.

## 4. SCONE Studio playback path

Studio runs `bin/studio-python -m deprl.play --checkpoint_file <step_N.pt> --num_episodes N`. Both routes were tested from the command line with the paths Studio uses (`~/SCONE/results/scone-rl/...`):
* PPO checkpoint → `scripts/ppo_evaluate.py` → `.sto` in `<run>/run_checkpoint_<N>/`
* depRL checkpoint → `deprl.play` → `.sto` in `<run>/run_checkpoint_<N>/`

In both cases the files landed inside `scone-rl/results/`, through the symlink `~/SCONE/results/scone-rl`. Whether the Studio GUI lists the symlinked folder was not checked by hand. If it doesn't, see README §10.

## 5. DEP-RL: smoke training (`configs/deprl/sconewalk_h0918_v1_smoke.yaml`)

* 1e5 steps, 4 × 4 envs, GPU learner: finished in **~90 s** at **~1,140 env steps/s**.
* Output: `results/deprl/sconewalk_h0918_v1_smoke/261118.110021.H0918v2j/` with `checkpoints/step_50000.pt`, `checkpoints/step_100000.pt`, `config.yaml`, `log.csv` (5 epochs) and `training_curves.png`.

```
$ python scripts/deprl_evaluate.py --experiment sconewalk_h0918_v1_smoke -n 2
Episodes: 1  Score: 149.636  Length: 16  Terminal: True
Episodes: 2  Score: 192.927  Length: 21  Terminal: True
.sto files written to …/run_checkpoint_100000:  00000_149.636.sto  00001_192.927.sto
```

The policy falls after about 0.5 s, which is expected after only 1e5 steps. (The run was deleted on 2026-11-23 to free disk space.)

## 6. DEP-RL: throughput of the full config (`configs/deprl/sconewalk_h0918_v1.yaml`)

Two short benchmarks with 15 × 13 = 195 envs (deleted afterwards):

| Epoch (2e5 steps each) | steps/s | epoch time |
|---|---|---|
| 1, warm-up, no gradient updates | **8,622** | 23 s |
| 2, learning | **2,563** | 78 s |
| 3, learning | **2,615** | 77 s |

GPU utilisation was ~1 % (0.7 GB). The learner (30 MPO updates of batch 256 every 1000 env steps, ~9 ms each) takes ~70 % of wall-clock time, so it is the bottleneck, not Hyfydy.

| Target | Wall-clock estimate at ~2,600 steps/s |
|---|---|
| 1e7 steps | ~1 h |
| 5e7 steps (published baseline length) | ~5.5 h |
| 5e8 steps (full config default) | ~53 h (~2.2 days) |

A full-config checkpoint folder is 1.4 GB (replay buffer of 1e6 transitions, overwritten at each save) plus 1.7 MB per `step_*.pt`. The H1622 / H2190 DEP-RL configs pass `--dry-run` but have not been trained.

An earlier benchmark exposed a bug in `--steps`, which also rewrote `epoch_steps` and `save_steps`. It is fixed in `scripts/deprl_train.py`.

**The Hyfydy trial license expires 2026-12-04.**

## 7. DEP-RL: H2190 (90 muscles) to 1e8 steps (2026-11-20 → 11-21)

`scripts/train_h2190_full.sh` (full config, `--steps 1e8 --save-steps 1e7`), ~1,700 env steps/s.
Run: `results/deprl/sconewalk_h2190_v1/261120.140636.H2190v2k/`. The first 10M steps were trained with `--no-buffer`, then resumed from `step_10000185` with an empty buffer.

* Test episodes first reached the full 25 s at ~13.6M steps; by 99–100M they average 900–1000 steps.
* Final `step_100000095.pt`: 3/3 episodes walked the full 25 s, ~32.5 m at ~1.30 m/s (target plateau 1.2 m/s). Rollouts in `run_checkpoint_100000095/`.
* On 2026-11-23 the replay buffer and checkpoints 10M–90M were deleted to save disk; only the final policy and the small optimiser / normaliser / effort-cost files remain. Resuming (`./scripts/train_h2190_full.sh --steps 2e8`) works, but starts with an empty buffer.
