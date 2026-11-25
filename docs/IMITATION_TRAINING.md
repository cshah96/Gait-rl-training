# Gait imitation training: make a muscle model walk like a specific person

This guide covers the **imitation (mimic) pipeline** in `scone-rl`. It takes one person's measured gait (motion capture, force plates, inverse dynamics), scales the 90-muscle H2190 Hyfydy model to that person, and trains a reinforcement-learning policy. The policy drives the 90 muscles so that the simulated body reproduces the person's **kinematics** (joint angles, segment positions) and **kinetics** (joint moments, ground reaction forces).

The walking pipelines in the main [`README.md`](../README.md) only ask the model to walk forward at about 1.2 m/s, and whatever gait comes out is accepted. Imitation instead asks it to walk *like this person*, which is what biomechanics applications need (subject-specific simulation, exoskeleton and prosthesis studies, "what if" experiments).

> **Status (2026-11-23).** The whole pipeline is built and has been tested end to end on Camargo subject AB07: data conversion, model scaling, Studio playback of the data, the imitation environment, a DEP-RL smoke training run, evaluation, and Studio playback of the policy. A long training run has **not** been done yet. A policy that walks and tracks well needs roughly 10⁸ or more environment steps (a day or more); see [§8](#8-step-4-train).

---

## Contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Quick start](#2-quick-start)
3. [Pipeline overview and file map](#3-pipeline-overview-and-file-map)
4. [Step 0: the data (Camargo 2021 via AddBiomechanics)](#4-step-0-the-data-camargo-2021-via-addbiomechanics)
5. [Step 1: extract the treadmill reference (`mimic_extract_b3d.py`)](#5-step-1-extract-the-treadmill-reference)
6. [Step 2: scale the model and convert the motion (`mimic_build_subject.py`)](#6-step-2-scale-the-model-and-convert-the-motion)
7. [Step 3: the imitation environment (`scone_mimic/`)](#7-step-3-the-imitation-environment)
8. [Step 4: train (DEP-RL)](#8-step-4-train)
    * [8b. Kinetic fine-tune](#8b-kinetic-fine-tune-joint-moments-grf-centre-of-pressure)
9. [Step 5: evaluate and view](#9-step-5-evaluate-and-view)
10. [Smoke-test results](#10-smoke-test-results)
11. [Tuning and troubleshooting](#11-tuning-and-troubleshooting)
12. [New subject, other speed, other scaling](#12-new-subject-other-speed-other-scaling)
13. [Known limitations and next steps](#13-known-limitations-and-next-steps)
14. [References](#14-references)

---

## 1. The idea in one page

```
 person in the lab                         simulation
 -----------------                         ----------
 markers + force plates                    H2190: 21 DoFs, 90 Hill-type muscles (Hyfydy)
   -> AddBiomechanics: scaled skeleton,      scaled to the person's segment lengths + mass
      joint angles, joint moments (ID),
      ground reaction forces                policy  pi(obs) -> 90 muscle excitations, 40 Hz
            |                                         ^                |
            v                                         |                v
   reference motion  q*(t), dq*(t),       obs = muscle states + body state + "where the
   tau*(t), GRF*(t), feet*(t)  ---------->       reference goes in the next 0.125 s"
            |                                                          |
            +---------> reward = how close is the simulation to the person, every 25 ms
```

* **Reference State Initialisation (RSI).** Every episode starts at a random instant of the person's walking, in their exact pose and velocities. The policy therefore practises every phase of the gait cycle from the start, instead of first having to learn how to get from standing to walking.
* **Tracking reward.** The reward is a sum of terms of the form `w * exp(-k * error²)`, one each for joint angles, joint velocities, pelvis position, foot and torso positions, joint moments and ground forces, minus small effort and smoothness penalties. Each term is 1×w when the error is 0 and goes towards 0 as the error grows. The form follows Choi et al. 2026, who used exactly this H2190 + Hyfydy + SCONE stack.
* **Early termination.** The episode ends if the model falls, drifts more than 40 cm from the reference pelvis, or reaches the end of the reference clip. Because every step's reward is ≥ 0, ending early always loses reward, so the policy learns to stay on the reference.
* **Learning algorithm.** DEP-RL (MPO + DEP exploration), the same algorithm that taught this H2190 model to walk in 100M steps on this machine (`docs/VERIFICATION.md` §7).

---

## 2. Quick start

All commands run from `scone-rl/`.

```bash
# 0. (once) data in ../Data/ - here: Camargo AB07 treadmill file from AddBiomechanics
.venv-data/bin/python scripts/data_inspect_b3d.py ../Data/AB07_split5-001.b3d   # what's inside

# 1+2. reference + scaled model + Studio playback of the measured motion   (~1 min)
./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07

# 3. sanity check of the environment
source env.sh && python scripts/check_install.py          # includes "imitation env (scone_mimic)"

# 4a. smoke test: 1e5 steps, 16 envs, ~3 min - checks training, checkpoints, playback
python scripts/deprl_train.py configs/deprl/sconemimic_h2190_ab07_smoke.yaml --fresh

# 4b. real training: detached, 1e8 steps (~20 h), resumable
./scripts/train_mimic.sh
tail -f logs/mimic_train.log

# 5. evaluate: report + plots + .sto files for SCONE Studio
python scripts/mimic_evaluate.py --experiment sconemimic_h2190_ab07 -n 5
```

---

## 3. Pipeline overview and file map

```
../Data/AB07_split5-001.b3d                     AddBiomechanics file (Camargo 2021, treadmill)
        |  scripts/mimic_extract_b3d.py          [.venv-data, Python 3.11, nimblephysics]
        v
data/mimic/AB07/AB07_treadmill_raw.npz          lab frame, OpenSim conventions, 23 DoFs
        |  scripts/mimic_build_subject.py        [.venv, Python 3.9, sconepy + Hyfydy]
        v
data/mimic/AB07/model/H2190_AB07.scone          scaled model (BodyModifier / ModelModifier)
data/mimic/AB07/AB07_reference.npz              reference in H2190 coordinates (what is tracked)
data/mimic/AB07/scaling.txt                     scale factors + validation
results/mimic/AB07/*.sto                        measured motion on the scaled model (Studio)
        |  scone_mimic/ (env)  +  configs/deprl/sconemimic_h2190_ab07*.yaml
        |  scripts/deprl_train.py / scripts/train_mimic.sh
        v
results/deprl/sconemimic_h2190_ab07/<run>/checkpoints/step_*.pt
        |  scripts/mimic_evaluate.py  /  scripts/deprl_evaluate.py  /  Studio double-click
        v
<run>/mimic_eval_step_<N>/{report.txt, tracking.png, *.sto}
```

| File | Role |
|---|---|
| `scripts/data_inspect_b3d.py` | Prints everything in a `.b3d` file: subject, model, trials, processing passes, every per-frame channel. |
| `scripts/mimic_extract_b3d.py` | Step 1: selects treadmill segments at the target speed and writes the raw `.npz`. |
| `scripts/mimic_build_subject.py` | Step 2: scales the model, converts conventions, validates, writes the reference `.npz` and the playback `.sto` files. |
| `scripts/mimic_prepare_subject.sh` | Steps 1 + 2 in one command (creates `.venv-data` if needed). |
| `scone_mimic/__init__.py` | Registers the gym id `sconemimic_h2190-v1`. |
| `scone_mimic/env.py` | `SconeMimicGym`: reset (RSI), observation, reward, termination; `Reference` loader. |
| `configs/deprl/sconemimic_h2190_ab07.yaml` | Full training config (15 × 13 = 195 envs). |
| `configs/deprl/sconemimic_h2190_ab07_smoke.yaml` | Smoke-test config (4 × 4 envs, 1e5 steps). |
| `scripts/train_mimic.sh` | Detached long run (1e8 steps ≈ 20 h, checkpoint every 2e7, no buffer save unless `BUFFER=1`). |
| `scripts/mimic_evaluate.py` | Tracking report (RMSE / r per joint, moments, GRF), plot, `.sto`. |
| `.venv-data/` | Python 3.11 env with `nimblephysics` (reads `.b3d`; sconepy needs 3.9, so they are separate). |
| `.venv/lib/python3.9/site-packages/scone_rl_root.pth` | Puts the repo on `sys.path`, so `scone_mimic` imports in depRL workers and in Studio playback (written by `setup.sh`). |

---

## 4. Step 0: the data (Camargo 2021 via AddBiomechanics)

**Dataset.** Camargo et al. 2021 recorded 22 able-bodied adults walking on level ground, ramps, stairs and a treadmill. It contains motion capture, force plates, EMG and IMUs. [AddBiomechanics](https://addbiomechanics.org) reprocessed it: for each subject it fitted a scaled OpenSim model (Rajagopal 2016 family, 23 DoFs, no arms), solved inverse kinematics, and solved inverse dynamics in a way that is **dynamically consistent** (residual forces are minimised by small adjustments to the motion). The result is one `.b3d` file per subject and "split".

**Splits (per subject):** 0 = level ground clockwise, 1 = level ground counter-clockwise, 2–3 = ramps, 4 = stairs, **5 = treadmill**. Overground trials only contain 1–2 steady strides between standing and turning, and force data is often missing. **Treadmill (split 5) is the one to use:** about 1,000 s per subject, belt speeds 0.5–2.0 m/s, and force data on 100 % of frames.

**What a `.b3d` contains** (`scripts/data_inspect_b3d.py`; run it to get the full dump, e.g. `.venv-data/bin/python scripts/data_inspect_b3d.py ../Data/AB07_split5-001.b3d > docs/camargo_b3d_inspection_AB07.txt`; dumps are not kept in git):

| Kind | Content |
|---|---|
| Subject | height, mass, sex, age; the scaled OpenSim model (80 muscles) as text |
| Per trial | name (e.g. `treadmill_05_01_segment_12`), length, dt = 0.005 s (200 Hz), missing-GRF flags, marker RMS |
| Per frame, 3 processing passes (KINEMATICS, LOW_PASS_FILTER, **DYNAMICS**) | `pos`, `vel`, `acc` (23 coordinates), `tau` (inverse-dynamics generalised forces), `groundContactForce` / `CenterOfPressure` / `Torque` per foot (calcn_r, calcn_l), `contact` flags, `comPos/Vel`, 12 `jointCenters`, residual wrench |
| Per frame, sensors | 28 marker positions. **No EMG and no IMU** in this release (they are only in the original Camargo download). |

We use the **DYNAMICS** pass. Its kinematics and kinetics are consistent with each other, which matters when the reward compares both.

**Subjects available here (treadmill):**

| Subject | Sex | Height | Mass | Treadmill time | Speeds |
|---|---|---|---|---|---|
| AB07 | m | 1.65 m | 55.3 kg | 1,293 s | 0.49–2.06 m/s |
| AB08 | m | 1.74 m | 72.6 kg | 1,004 s | 0.49–1.88 m/s |
| AB09 | f | 1.63 m | 63.5 kg | 1,004 s | 0.47–1.77 m/s |
| AB11 | m | 1.75 m | 77.1 kg | 1,004 s | 0.48–1.84 m/s |

Quality per segment (speed, cadence, peak forces and moments) is in `docs/camargo_treadmill_stats.txt` (local only, not in git).

> **Gotcha: nimblephysics 0.10.x segfaults when its objects are freed.** All scripts that read `.b3d` files keep the objects in a global list and end with `os._exit(0)`. If you write your own reader, do the same, or it will crash at exit (or between files).

---

## 5. Step 1: extract the treadmill reference

```bash
.venv-data/bin/python scripts/mimic_extract_b3d.py ../Data/AB07_split5-001.b3d --subject AB07 \
    [--speed 1.2] [--tol 0.1] [--max-std 0.06]
```

**Belt speed.** The file does not store the treadmill speed, so it is estimated: while a foot is on the belt it moves backwards at belt speed. The horizontal speed of the right subtalar joint centre during right-foot contact is taken, a 1-s running median is applied, and gaps (swing) are filled by interpolation.

**Segment selection.** In Camargo's protocol the belt speed ramps during each 150-s run (which AddBiomechanics cut into 10-s segments). A segment is kept when:
* no frame is missing force data,
* its median belt speed is within `--tol` of `--speed` (1.2 ± 0.1 m/s), and
* the speed's standard deviation is ≤ `--max-std` (0.06 m/s), which excludes segments where the speed is still ramping.

For AB07 this keeps **9 segments × 10 s = 90 s** (about 90 right strides) at a mean of 1.20 m/s.

**Segment lengths** are measured from the median joint-centre distances: hip→knee (femur), knee→ankle (tibia), ankle→mtp (foot), hip→hip (pelvis width), pelvis origin→lumbar joint. They are used for scaling in step 2.

**Output** `data/mimic/<S>/<S>_treadmill_raw.npz` (all frames of the kept segments concatenated; `seg_start` / `seg_len` mark the boundaries): `q, dq, ddq, tau` (N×23), `grf, cop, grm` (N×2×3), `contact` (N×2), `joint_centers` (N×12×3), `com`, `belt_speed`, the subject's height / mass / sex / age, and the `len_*` segment lengths.

---

## 6. Step 2: scale the model and convert the motion

```bash
source env.sh
python scripts/mimic_build_subject.py --subject AB07 [--scaling segments|height]
```

### 6.1 Scaling the H2190 model

SCONE can scale any Hyfydy model inside the scenario file. The bundled example is `/opt/scone/scone/scenarios/Examples3/Gait2D - GeyerHerr2010 - Scaled - Hyfydy.scone`.
* `BodyModifier { include = <bodies> scale = [x y z] }` scales a body: its geometry, its joint positions, its muscle attachment points and contact spheres, and its mass and inertia (by volume).
* `ModelModifier { mass = M }` then rescales all masses so the total is the subject's mass.
* (Not used yet:) `MuscleModifier { include = "vasti*" max_isometric_force { factor = 0.9 } }` changes muscle strength.

The script writes `data/mimic/<S>/model/H2190_<S>.scone`. This is the sconegym `H2190.scone` with these lines added (AB07):

```
BodyModifier { include = "pelvis" scale = [ 0.9167 0.9167 0.8826 ] }
BodyModifier { include = "femur_r;femur_l" scale = [ 1.1352 1.1352 1.1352 ] }
BodyModifier { include = "tibia_r;tibia_l" scale = [ 0.9418 0.9418 0.9418 ] }
BodyModifier { include = "calcn_r;calcn_l" scale = [ 1.0181 1.0181 1.0181 ] }
BodyModifier { include = "torso" scale = [ 0.9167 0.9167 0.9167 ] }
ModelModifier { mass = 55.30 }
```

How the factors are computed (`--scaling segments`, the default):

| Segment | Measured length | H2190 length (from `H2190v2k.hfd`) | AB07 factor |
|---|---|---|---|
| femur | hip→knee joint centres | 0.170 + 0.226 = 0.396 m | 0.450 / 0.396 = **1.135** |
| tibia | knee→ankle | 0.1867 + 0.2433 = 0.430 m | 0.405 / 0.430 = **0.942** |
| foot (calcn) | ankle→mtp | ankle→front contact sphere = 0.143 m | 0.145 / 0.143 = **1.018** |
| pelvis (z = width) | hip_r→hip_l | 2 × 0.0835 = 0.167 m | 0.147 / 0.167 = **0.883** |
| pelvis (x, y), torso | no head or shoulder markers | body height / 1.80 m | 1.65 / 1.80 = **0.917** |

`--scaling height` instead scales every segment uniformly by height / 1.80 m. AB07's measured thigh is long for a 1.65 m person (0.27 × height; a typical value is 0.245), which may partly come from how AddBiomechanics fitted the knee to the markers. Segment scaling follows the data, so the joint angles are tracked on the same skeleton they were measured on.

### 6.2 Converting the motion to H2190 conventions

The data comes from the OpenSim Rajagopal 2016 model, while H2190 follows OpenSim gait2392 conventions with its own pelvis origin. The conversions found and checked:

| Issue | Conversion |
|---|---|
| DoFs | The 21 H2190 DoFs have the same names as in the data; the extra `mtp_r/l` are dropped. |
| **Knee sign** | H2190 knee flexion is **negative** (range −120…10°), the Rajagopal walker knee is positive. `knee_angle_r/l` position, velocity and moment are negated. |
| Pelvis rotation order | Both are intrinsic Z (tilt) – X (list) – Y (rotation). Checked against Hyfydy's pelvis quaternion: 0.27° difference. |
| Heading | The lab yaw is removed (a constant rotation about vertical), so the subject faces +x. Positions, forces and centres of pressure are rotated with it. |
| Treadmill → overground | The simulation has no treadmill. So `pelvis_tx` (and every position) gets the integrated belt travel ∫v_belt dt added, and the subject walks forward at the belt speed, as an overground walker would. Each 10-s segment starts at x = 0, z ≈ 0. |
| **Pelvis origin** | In OpenSim the hips are 5.6 cm behind and 7.9 cm below the pelvis origin; in H2190 they are straight below it. The model pelvis is placed so that **the model's hip-centre midpoint equals the measured one** (otherwise the whole model sits about 7 cm off). |
| Ground height | `pelvis_ty` is shifted so that the lowest contact sphere of the stance foot is at y = 0 (median over stance frames). |
| Pelvis velocities | Recomputed by finite differences after the rotation and the belt travel are applied. |
| Torque spikes | Inverse-dynamics hip moments peak at about 6 N·m/kg in a few frames (normal 1.5–2). The torque reward uses a copy clipped per DoF at the 99.5th percentile (`tau_clipped`; AB07 hip flexion is clipped at 1.7 N·m/kg). |

### 6.3 Validation (written to `scaling.txt`)

Every 10th reference frame is set on the scaled model in Hyfydy, and the model's joint centres are compared with the measured ones:

| AB07 | median | 95th pct |
|---|---|---|
| hip | 0.8 cm | 1.4 cm |
| knee | 3.5 cm | 3.8 cm |
| ankle | 2.8 cm | 3.2 cm |
| lowest foot point | 0.0 cm | min −1.6 cm (slightly into the ground) |

The remaining ~3 cm at the knee and ankle is a near-constant offset between the two models' joint definitions (for example, the Rajagopal walker knee moves its centre with flexion). Before the knee-sign and pelvis-origin fixes, the ankle error was 40–80 cm, so this check is worth keeping whenever a new subject or model is added.

### 6.4 Model-consistent end-effector positions

For every reference frame, the script also stores the world positions of `calcn_r`, `calcn_l` and `torso` **as computed by the scaled H2190 model** (`ee_pos`). The end-effector reward compares the simulation with these, so it does not also penalise the few-cm offset between the two models.

### 6.5 Outputs and viewing the measured motion

* `data/mimic/<S>/<S>_reference.npz` (H2190 DoF order; 200 Hz): `dof_names, q, dq, tau, tau_clipped, grf, cop, contact, joint_centers, com, ee_pos, ee_names, belt_speed, seg_start, seg_len, dt, mass, height, model_file`.
* `results/mimic/<S>/<S>_0k_<segment>.sto`: the first 3 segments played on the scaled model. `default_model/` holds the same motion on the unscaled H2190.

**View in SCONE Studio:** Results panel → `scone-rl/mimic/AB07` (or File → Open `~/SCONE/results/scone-rl/mimic/AB07/`) → double-click a `.sto` → Play. Each `.sto` contains the full Hyfydy state (body positions and orientations, muscle activations and fibre lengths) and, for plotting, the DoF columns `<dof>` and `<dof>_u`. Muscles show idle activations, because this is kinematic playback, not simulation.

---

## 7. Step 3: the imitation environment

`scone_mimic.SconeMimicGym` subclasses sconegym's `GaitGym`, so model loading, muscle observations, fall detection and `.sto` writing are the same as in the walking environments.

```python
import gym, scone_mimic
env = gym.make("sconemimic_h2190-v1", reference="data/mimic/AB07/AB07_reference.npz")
```

The reference file names its scaled model, so the subject's model is loaded automatically. Every parameter below is an attribute that a deprl config can override in `env_args`. The defaults are in `scone_mimic/env.py: DEFAULT_MIMIC_ARGS`.

### 7.1 Timing

* Hyfydy integrates internally with variable steps. The policy acts every **`step_size` = 0.025 s (40 Hz)**; Choi et al. used 25 Hz.
* Reference index: `i = start + round(t / 0.005)`, so the reference advances 5 frames per control step.
* **`max_episode_steps` = 300**, i.e. 7.5 s episodes. Episodes also end at the end of the 10-s reference segment.

> deprl's `SconeWrapper` calls `_get_rew`, `_get_obs`, `_get_done` directly and forces its own 1,000-step limit, so the environment ends episodes itself (`_get_done`) and uses `model.time()` for indexing.

### 7.2 Reset: Reference State Initialisation

1. Pick a random reference segment and a random start frame in it, leaving room for a full episode plus the look-ahead.
2. Set the DoF positions and velocities to the reference at that frame, plus Gaussian noise (`init_pos_noise` = 0.02 rad or m, `init_vel_noise` = 0.05).
3. Set all 90 muscle activations to `init_activations_mean` = 0.05, then `init_state_from_dofs()` (Hyfydy places the bodies and equilibrates the muscle fibres).
4. No leg switching: the reference is a real, slightly asymmetric person.

### 7.3 Observation (611 numbers)

| Part | Size | Content |
|---|---|---|
| sconegym 3-D observation | 505 | muscle fibre lengths (90), fibre velocities (90), forces (90), excitations (90); head (torso) orientation quaternion (4) and angular velocity (3); feet relative to pelvis (6); DoF positions (21, with `pelvis_tx` / `pelvis_tz` set to 0); DoF velocities (21); activations (90) |
| reference look-ahead | 5 × 21 = 105 | `q*(t + k·0.025 s) − q(t)` for k = 1…5 (the next 0.125 s), all 21 DoFs (pelvis translations included, so the policy sees whether it is ahead of or behind the reference) |
| time left | 1 | `1 − steps / max_episode_steps`, so the critic can anticipate the episode end |

### 7.4 Action

90 muscle excitations in [0, 1] (`clip_actions: false`; the walking config clips at 0.5). Activation dynamics in Hyfydy turn excitations into activations.

### 7.5 Reward

Each control step, with `*` = reference at the current time:

| Term | Formula | Weight `w` | Sharpness `k` | What it asks for |
|---|---|---|---|---|
| `pos` | exp(−k Σⱼ cⱼ (qⱼ − q*ⱼ)²) | 0.25 | 2.0 | joint angles (18 rotational DoFs incl. pelvis tilt / list / rotation and lumbar) |
| `vel` | exp(−k Σⱼ cⱼ (q̇ⱼ − q̇*ⱼ)²) | 0.10 | 0.05 | joint angular velocities |
| `root` | exp(−k ‖p_pelvis − p*_pelvis‖²) | 0.15 | 500 | pelvis position, i.e. walking speed and height (5 cm error gives 0.29) |
| `ee` | exp(−k Σₑ ‖(pₑ − p_pelvis) − (p*ₑ − p*_pelvis)‖²) | 0.25 | 80 | feet (calcn) and torso positions relative to the pelvis: step length and width, foot clearance, trunk lean |
| `torque` | exp(−k Σⱼ ((τ_muscle,ⱼ − τ*ⱼ)/m)²) | 0.25 | 2.0 | **kinetics**: net muscle moment (`dof.muscle_moment()`) vs inverse-dynamics moment at hip flexion / adduction / rotation, knee and ankle, both sides (10 DoFs), in N·m/kg |
| `grf` | exp(−k Σ_f ‖(F_f − F*_f)/(m g)‖²) | 0.10 | 5.0 | **kinetics**: 3-D ground reaction force of each foot, in body weights |
| `effort` | −w · mean(a²) | 0.05 | | muscle effort (a = activation) |
| `smooth` | −w · mean((e_t − e_{t−1})²) | 0.1 | | smooth excitations |
| | **r = max(Σ terms, `min_reward` = 0)** | | | |

* cⱼ = 1, except `subtalar_weight` = 0.2 for the subtalar angles. AB07's subtalar angle is outside H2190's −10…30° range on 42–64 % of frames, so tracking it closely is impossible without straining the joint limit.
* The positive weights sum to **1.1**, so a perfect 7.5-s episode returns about 330 and the per-step reward is in [0, 1.1].
* `pos`, `vel`, `root`, `ee` and `torque` follow Choi et al. 2026 (their "base policy": same weights and k values; they used metabolic energy for effort and smoothness weight 1.0). The `grf` term follows Liu et al. 2026, who showed that imitating kinematics alone produced unrealistic forces and moments, and that adding GRF and centre-of-pressure terms fixed most of it.
* **Why `min_reward` = 0 and `w_smooth` = 0.1.** The first smoke test used the paper's smoothness weight of 1.0. DEP exploration produces jerky excitations, so the smoothness penalty exceeded the tracking reward and training episodes had a **negative** return (−3). With early termination, a negative step reward teaches the policy to fall as fast as possible. Clipping at 0 (as DeepMimic does) and using a weight of 0.1 fixed it: training returns became positive (+1.5 → +2.4).
* **Check:** with the model set exactly onto the reference, the terms are pos 0.25, vel 0.10, root 0.15, ee 0.25, which is the maximum. (`torque` and `grf` need a simulated step to be meaningful.)
* The muscle moment is the net moment of all 90 muscles about the DoF. The inverse-dynamics moment also contains ligament and passive effects, so a perfect match is not expected. The sign convention was checked (ankle plantarflexion moment negative in both).

### 7.6 Termination

* **Fall:** COM below 0.5 m or head (torso) below 0.9 m (sconegym).
* **Drift:** ‖p_pelvis − p*_pelvis‖ > **`max_root_dev` = 0.4 m** (Choi et al.).
* **Episode length:** 300 steps.
* **End of the reference segment.**

---

## 8. Step 4: train

### 8.1 Algorithm and config

`configs/deprl/sconemimic_h2190_ab07.yaml`. It is the H2190 walking DEP-RL configuration with the environment swapped:
* `header: "import deprl, gym, sconegym, scone_mimic"`: registers the env in every worker.
* `environment: deprl.environments.Gym('sconemimic_h2190-v1', scaled_actions=False, reference='data/mimic/AB07/AB07_reference.npz')`.
* Agent: `dep_factory(3, TunedMPO())` with a **plain replay buffer** (1e6 transitions, 30 updates of batch 256 every 1,000 env steps, 2e5 warm-up steps). The walking configs' `AdaptiveEnergyBuffer` is not used, because the reward already has an effort term and returns never reach its 1,000 threshold.
* `mpo_args`: 1024-unit networks and low learning rates (upstream H2190 values).
* 15 workers × 13 envs = 195 parallel simulations.
* `env_args`: all reward weights and episode settings from §7, so tuning means editing the yaml, not code.
* `DEP`: unchanged from the paper.

Why DEP-RL rather than the SAC of Choi et al.: DEP-RL is already installed, proven on this exact model here (H2190 walked after ~14M steps and steadily by 100M), and its DEP exploration is designed for many-muscle models. A SAC implementation (Stable-Baselines3) could be added if a direct comparison with the paper is needed.

### 8.2 Smoke test (about 3 minutes)

```bash
source env.sh
python scripts/deprl_train.py configs/deprl/sconemimic_h2190_ab07_smoke.yaml --fresh
```
4 × 4 envs, 1e5 steps, warm-up 2e4, checkpoints at 50k and 100k, no replay-buffer saving. Results go to `results/deprl/sconemimic_h2190_ab07_smoke/<run>/`. It checks that every part runs: it is **not** expected to learn walking (see §10).

### 8.3 Long run

```bash
./scripts/train_mimic.sh                       # detached; 1e8 steps; checkpoint every 2e7; no buffer save
./scripts/train_mimic.sh configs/deprl/sconemimic_h2190_ab07.yaml --steps 2e8   # longer
BUFFER=1 ./scripts/train_mimic.sh              # also save the ~6 GB replay buffer (exact resume)
tail -f logs/mimic_train.log                    # progress bar
pkill -f "deprl_train.py configs/deprl/sconemimic"   # stop; re-run train_mimic.sh to resume
```

* **Full-scale test (2026-11-23).** The full config (195 envs) ran for 10 min.

  | Phase | Speed |
  |---|---|
  | Warm-up (2e5 steps, no learning) | 2,100 env steps/s |
  | **Learning** | **~1,425 env steps/s** (one 2e5 epoch takes 140 s) |

  * **1e8 steps take about 19–20 h.**
  * Resources: about 12 GB of RAM on top of the idle machine (24 GB used in total) and 1.4 GB of GPU memory.
  * The test episode length rose from 15.6 to 25 steps in 6e5 steps. At 9e5 steps the policy stays up for 0.73 s (the smoke run's was 0.38 s).
  * The time-limit stop, the checkpoint saves and `mimic_evaluate.py` all worked.
* **Disk.**
  * Each `step_<N>.pt` is 29 MB, so 1e8 steps at the default 2e7 interval gives 5 files, about 150 MB.
  * `actor.pt` / `critic.pt` (overwritten) add 30 MB.
  * The replay buffer (~6 GB, overwritten) is **not** saved unless `BUFFER=1`. A resumed run then starts with an empty buffer, which refills in a few minutes.
  * Choi et al. trained their base policy for 600M steps (~48 h on an RTX 5090, SAC).
* **Watch progress.**
  * In `<run>/log.csv`, `test/episode_length/mean` is the deterministic policy's episode length (300 = it always survives 7.5 s).
  * `test/episode_score/mean` is the tracking score (330 = perfect tracking).
  * To plot them: `python scripts/deprl_plot.py --experiment sconemimic_h2190_ab07`.
* **Resume / extend.** Re-running the same command resumes the newest run of the config (`resume: true`). To train longer, pass a larger `--steps`. The script refuses to start a second copy while one is running.

## 8b. Kinetic fine-tune (joint moments, GRF, centre of pressure)

The 1e8-step AB07 run (2026-11-24, 16.8 h) tracks the kinematics well:
* joint angles within 3–5° (r 0.97–0.98 for hip and knee flexion);
* ground force r ≈ 0.95;
* ankle moment r 0.95.

But it earned only 35 % of the `torque` term, and the hip moments correlate only about 0.55 with the measured ones.

The causes:
* the inverse-dynamics hip moments are spiky;
* the original torque term sums 10 squared errors inside one exponential, so a single bad joint drives the whole term to about 0;
* kinematics dominated the total reward.

`configs/deprl/sconemimic_h2190_ab07_kinetic.yaml` therefore continues **from the 100M checkpoint** with these `env_args` changes:

| Setting | Original | Kinetic fine-tune |
|---|---|---|
| `w_torque` / `k_torque` / `torque_mode` | 0.25 / 2 / `sum` | **0.5 / 10 / `per_joint`**: `mean_j exp(-k e_j²)` |
| `tau_lowpass_hz` | off | **8 Hz** (4th-order zero-lag Butterworth per segment; hip peaks ±1.7 → +0.7/−0.9 N·m/kg) |
| `w_grf` / `k_grf` / `grf_mode` | 0.10 / 5 / `sum` | **0.2 / 10 / `per_foot`** |
| `w_cop` / `k_cop` | — | **0.1 / 500**: horizontal COP relative to the calcn, per loaded foot (Liu et al. 2026) |

The maximum per-step reward rises from 1.1 to 1.55. The observation is unchanged, so the trained network is reused as is.

```bash
./scripts/train_mimic_kinetic.sh          # +3e7 steps (to 1.3e8), ~5 h, checkpoint every 1e7, detached
tail -f logs/mimic_kinetic.log
python scripts/mimic_evaluate.py --experiment sconemimic_h2190_ab07_kinetic -n 5
```

On the first call, the script copies the source run's last checkpoint, its optimiser and normaliser files, and `log.csv` into `results/deprl/sconemimic_h2190_ab07_kinetic/<same run>/`. depRL then *resumes* there at step 1e8, so the step count and learning curve continue. The original run is not changed.

Test (4 min, 3.9e5 steps of fine-tuning), evaluated against the new reward:
* every episode lasts 7.5 s at 1.20 m/s;
* the policy earns, as a share of each term's maximum: torque 74 %, grf 83 %, cop 64 %;
* hip moment r is 0.66–0.74 (partly because the reference is now smoothed);
* COP error is 2.6–2.9 cm fore-aft.

`mimic_evaluate.py` now also reports each term as a percentage of its maximum and the COP error.

> **Resume bug fixed (affects all depRL resumes).** PyTorch ≥ 2.6 loads with `weights_only=True`, which rejects the numpy arrays in `obs_norm.pt`. Every resume therefore printed "Not loading observation normalizer" and restarted the input normalisation. `scripts/deprl_train.py` now loads this repo's own checkpoints with `weights_only=False` (`trust_own_checkpoints()`).

### Why SCONE Studio shows several GRF arrows per foot

H2190 has **three contact spheres per foot**: `heel` (r = 3 cm), `toe_lat` and `toe_med` (r = 2 cm). Hyfydy computes a Hunt–Crossley contact force for each sphere touching the ground, and Studio draws one arrow per sphere. So a foot shows 1 arrow at heel strike, 3 in mid-stance and 1–2 at push-off, and up to 6 in double support. The measured GRF is the **sum** of a foot's arrows, applied at the centre of pressure; that is exactly what `leg.contact_force()` / `contact_pos()` return and what the reward compares. The extra arrows are not a bug. The kinetic fine-tune does not change their number, but the COP term makes the load move from the heel arrow to the toe arrows over stance as it does for the person.

---

## 9. Step 5: evaluate and view

### 9.1 Tracking report and plots

```bash
python scripts/mimic_evaluate.py --experiment sconemimic_h2190_ab07 -n 5     # newest run, last checkpoint
python scripts/mimic_evaluate.py --checkpoint <run>/checkpoints/step_N.pt -n 10
```
This runs the deterministic policy from noise-free reference starts and writes `<run>/mimic_eval_step_<N>/`:
* `report.txt`:
  * episode duration (vs the 7.5 s maximum), return, COM speed vs belt speed;
  * mean reward terms;
  * per-joint **RMSE (deg) and Pearson r** for joint angles;
  * **RMSE (N·m/kg) and r** for hip, knee and ankle moments;
  * **RMSE (BW) and r** for each GRF component.
* `tracking.png`: measured (black) vs simulated curves for hip, knee and ankle angles and moments, both vertical GRFs, and the tracking reward, for the longest episode.
* `*.sto` + `config.scone` + model files: the evaluated episodes, ready for SCONE Studio.

What "good" looks like (Choi et al. 2026, base policy, H2190): joint angle RMSE hip / knee / ankle ≈ **2.9° / 4.0° / 3.7°**, moments ≈ **0.10 N·m/kg**, GRF ≈ **0.14 BW**. Typical between-person variability is several degrees, so RMSE < 5° with r > 0.9 is a strong result.

### 9.2 Watching the policy in SCONE Studio

* Results panel → `scone-rl/deprl/sconemimic_h2190_ab07/<run>/` → double-click a `checkpoints/step_*.pt`. Studio runs `bin/studio-python -m deprl.play`, which imports `scone_mimic` through the `.pth` file, and writes `run_checkpoint_<N>/*.sto` with the **scaled** AB07 model.
* Or open `mimic_eval_step_<N>/*.sto`.
* Or from the terminal: `python scripts/deprl_evaluate.py --experiment sconemimic_h2190_ab07 -n 3`.
* To compare with the person, open `results/mimic/AB07/AB07_00_*.sto` (the measured motion) in a second Studio window.

---

## 10. Smoke-test results

`configs/deprl/sconemimic_h2190_ab07_smoke.yaml`, 2026-11-23: 1e5 steps, 16 envs, **2 min 56 s**, ~580 env steps/s.

| Check | Result |
|---|---|
| env builds, obs 611, act 90 | ✅ (`scripts/check_install.py`) |
| reward at exact reference = maximum of the kinematic terms | ✅ pos 0.25, vel 0.10, root 0.15, ee 0.25 |
| GRF order / sign, moment sign | ✅ stance foot positive vertical force; ankle moments agree in sign |
| training runs, logs, checkpoints (`step_50000.pt`, `step_100000.pt`) | ✅ |
| training return positive (after the `min_reward` fix) | ✅ +1.5 → +2.4 per episode (was −3) |
| `mimic_evaluate.py` report + plot + `.sto` | ✅ |
| Studio playback route (`deprl_evaluate.py` → `deprl.play`) with the scaled model | ✅ `run_checkpoint_100000/` contains the AB07 `config.scone` (6 modifiers) |

At 1e5 steps the policy falls after about 0.4 s. Its COM speed over that time is 1.16 m/s and the knee collapses (knee RMSE ~57°). This is the expected state before training: the walking run needed about 1e7 steps before it stood up for more than 2 s.

---

## 11. Tuning and troubleshooting

| Symptom | What to try |
|---|---|
| Falls immediately for a long time (episode length < 20 steps after 1e7 steps) | Lower `k_root` (500 → 100) and `k_ee` (80 → 40) so early, rough imitation still earns reward. Raise `init_activations_mean` (0.05 → 0.1–0.2, the model starts with almost slack muscles in mid-stance). Lower `init_pos_noise`. |
| Training return falls to ~0 and stays there | Check that `min_reward` is 0 and the penalties are small. Look at the reward terms with `mimic_evaluate.py`. |
| Stays upright but drifts behind or ahead of the reference | Raise `w_root`. Check that the reference speed (`belt_speed`) is the intended one. |
| Kinematics good, moments or GRF poor | Raise `w_torque` / `w_grf` after the kinematics are learnt (a curriculum: start with `w_torque` = `w_grf` = 0, then raise them and resume). Liu et al. used a separate kinetic phase. |
| Jerky, high-frequency muscle activity | Raise `w_smooth` (0.1 → 0.5) once the policy survives whole episodes. |
| "subtalar" or joint-limit fights | `subtalar_weight` 0.2 → 0. |
| `ModuleNotFoundError: scone_mimic` in workers or Studio | `.venv/lib/python3.9/site-packages/scone_rl_root.pth` must contain the repo path (re-run `./setup.sh`). |
| Segfault when reading `.b3d` | nimblephysics teardown bug: keep objects alive and end with `os._exit(0)` (see §4). |
| New subject: model and data joint centres differ by > 5 cm | A convention problem (sign, pelvis origin, DoF mapping). See §6.2; compare with `scaling.txt` for AB07. |

Episode length and speed settings:
* Longer episodes (`max_episode_steps` 300 → 400) need segments ≥ 10 s plus look-ahead; the AB07 segments are 10 s.
* Faster or slower gaits: rebuild the reference with `SPEED=…` (§12). A policy trained at one speed will not reproduce another person's gait at a different speed.

---

## 12. New subject, other speed, other scaling

```bash
# another Camargo subject (treadmill split 5)
./scripts/mimic_prepare_subject.sh ../Data/AB08_split5/AB08_split5.b3d AB08
# another speed for AB07
SPEED=1.0 ./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07_1p0
# uniform scaling by height instead of measured segments
SCALING=height ./scripts/mimic_prepare_subject.sh ../Data/AB07_split5-001.b3d AB07_h
```
Then copy `configs/deprl/sconemimic_h2190_ab07.yaml`, change `name:` and `reference=` in `environment:`, and train.

**Your own lab data.** Process it with [AddBiomechanics](https://addbiomechanics.org) (upload markers + force plates, download the `.b3d`), then run the same command. A treadmill trial with ≥ 10 s of steady walking and complete force data is ideal. For overground data, change the segment selection in `mimic_extract_b3d.py` (steady strides between start and turn) and drop the belt-travel step.

---

## 13. Known limitations and next steps

* **No long training yet**, so there are no tracking numbers for a trained policy. This is the next step (`./scripts/train_mimic.sh`).
* **No EMG** in the AddBiomechanics files. To compare simulated muscle activations with measured EMG (a strong validation for the licence case), use the original Camargo download (11 EMG channels per leg).
* **Model differences.** H2190 (gait2392-like knee, combined ankle–subtalar body, no toes, no arms) vs the data model (Rajagopal walker knee, separate talus, toes): ~3 cm joint-centre offsets remain, and subtalar ranges differ.
* **Single speed and subject per policy.** Choi et al. fine-tune a base policy per condition; a multi-subject or speed-conditioned policy would add the subject's scale factors and target speed to the observation.
* **Muscle strength is not personalised.** Only geometry and mass are scaled. `MuscleModifier` could scale max isometric force with mass or with measured strength.
* **Effort.** Mean squared activation is used, not a metabolic model (Choi et al. used Umberger / Uchida metabolic rate).

---

## 14. References

* Choi, Park, Halilaj, Kang (2026). *Musculoskeletal Motion Imitation for Learning Personalized Exoskeleton Control Policy in Impaired Gait.* arXiv 2604.09431. H2190 + Hyfydy + SCONE, SAC, tracking reward (source of the reward form, weights and the 0.4 m termination).
* Liu, Ahn, Lobaton, Si, Huang (2026). *Imitation Learning from Human Motion Alone Does Not Guarantee Biomechanically Plausible Gait Kinetics.* arXiv 2603.12408 (GRF / CoP reward terms).
* Park, Choi, Ahn, Ahn (2026). *Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation.* arXiv 2603.10474 (H2190, Camargo validation).
* Simos, Chiappa, Mathis. *KINESIS: Motion Imitation for Human Musculoskeletal Locomotion.* arXiv 2503.14637, ICRA (MyoLeg, PPO, large-scale imitation).
* Peng, Abbeel, Levine, van de Panne (2018). *DeepMimic.* ACM TOG (reference state initialisation, early termination, exp-tracking rewards).
* Camargo, Ramanathan, Flanagan, Young (2021). *A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions.* J. Biomech. 119:110320.
* Werling et al. (2023). *AddBiomechanics: Automating model scaling, inverse kinematics, and inverse dynamics from human motion data through sequential optimization.* PLOS ONE.
* Schumacher, Häufle, Büchler, Schmitt, Martius (2023). *DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems.* ICLR.
* Geijtenbeek (2019). *SCONE: Open Source Software for Predictive Simulation of Biological Motion.* JOSS; Hyfydy (hyfydy.com).
