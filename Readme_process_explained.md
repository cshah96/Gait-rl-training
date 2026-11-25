# How the training works, explained

`README.md` covers *how to run* things. This document explains *what is actually happening*: what is trained, what goes in and out, how the reward is computed, what happens in every cycle, what Hyfydy does, and the basic math behind each part. It covers both pipelines in this repo:

* **PPO** (`scone_ppo/`, the default, formerly the `gait-training` repo), and
* **DEP-RL** (MPO + DEP, `third_party/depRL`, for more complex models).

Everything here comes from the code that runs: `scone_ppo/*`, `../sconegym/sconegym/gaitgym.py`, `third_party/depRL/deprl/*`, the model files in `../sconegym/sconegym/data-v1/`, and the Hyfydy user manual (`../hyfydy-user-manual.pdf`).

---

## Contents

**Part A: The shared basics (both pipelines)**
1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [The big picture in one diagram](#2-the-big-picture-in-one-diagram)
3. [Reinforcement learning in 5 minutes](#3-reinforcement-learning-in-5-minutes)
4. [The simulation stack: Hyfydy → SCONE → SconePy → sconegym](#4-the-simulation-stack-hyfydy--scone--sconepy--sconegym)
5. [The body: the H0918 musculoskeletal model](#5-the-body-the-h0918-musculoskeletal-model)
6. [The role of Hyfydy, and the physics math](#6-the-role-of-hyfydy-and-the-physics-math)
7. [Inputs: the observation (121 numbers)](#7-inputs-the-observation-121-numbers)
8. [Outputs: the action (18 numbers)](#8-outputs-the-action-18-numbers)
9. [sconegym's reward, term by term](#9-sconegyms-reward-term-by-term)
10. [Episodes: start, end, falling](#10-episodes-start-end-falling)

**Part B: PPO (default pipeline)**

11. [What PPO adds to the environment: constant speed + energy](#11-what-ppo-adds-to-the-environment-constant-speed--energy)
12. [The PPO networks](#12-the-ppo-networks)
13. [PPO math: advantages, clipped objective, value loss](#13-ppo-math-advantages-clipped-objective-value-loss)
14. [What happens in every PPO cycle](#14-what-happens-in-every-ppo-cycle)
15. [What PPO learning looks like (measured)](#15-what-ppo-learning-looks-like-measured)

**Part C: DEP-RL (for complex models)**

16. [DEP-RL in one picture](#16-dep-rl-in-one-picture)
17. [MPO: the learner, and its math](#17-mpo-the-learner-and-its-math)
18. [DEP: the exploration controller, and its math](#18-dep-the-exploration-controller-and-its-math)
19. [The adaptive energy cost](#19-the-adaptive-energy-cost)
20. [What happens in every DEP-RL cycle](#20-what-happens-in-every-dep-rl-cycle)
21. [The DEP-RL timeline and logs](#21-the-dep-rl-timeline-and-logs)

**Part D: Choosing and context**

22. [PPO vs DEP-RL: why PPO is enough for H0918, and when to switch](#22-ppo-vs-dep-rl-why-ppo-is-enough-for-h0918-and-when-to-switch)
23. [RL vs SCONE's own optimization](#23-rl-vs-scones-own-optimization)
24. [Glossary](#24-glossary)

---

# Part A: The shared basics

## 1. The problem in one paragraph

We have a 2-D model of a human (pelvis + torso and two legs) driven by **18 Hill-type muscles**. Every control step, a neural network reads the body's state (muscle lengths, forces, joint angles, and so on) and outputs how strongly to excite each of the 18 muscles. The step is 10 ms in PPO and 25 ms in DEP-RL. The physics engine, Hyfydy, applies those excitations; the muscles pull on the bones; the body moves, touches the ground, and possibly falls. Each step is scored with a **reward**: high when the model walks forward at about 1.2 m/s, lower when it uses a lot of muscle activation, hits joint limits, slams into the ground, or jerks its muscles around. **Reinforcement learning** adjusts the network's weights so that the total reward over an episode is as high as possible. The result is a controller that walks without falling while spending little muscle effort. No motion-capture data, reference trajectory or reflex model is used: walking *emerges* because it is the behaviour that collects the most reward.

## 2. The big picture in one diagram

```
                 ┌──────────────────────────── one control step (10 ms PPO / 25 ms DEP-RL) ─────────────────────┐
                 │                                                                                                 │
   observation   │   ┌──────────────┐  18 muscle      ┌───────────────────────────────────────────────┐           │
   s_t (121)  ───┼──►│   POLICY     │  excitations    │  HYFYDY physics engine (inside SCONE/sconepy)  │           │
                 │   │  π_θ(a | s)  ├────a_t─────────►│  excitation → activation → muscle force        │           │
                 │   │  neural net  │                 │  → joint torques → body motion + foot contact  │           │
                 │   └──────────────┘                 └──────────────────────┬────────────────────────┘           │
                 │          ▲                                                │                                    │
                 │          │ learning update                                ▼                                    │
                 │   ┌──────┴───────┐   (s, a, r, s')     new state s_t+1, reward r_t, fell?                      │
                 │   │   LEARNER    │◄──────────────────────────────────────┘                                     │
                 │   │ PPO  or  MPO │   PPO: last 8,192 steps, then discarded                                     │
                 │   └──────────────┘   MPO: replay buffer of the last 1,000,000 steps                            │
                 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
       many copies of the body simulate in parallel: PPO 64 (16 processes × 4), DEP-RL 195 (15 × 13)
```

## 3. Reinforcement learning in 5 minutes

| RL term | In this project |
|---|---|
| **Agent** | the neural-network controller |
| **Environment** | the simulated body + ground (sconegym `GaitGym`; PPO wraps it in `EnergyWalkEnv`) |
| **State / observation** $s$ | 121 numbers describing the body right now (§7) |
| **Action** $a$ | 18 muscle excitations (§8) |
| **Step** | one control interval: 10 ms (PPO) or 25 ms (DEP-RL) of simulated time |
| **Reward** $r$ | one number per step: "how good was that step?" (§9, §11) |
| **Episode** | one attempt: start → fall, or time limit (10 s PPO / 25 s DEP-RL) |
| **Policy** $\pi_\theta(a\mid s)$ | the network: a probability distribution over actions given the state |
| **Return** $G$ | the sum of discounted future rewards, $r_0 + \gamma r_1 + \gamma^2 r_2 + \dots$ |
| **Value** $V(s)$ / $Q(s,a)$ | expected return from a state (or state + action), predicted by a second network, the **critic** |

The loop is:

```
s = env.reset()
repeat:
    a = policy(s)                # network picks muscle excitations (with some randomness to explore)
    s, r, done = env.step(a)     # physics runs one step, we get a reward
    remember (s, a, r)
    if done: s = env.reset()
every so often: update the policy so actions that led to higher-than-expected return become more likely
```

**Discounting.** Both pipelines use $\gamma = 0.99$ per step. In PPO (10 ms steps) a reward 1 s ahead counts $0.99^{100} \approx 37\%$. In DEP-RL (25 ms steps) the same 37 % is reached 2.5 s ahead. So the agent plans about one to two strides ahead.

## 4. The simulation stack: Hyfydy → SCONE → SconePy → sconegym

```
 scone_ppo/  or  depRL          learning algorithm, networks, parallel envs
        │ uses
 sconegym  (GaitGym)            gym API: reset(), step(action) → obs, reward, done
        │ uses
 SconePy   (sconepy module)     Python API: load_model, set_actuator_inputs, advance_simulation_to, muscle_force_array, ...
        │
 SCONE     (C++ framework)      model loading, muscles, sensors, neural delays, state init, results (.sto), SCONE Studio viewer
        │
 Hyfydy    (physics engine)     rigid bodies, joints, contacts, muscle forces, integration
```

SCONE loads the model through the scenario file `../sconegym/sconegym/data-v1/H0918.scone`:
```
ModelHyfydy {
    model_file = H0918v2j.hfd            # the Hyfydy model (bodies, joints, muscles, contacts)
    state_init_file = InitStateGait10.zml # starting pose + velocities (mid-stride at ~1 m/s)
    fixed_control_step_size = 0.005      # muscle inputs are updated every 5 ms internally
    initial_equilibration_activation = 0.05
    zero_velocity_equilibration = 1
    << neural_delays_FEA_v4.zml >>       # sensor delays (used only with use_delayed_sensors)
}
```

Every environment step, sconegym calls:
```python
model.set_actuator_inputs(action)               # 18 excitations
model.advance_simulation_to(time + step_size)   # Hyfydy integrates 10 ms (PPO) or 25 ms (DEP-RL)
model.muscle_force_array(), model.dof_position_array(), model.com_vel(), ...   # build obs and reward
```

## 5. The body: the H0918 musculoskeletal model

File: `../sconegym/sconegym/data-v1/H0918v2j.hfd`. **"H0918"** = **H**uman, **09** degrees of freedom, **18** muscles. It is **planar** (sagittal plane): it can move forward and back, up and down, and pitch, but cannot fall sideways. Bodies: pelvis + torso (head and arms lumped in), and per leg femur, tibia and calcaneus/foot. **Total mass 74.5 kg** (measured).

**9 degrees of freedom**

| DOF | Meaning | Range |
|---|---|---|
| `pelvis_tx`, `pelvis_ty` | forward position, height | free |
| `pelvis_tilt` | trunk pitch | ±90° |
| `hip_flexion_r/l` | hip angle | −30° … 120° |
| `knee_angle_r/l` | knee angle (negative = flexed) | −120° … 10° |
| `ankle_angle_r/l` | ankle angle | −60° … 25° |

**18 muscles (9 per leg)**: the same set as the classic Geyer & Herr (2010) reflex walking model.

| # (r / l) | Muscle | Max force | Crosses | Main function |
|---|---|---|---|---|
| 0 / 9 | `hamstrings` | 2594 N | hip + knee | hip extension + knee flexion |
| 1 / 10 | `bifemsh` | 804 N | knee | knee flexion |
| 2 / 11 | `glut_max` | 1944 N | hip | hip extension (push leg back) |
| 3 / 12 | `iliopsoas` | 2186 N | hip | hip flexion (swing leg forward) |
| 4 / 13 | `rect_fem` | 1169 N | hip + knee | hip flexion + knee extension |
| 5 / 14 | `vasti` | 4530 N | knee | knee extension (support body weight) |
| 6 / 15 | `gastroc` | 2241 N | knee + ankle | plantarflexion (push-off) + knee flexion |
| 7 / 16 | `soleus` | 3549 N | ankle | plantarflexion (push-off, stance) |
| 8 / 17 | `tib_ant` | 1579 N | ankle | dorsiflexion (lift toes in swing) |

Each muscle is defined by a few physiological parameters. For example, `hamstrings_r` has max isometric force $F_{max}$ = 2594 N, optimal fiber length $\ell^M_{opt}$ = 9.76 cm, tendon slack length $\ell^T_s$ = 31.9 cm, and pennation angle 0.20 rad. It also has a path of attachment points on the bones, which sets its length and moment arm at every posture. **Feet** touch the ground through four contact spheres (`heel_r/l`, `toe_r/l`); ground friction is $\mu_s$ = 0.9 static and $\mu_d$ = 0.6 dynamic.

The key point for RL: **the policy never commands joint torques.** It can only excite muscles. The torque at a joint is whatever the muscles crossing it produce, given their current length, velocity, activation and tendon stretch. That makes the problem hard, and also biologically meaningful.

## 6. The role of Hyfydy, and the physics math

**Hyfydy is the physics engine.** On every `advance_simulation_to` call it integrates all of the equations below, from the current state and the 18 excitations. It uses the same muscle and contact models as OpenSim but runs about **100× faster** (manual §1). That is what makes RL feasible: PPO's 20M steps are 55 hours of simulated walking, and DEP-RL's 5e7 steps are 14.5 days. SCONE is the framework around Hyfydy: it loads the model files, exposes `sconepy`, and writes `.sto` result files for SCONE Studio.

### 6.1 Muscle activation dynamics (excitation → activation)

The policy outputs a neural **excitation** $u$ per muscle, like a motor-neuron firing rate. A muscle cannot respond instantly, so the **activation** $a \in [0,1]$ follows a first-order filter:

$$
\frac{da}{dt} = \begin{cases} (u - a)\, r_{act} & u > a \quad\text{(activating, fast)} \\ (u - a)\, r_{deact} & u \le a \quad\text{(deactivating, slower)} \end{cases}
$$

with Hyfydy's defaults $r_{act} = 100\ \text{s}^{-1}$ (≈ 10 ms) and $r_{deact} = 25\ \text{s}^{-1}$ (≈ 40 ms) (manual §4.3). This is the standard form; Hyfydy's exact discrete update is in manual §4.3. Consequence: the policy has to *plan ahead*. A muscle excited now produces force over the next tens of milliseconds, and keeps producing some force after excitation stops.

### 6.2 Muscle force (Hill-type, Millard 2012 equilibrium model)

Hyfydy's `muscle_force_m2012fast` (manual §3.3.1) is a fast implementation of the Millard et al. Hill-type model. A **musculotendon unit** is a contractile fiber in series with an elastic tendon:

```
   bone ──[ tendon (spring) ]──[ muscle fiber: active + passive + damping ]── bone
          length ℓ^T                 length ℓ^M, pennation angle α
```

With normalised fiber length $\tilde\ell = \ell^M/\ell^M_{opt}$ and velocity $\tilde v = \dot\ell^M/(v_{max}\,\ell^M_{opt})$, where $v_{max} = 10\ \ell_{opt}/\text{s}$:

$$
F^M = F_{max}\left[\, a \cdot f_L(\tilde\ell) \cdot f_V(\tilde v) \;+\; f_{PE}(\tilde\ell) \;+\; \beta\,\tilde v \,\right]\cos\alpha
$$

* $f_L$, **active force–length**: a bell curve peaking at $\tilde\ell = 1$ (strongest at optimal length).
* $f_V$, **force–velocity**: force drops when shortening fast and rises when lengthening.
* $f_{PE}$, **passive elastic force**: a stretched muscle resists even when inactive.
* $\beta\tilde v$: small **damping** ($\xi = 0.1$), which keeps the model well-posed at zero activation.
* $\cos\alpha$: projection onto the tendon (pennation).

The **tendon** is a nonlinear spring. With strain $\varepsilon^T = (\ell^T-\ell^T_s)/\ell^T_s$ it produces $F^T = F_{max} f_T(\varepsilon^T)$, which is zero when slack. The equilibrium model enforces $F^T = F^M$. Hyfydy uses polynomial curves instead of OpenSim's splines, which is one reason it is fast. The joint torque is $\tau_j = \sum_m r_{m,j}(q)\,F^T_m$, where $r_{m,j}(q)$ is the posture-dependent moment arm.

### 6.3 Joints, joint limits, contact, gravity

* **Joints.** Every body has a full 6-DOF state, and joints are enforced by very stiff spring–damper forces (`joint_stiffness = 1e6`), not by reduced coordinates as in OpenSim (manual §5.1). Joints therefore have tiny, realistic "give".
* **Joint limits** (e.g. the knee cannot hyperextend past 10°) are non-linear spring–damper torques beyond the limit (`joint_limit_stiffness = 500`). The reward's `constr` term penalises these torques, because leaning on ligaments is "cheating".
* **Foot–ground contact.** Hunt–Crossley contact with penetration depth $d$: normal force $F_n = k\,d^{3/2}(1 + \tfrac32 c\,\dot d)$, with $k$ = 11006 and $c$ = 1 from the material, plus static and dynamic friction.
* **Gravity** acts on every body.

### 6.4 Integration

All of this forms a large ODE system (body states + each muscle's activation and fiber length). Hyfydy advances it with an **error-controlled variable-step integrator** (manual §4). It takes as many internal sub-steps as needed, more at heel strike and fewer in swing. Each RL action is held constant for the whole step (2 internal 5 ms control ticks in PPO, 5 in DEP-RL). To the RL agent, Hyfydy is a black-box function $s_{t+1} = f_{\text{Hyfydy}}(s_t, a_t)$ that is fast, deterministic and physically realistic.

## 7. Inputs: the observation (121 numbers)

Built in `GaitGym._get_obs_2d()` after every step. It is identical for both pipelines.

| Index | Size | Content | Biological analogue |
|---|---|---|---|
| 0–17 | 18 | muscle **fiber length** (normalised, 1.0 = optimal) | muscle spindles |
| 18–35 | 18 | muscle **fiber velocity** | muscle spindles |
| 36–53 | 18 | muscle **force** (normalised to max force) | Golgi tendon organs |
| 54–71 | 18 | muscle **excitation** (last command) | efference copy |
| 72–75 | 4 | **head/torso orientation** (quaternion) | vestibular system |
| 76–78 | 3 | **head/torso angular velocity** | vestibular system |
| 79–84 | 6 | **left and right foot position relative to pelvis** (x, y, z) | proprioception |
| 85–93 | 9 | **DOF positions** (`pelvis_tx` set to 0) | joint receptors |
| 94–102 | 9 | **DOF velocities** (includes forward speed) | joint receptors |
| 103–120 | 18 | muscle **activation** | — |

* `pelvis_tx` is zeroed: being at x = 3 m or 30 m should give the same behaviour.
* **Normalisation.** The scales differ wildly (forces ~0–1, joint velocities ~10 rad/s), so both pipelines z-score every input with running statistics, $\hat s = (s-\bar s)/\sigma_s$. PPO also clips the result to ±10. These statistics are saved in every checkpoint, because a policy only works with the normalisation it was trained with.
* **Delayed sensors** (`use_delayed_sensors=True`, off by default) replace this with a smaller observation of neurally-delayed signals (10 ms for hip muscles, 20 ms vasti/bifemsh, 40 ms gastroc, 100 ms head and pelvis; from `neural_delays_FEA_v4.zml`). This is more realistic and harder.

## 8. Outputs: the action (18 numbers)

* One **excitation per muscle**, in the order of the table in §5.
* During training the policy outputs a **Gaussian**: mean $\mu_\theta(s)$ plus noise with std $\sigma$. An action is *sampled* from it, which is how the agent explores. For evaluation and SCONE Studio playback the **mean** is used, without noise.
* sconegym **clips** each excitation to **[0, 0.5]** (`clip_actions=True`). Muscles can only pull, and the 0.5 cap prevents violent, unphysiological bursts.
* Excitation is **not** force: excitation → (activation dynamics, 10–40 ms) → activation → (force–length–velocity × max force) → muscle force → (moment arms) → joint torques. The agent must learn all of this implicitly.

## 9. sconegym's reward, term by term

Computed every step in `GaitGym._update_rwd_dict()`. Both pipelines use these terms; the coefficients differ slightly:

$$
r_t = 10\cdot R_{vel} \;-\; c_{grf}\max(0, \text{GRF}-1.2) \;-\; 0.1307\,\overline{|\tau_{lim}|} \;-\; 0.097\,\overline{\Delta u^2} \;-\; 1.579\,\tfrac{\#\{a_i>0.15\}}{18}
$$

| Term | Definition | Coefficient | Why |
|---|---|---|---|
| `gaussian_vel` $R_{vel}$ | $e^{-(v-1.2)^2}$ if COM speed $v$ < 1.2 m/s, else 1 (a plateau) | **+10** | walk forward at ≥ 1.2 m/s. *PPO replaces this term (§11).* |
| `grf` | ground reaction force above 1.2 body weights | $c_{grf}$ = 0.073 (PPO) / 0.173 (DEP-RL) | no stomping or hard impacts |
| `constr` | mean \|joint-limit torque\| | 0.1307 | don't hang on joint limits |
| `smooth` | mean $(u_t-u_{t-1})^2$ | 0.097 | smooth, not jittery, muscle commands |
| `number_muscles` | fraction of muscles with activation > 0.15 | 1.579 | use few muscles at once (effort proxy) |
| `self_contact` | contact force on non-foot bodies | 0 for walking | used in the running tasks |

The best possible step reward is about +10. **There is no explicit fall penalty**: falling ends the episode, which forfeits all future reward. The discount factor makes that loss felt.

## 10. Episodes: start, end, falling

**Start (`GaitGym.reset`)**: every episode starts from a slightly different state:
1. Load the **mid-stride** initial state `InitStateGait10.zml`: pelvis moving forward at 1 m/s, one leg in swing.
2. Add Gaussian noise: σ = 0.05 on every DOF position and σ = 0.1 on every DOF velocity.
3. With 50 % probability **swap left and right legs**.
4. Set initial muscle activations. PPO uses sconegym's default, random 0.3 ± 0.1; DEP-RL's config uses a fixed 0.01.
5. Equilibrate the muscles (`init_state_from_dofs`) and lower the body until the feet carry 50 % of body weight (`adjust_state_for_load(0.5)`).

**End**
* **Fall (termination):** pelvis COM < 0.5 m or head < 0.9 m. There is no future value, so this is the implicit penalty.
* **Time limit (truncation):** 1000 steps, which is 10 s (PPO) or 25 s (DEP-RL). This is *not* a failure. Both pipelines **bootstrap** the critic's value of the last state, so the agent isn't taught that running out of time is bad.
* After any end, the environment resets automatically.

---

# Part B: PPO (default pipeline)

## 11. What PPO adds to the environment: constant speed + energy

PPO trains on `EnergyWalkEnv` (`scone_ppo/envs.py`), a thin wrapper around sconegym's `GaitGym`. It keeps sconegym's observation, action, termination and penalty terms, and adds the following.

### 11.1 Constant-speed velocity reward (`--speed_mode constant`, the default)

sconegym's term rewards **at least** 1.2 m/s (flat for any faster speed) and is very wide: 0.2 m/s off still earns 96 %. PPO replaces it with a bell curve that peaks **at** the target:

$$
\bar v \leftarrow \bar v + \frac{\Delta t}{T_{avg}}\,(v - \bar v), \qquad
r_{vel} = 10 \cdot \exp\!\Big(-\big(\tfrac{\bar v - v_{target}}{\sigma}\big)^2\Big),
\quad v_{target}=1.2,\ \sigma=0.25,\ T_{avg}=0.5\ \text{s}
$$

| $\bar v$ (m/s, target 1.0, σ 0.25 example) | 0.5 | 0.8 | 1.0 | 1.2 | 1.5 |
|---|---|---|---|---|---|
| $r_{vel}$ | 0.18 | 5.27 | **10.0** | 5.27 | 0.18 |

* **Averaged speed**: the COM speeds up and slows within each stride, so a ~0.5 s average means natural oscillation isn't punished, only drift away from the target.
* **Blocks the "dive" exploit**: falling forward fast (about 1.4–1.5 m/s) earns almost nothing.
* `--speed_mode min` restores sconegym's plateau.

### 11.2 Energy-efficiency term (always on)

$$
r_{effort} = -\,c_{effort}\sum_{i=1}^{18} w_i\, a_i^2, \qquad w_i = \frac{\text{volume}_i}{\sum_j \text{volume}_j}, \quad c_{effort} = 10
$$

* $a_i$ is the activation of muscle $i$. The muscle volume (from SconePy's `Muscle.volume()`: max force × optimal fiber length / specific tension) makes **big muscles cost more** than small ones (vasti 12.5 %, glut_max 8.5 %, hamstrings 7 % versus bifemsh and rect_fem about 2.5 %), as in real metabolism.
* **Squaring** makes it cheaper to share load across several muscles than to fire one hard, a classic muscle-redundancy result.
* Other options: `--effort_type squared` (mean $a^2$), `cubed` (mean $a^3$, fatigue-like).
* **Trade-off:** if $c_{effort}$ is too large, the cheapest strategy is to relax and fall, which ends all effort costs. If too small, the gait works but wastes energy.
* Logged per episode: `mean_effort` and `effort_per_m` = total effort × Δt / distance, a *cost-of-transport-like* measure that is the fairest way to compare gaits.

**Total PPO step reward** = sconegym terms (with $r_{vel}$ replaced) + $r_{effort}$. Scale check with all muscles at 0.1 excitation (measured): `gaussian_vel` +10.0, `number_muscles` −1.3, `constr` −0.26, `effort` −0.59, total ≈ **7.7**.

### 11.3 PPO's other environment settings

* Control step **10 ms** (sconegym default `step_size = 0.01`), so each action is held for 2 internal 5 ms control ticks.
* Episodes are cut off at **1000 steps = 10 s** (truncation, bootstrapped as in §10).

## 12. The PPO networks

`scone_ppo/networks.py`: two **separate** networks (no shared layers), tanh activations, orthogonal initialisation.

| Network | Maps | Size |
|---|---|---|
| **Actor** (policy) $\pi_\theta$ | 121 → 256 → 256 → **18 means**, plus a learned, state-independent vector of 18 log-stds | ≈ 102 k parameters |
| **Critic** $V_\phi$ | 121 → 256 → 256 → **1** (expected discounted return) | ≈ 97 k parameters |

* The actor's last layer starts with tiny weights and **bias 0.1**, so every muscle starts at a gentle excitation of about 0.1. The exploration std starts at $e^{-1.5} \approx 0.22$ (`init_log_std`). Early actions are therefore *mild variations around a relaxed posture*, not violent flailing.
* Action sent to the env: $\text{clip}(\mu_\theta(s) + \sigma\,\varepsilon,\ 0,\ 1)$ with $\varepsilon\sim\mathcal N(0,I)$; sconegym then clips to [0, 0.5]. PPO stores the *raw* unclipped sample, so log-probabilities stay exact.
* A PPO checkpoint (`step_N.pt`, `best.pt`, `latest.pt`) contains the network weights, the observation-normaliser statistics, and the full config.

## 13. PPO math: advantages, clipped objective, value loss

PPO (Proximal Policy Optimization, Schulman et al. 2017) is **on-policy**. It collects a batch with the current policy, improves the policy a little on that batch, throws the batch away, and repeats. Code: `scone_ppo/ppo.py`.

### 13.1 Judge each action: advantages with GAE

*Was this action better or worse than the critic expected?*

$$
\delta_t = r_t + \gamma\,V(s_{t+1})(1-d_t) - V(s_t)
\qquad\qquad
A_t = \delta_t + \gamma\lambda\,(1-d_t)\,A_{t+1}
$$

* $\delta_t$ is the one-step "surprise". $A_t$ (Generalised Advantage Estimation) blends 1-step to ∞-step surprises with $\lambda = 0.95$, trading bias against variance.
* $A_t > 0$: better than expected → make that action more likely. $A_t < 0$: make it less likely.
* $\hat R_t = A_t + V(s_t)$ is the critic's regression target.
* **Reward scaling** (`RewardScaler`): rewards are divided by the running std of the discounted return, keeping targets around ±1–10 without changing which policy is optimal.

### 13.2 Improve the policy, but not too much: the clipped objective

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
\qquad
L^{CLIP}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big],\quad \epsilon = 0.2
$$

**The clip is PPO's key idea.** Once an action's probability has changed by more than ±20 %, there is no more gradient pushing it further. Every update is therefore small and safe; big policy jumps are what make plain policy-gradient methods collapse. Advantages are normalised per minibatch.

### 13.3 Critic loss and the full update

$$
L^{V}(\phi) = \tfrac12\,\mathbb{E}_t\Big[\max\big((V_\phi(s_t)-\hat R_t)^2,\ (V^{clip}_\phi(s_t)-\hat R_t)^2\big)\Big]
\qquad
L = L^{CLIP} + 0.5\,L^{V}
$$

The value prediction is clipped around the old value, like the policy. There is no entropy bonus (`ent_coef = 0`); the learned std handles exploration.

* The 8,192 transitions are reused for **up to 10 epochs** in shuffled minibatches of **2,048** (4 per epoch). That is at most 40 Adam steps per batch.
* **KL early stop:** if the approximate KL divergence between old and new policy exceeds 0.03, the remaining epochs are skipped.
* **Gradient clipping** (norm 0.5) and **learning-rate annealing** (3e-4 → 0 linearly over the run).

### 13.4 PPO hyperparameters (`scone_ppo/config.py`)

| Name | Default | Meaning |
|---|---|---|
| `num_workers × envs_per_worker` | 16 × 4 = 64 | parallel simulations |
| `n_steps` | 128 | steps per env per update (1.28 s of sim) |
| batch | 8,192 | transitions per update |
| `n_epochs`, `minibatch_size` | 10, 2048 | reuse of each batch |
| `learning_rate` | 3e-4, annealed | Adam step size |
| `gamma`, `gae_lambda` | 0.99, 0.95 | discount, advantage smoothing |
| `clip_coef`, `target_kl` | 0.2, 0.03 | PPO clip range, early-stop threshold |
| `hidden_size`, `init_log_std`, `init_action_mean` | 256, −1.5, 0.1 | network width, initial noise, initial excitation |
| `speed_mode`, `target_speed`, `speed_sigma`, `speed_avg_time` | constant, 1.2, 0.25, 0.5 | speed reward (§11.1) |
| `effort_type`, `effort_coeff` | volume_squared, 10 | energy term (§11.2) |
| `total_steps`, `save_interval` | 20M, 25 updates | run length, snapshot interval |

## 14. What happens in every PPO cycle

```
for update in 1 .. 2441:                                         (20M steps / 8,192)
  ┌ ROLLOUT (~0.7 s wall time) ──────────────────────────────────────────────────────────┐
  │ repeat 128 times:                                                                    │
  │   obs (64×121) → normalise → actor → sample 64×18 excitations                        │
  │   send to 16 worker processes → each steps its 4 Hyfydy models by 10 ms              │
  │   ← new obs, rewards (velocity + penalties − effort), done flags                     │
  │   scale rewards; bootstrap truncated episodes; store in RolloutBuffer                │
  │   finished episodes → auto-reset, episode stats recorded                             │
  └──────────────────────────────────────────────────────────────────────────────────────┘
  ┌ LEARN (~0.1–0.5 s) ──────────────────────────────────────────────────────────────────┐
  │ critic values → GAE advantages & returns                                             │
  │ ≤ 10 epochs × 4 minibatches of clipped-PPO + value-loss gradient steps (Adam)        │
  └──────────────────────────────────────────────────────────────────────────────────────┘
  log a row to progress.csv; save latest.pt (+ best.pt if improved; checkpoints/step_N.pt every 25 updates)
```

**Where the time goes.** The rollout, which is mostly Hyfydy simulation in the 16 workers, dominates. The learning step is small: at most 40 gradient steps per 8,192 environment steps. That is why PPO here runs at about **7,000–9,500 steps/s on the CPU alone**: 70–95 seconds of simulated walking per real second.

## 15. What PPO learning looks like (measured)

From the original 20M-step run (`results/ppo/20261105_101425_const_1p2_full/progress.csv`):

| Env steps | Wall-clock | Avg. episode length (max 1000) | Falls | Speed | What is happening |
|---|---|---|---|---|---|
| 0 – 1M | 0 – 2 min | ~65–68 (0.7 s) | 100 % | 1.29 m/s (falling forward) | random twitching around a relaxed posture; the critic learns first (explained variance → 0.97) |
| 1M – 11M | 2 – 22 min | 68 → 170 | 100 % | ~1.21–1.26 | learns to brace, delay the fall, take the first steps |
| **11M – 13M** | **22 – 28 min** | **170 → 850** | **100 % → 20 %** | 1.21 | the breakthrough: sustained stepping appears |
| 13M – 16M | 28 – 36 min | ~930–1000 | 4–12 % | 1.20 ± 0.03 | reliable walking at the target speed; `best.pt` at 15.6M (no falls in the last 50 episodes) |
| 16M – 20M | 36 – 49 min | ~880–910 | ~10 % | 1.22 ± 0.04 | refinement; effort drifts down (0.040 → 0.035) as the learning rate anneals |

Training-time numbers include exploration noise. With the deterministic mean policy, the final checkpoint `latest.pt` walks **19/20 episodes** for the full 10 s at ~1.2 m/s, and **4/4 episodes of 60 s** (~71 m) without falling, six times longer than any training episode. The gait is stable, not just surviving the time limit.

**The second run (2026-11-18, on this machine, `results/ppo/20261118_114942_const_1p2/`)** used identical settings. It took **39 min 22 s** (avg ~8,500 steps/s). Its breakthrough came later, at **13.1M → 13.8M steps** (`len` 204 → 768). Its final checkpoint also walks **19/20** episodes at 1.21 m/s (speed error 0.02–0.03), and 2 of 3 starts walked 60 s / 72.5 m. Its `best.pt` (update 2370) walks only 10/20: "best" is chosen from a noisy training average, so **always evaluate the final checkpoint**. The remaining ~5 % of falls happen in the first 2–3 s, from the few randomised start states the policy cannot recover from. Details are in `docs/VERIFICATION.md`.

Note that an early 5M-step validation run (`results/ppo/20261104_190009_validation/`, curves in `docs/ppo_validation_5M.png`, local only) never walked. In a 5M-step run the learning rate anneals to 0 before the breakthrough at about 11M steps, so short runs are not just "less trained"; they stop too early.

---

# Part C: DEP-RL (for complex models)

## 16. DEP-RL in one picture

**DEP-RL** = **D**ifferential **E**xtrinsic **P**lasticity + **R**einforcement **L**earning (Schumacher et al., ICLR 2023). It has two parts that take turns controlling the body during training:

* **MPO** (the "student"): an off-policy actor-critic. It learns from a **replay buffer** of the last 1,000,000 transitions, so data is reused many times. This is the part that becomes your controller.
* **DEP** (the "playful toddler"): a small controller with **no reward and no goal** that amplifies whatever movement the body is already making. This produces **large, coordinated, rhythmic** muscle activity instead of random twitching.

Random exploration noise on every muscle mostly makes flexors and extensors fight each other, so the body barely moves and the learner sees nothing worth learning. The more muscles, the worse this gets. DEP's coordinated movements give the critic informative data. **After training, DEP is thrown away**; the final controller is purely the MPO policy network.

DEP-RL environment settings (`configs/deprl/sconewalk_h0918_v1.yaml → env_args`): control step **25 ms**, episodes up to 1000 steps = **25 s**, sconegym's plateau speed reward (≥ 1.2 m/s), GRF coefficient 0.173, initial activations fixed at 0.01.

## 17. MPO: the learner, and its math

MPO (Maximum a-posteriori Policy Optimisation, Abdolmaleki et al. 2018), implemented in depRL as `custom_mpo_torch.TunedMPO` on the TonicRL library.

**Networks.** The actor is an MLP 121 → 256 → 256 → (18 means, 18 stds), ReLU, Gaussian policy, ≈ 106 k parameters. The critic $Q_\phi(s,a)$ is an MLP (121 + 18) → 256 → 256 → 1, ≈ 102 k parameters. Each has a slowly-tracking **target copy** ($\phi' \leftarrow 0.995\phi' + 0.005\phi$ after every update). Observations are z-scored by a running normaliser. The H2190 config uses 1024-wide networks.

**Critic (Expected SARSA).** For replay transitions $(s,a,r,s',d)$:

$$
y = r + \gamma(1-d)\,\frac1N\sum_{j=1}^{N} Q_{\phi'}(s', a'_j),\quad a'_j\sim\pi_{\theta'}(\cdot\mid s'),\ N=20
\qquad
\mathcal L_{critic} = \mathbb E\big[(Q_\phi(s,a)-y)^2\big]
$$

**Actor, E-step: "which of my candidate actions were good?"** For each state, sample 20 actions from the target policy and weight them by a softmax of their Q-values with temperature $\eta$:

$$
q(a_j\mid s) = \frac{\exp(Q_\phi(s,a_j)/\eta)}{\sum_k \exp(Q_\phi(s,a_k)/\eta)},
\qquad
\min_\eta\ \eta\varepsilon + \eta\,\mathbb E_s\Big[\log\tfrac1N\textstyle\sum_j e^{Q_\phi(s,a_j)/\eta}\Big],\ \varepsilon = 0.1
$$

**Actor, M-step: "move towards the good actions, a little."** Weighted maximum likelihood with trust regions on the Gaussian's mean and std:

$$
\max_\theta\ \mathbb E_s\sum_j q(a_j\mid s)\log\pi_\theta(a_j\mid s)
\quad\text{s.t.}\quad \mathrm{KL}_\mu \le 10^{-3},\ \ \mathrm{KL}_\sigma \le 10^{-6}
$$

The constraints are enforced with learned Lagrange multipliers. An MO-MPO "action penalisation" term keeps the means inside the valid range. Adam learning rates are 1e-3 (critic), 3e-4 (actor) and 1e-2 (dual variables).

**Data reuse.** Every 1,000 environment steps the learner does **30 updates** on random batches of 256, so each transition is reused about 7.7 times. MPO is designed to squeeze much more learning out of each sample than PPO, but each update is far more compute (20 sampled actions per state, several networks).

**Intuition.** MPO is a careful coach. It tries 20 variations of what the policy would do, asks the critic which looked best, and nudges the policy towards them, never changing behaviour too much at once.

## 18. DEP: the exploration controller, and its math

**Input.** One number per muscle: the muscle length, normalised by its running min/max to about [−1, 1], then lightly smoothed (`s4avg = 2`):

$$
x_i = 2\,\frac{\ell_i - \ell^{min}_i}{\ell^{max}_i - \ell^{min}_i + 0.1} - 1,
\qquad \bar x_t = \bar x_{t-1} + \tfrac12(x_t - \bar x_{t-1})
$$

**Learning rule** (it re-learns every single step from the last τ = 40 steps):

$$
C_t = \sum_{s=2}^{\tau}\big(M\,\Delta x_{t-s}\big)\big(\Delta x_{t-s-\theta}\big)^{\!\top},
\qquad \Delta x_t = \bar x_t - \bar x_{t-1},\ \ \theta = 5,\ \ M = -I
$$

$C$ (18 motors × 18 sensors) correlates how the sensors change *now* with how they changed θ = 5 steps (125 ms) earlier. $M = -I$ is a trivial inverse model: exciting muscle $i$ shortens muscle $i$. $C$ is then normalised row-wise to gain κ = 1000, which keeps the controller at the edge between dying out and exploding.

**Action:**

$$
y_t = \tanh\big(\kappa\cdot \widehat{C_t\bar x_t} + h_t\big),\qquad h \leftarrow h - \mathrm{clip}(0.002\,y,\pm0.05) - 0.001\,h
$$

Here $\widehat{\cdot}$ is an L2 normalisation and $h$ is a slow bias that stops muscles staying saturated. **Intuition:** if a leg starts swinging, the current change resembles the change 125 ms ago, $C$ strengthens that pattern, and the swing grows into a sustained oscillation shaped by the body's own mechanics.

**Who controls the muscles when** (`dep_factory(3, …)`, "stochastic switch"):

| Phase | Controller |
|---|---|
| Warm-up: first 2e5 steps | **DEP only.** It fills the replay buffer with rich movement; no gradient updates yet. |
| Normal training episodes | The **policy** (sampled). At every step, with probability 0.00371, DEP takes over for **8 steps (200 ms)**, about once every 7 s of simulated time. DEP keeps learning in the background. |
| Every 3rd episode | **Policy only**, no DEP ("greedy"), for clean on-policy data |
| Test episodes, evaluation, SCONE Studio | **Policy mean only**; DEP is never used |

## 19. The adaptive energy cost

The replay buffer `AdaptiveEnergyBuffer` *relabels* every sampled reward with an effort cost:

$$
r' = r - c\sum_{i=1}^{18} a_i^3
$$

The coefficient $c$ acts as a curriculum. After each episode the running return is updated: $\bar R \leftarrow 0.8\bar R + 0.2R$. While $\bar R$ < 1000 (cannot walk yet), $c$ decreases towards 0. Once $\bar R$ > 1000, $c$ increases each episode, pushing the gait to be more economical. Its step size shrinks after a performance collapse. You can watch it in `log.csv` as `train/energy_buffer/action_cost_intern`.

Compare PPO (§11.2): there the energy term is **always on** with a fixed coefficient, and uses squared, volume-weighted activations.

## 20. What happens in every DEP-RL cycle

```
one cycle = one 25 ms step for all 195 environments (15 processes × 13 envs)
 1. main process: observations (195×121) + muscle lengths (195×18)
 2. actions: warm-up → DEP;  otherwise policy sample (or DEP inside an 8-step intervention)
 3. workers: clip → Hyfydy integrates 25 ms → reward, obs, done → auto-reset
 4. store 195 transitions in the replay buffer;  steps += 195
 every 1,000 steps (~5 cycles)  LEARN: 30 × [sample 256 → energy relabel → critic step → MPO E/M-step → target update]
 every 200,000 steps            EPOCH: deterministic test episode → one row in log.csv
 every 1,000,000 steps          CHECKPOINT: checkpoints/step_N.pt (+ resume state: replay buffer, optimisers)
```

**Where the time goes (measured here):** steps 1–4 alone run at about 8,600 steps/s. With the LEARN block, throughput drops to about **2,600 steps/s**; the 30 gradient updates (~9 ms each on the GPU) take about 70 % of wall-clock time. The networks are tiny, so the GPU is latency-bound, not busy.

## 21. The DEP-RL timeline and logs

| Steps | Wall-clock here | Typical behaviour |
|---|---|---|
| 0 – 2e5 | ~25 s | DEP-only warm-up: coordinated flailing, many falls |
| 2e5 – 1e7 | ~1 h | MPO learns not to fall immediately |
| 1e7 – 5e7 | ~1–5.5 h | stepping emerges, then a gait. The **published DEP-RL baseline was trained for 5e7 steps.** |
| 5e7 – 5e8 | up to ~53 h | refinement; the energy cost rises, and the gait gets smoother and more efficient |

These are typical ranges from upstream, not yet measured on this machine. Key `log.csv` columns: `test/episode_length/mean` (→ 1000), `test/episode_score/mean`, `test/rwd_metrics/*` (each reward term), `train/energy_buffer/action_cost_intern` (the coefficient $c$), `critic/q`, `train/steps_per_second`.

---

# Part D: Choosing and context

## 22. PPO vs DEP-RL: why PPO is enough for H0918, and when to switch

| | **PPO** (`scone_ppo`) | **DEP-RL** (MPO + DEP) |
|---|---|---|
| Learning style | on-policy: use each batch for ≤ 10 epochs, then discard | off-policy: 1M-transition replay buffer, reused ~7.7× |
| Exploration | Gaussian noise on each muscle (small, σ ≈ 0.22) | DEP's coordinated movements + Gaussian noise |
| Compute per 1,000 env steps | ~5 gradient steps | 30 gradient steps × 20 sampled actions each |
| Throughput here | ~7,000–9,500 steps/s (CPU) | ~2,600 steps/s (CPU sim + GPU learner) |
| H0918 time to walk | **~28 min** (measured, 13M steps) | hours (published baseline: 5e7 steps ≈ 5.5 h here) |
| Code | ~300 lines, readable | research library |

**Why PPO's plain noise suffices for H0918:**
1. **Small model.** 2-D, 18 muscles, cannot fall sideways. DEP's advantage grows with the number of muscles.
2. **Episodes start mid-stride**, moving forward at 1 m/s. The agent only has to learn to *continue* walking, not to discover it from standing.
3. **Dense reward.** The velocity term rewards forward motion every step.
4. **Gentle initial policy.** Excitation starts at 0.1 with small noise, so early behaviour is mild variation around a sensible posture.

**Switch to DEP-RL when** exploring gets hard: 3-D models (H1622 with 22 muscles, **H2190 with 90 muscles**), starting from standing, running, delayed sensors, or when you want to compare against published DEP-RL results. Configs are ready in `configs/deprl/` (README §8–9).

## 23. RL vs SCONE's own optimization

SCONE's tutorials (e.g. *Tutorial 4a – Gait*) solve the same walking problem differently:

| | SCONE optimization (Studio) | This repo (RL) |
|---|---|---|
| Controller | Geyer–Herr **reflex** model: stimulation = f(delayed length, force, trunk angle), ~100 gains | neural network, ~100 k weights |
| What is optimised | the reflex gains | the network weights |
| Optimiser | **CMA-ES**: scores whole 10–20 s simulations | **PPO / MPO**: learn from every step |
| Objective | `CompositeMeasure` (GaitMeasure + EffortMeasure + DofMeasure), once per simulation | per-step reward (velocity + penalties + effort) |
| Strength | physiological, interpretable, few parameters | flexible: any sensors, any task, can adapt to perturbations |

Both run on the same Hyfydy model and both produce `.sto` files for SCONE Studio. The concepts map directly: SCONE's `GaitMeasure` ≈ the velocity term + fall termination, and `EffortMeasure` ≈ the effort term.

## 24. Glossary

| Term | Meaning |
|---|---|
| **Excitation** $u$ | neural command to a muscle (0–1). This is the action. |
| **Activation** $a$ | the muscle's actual contractile state; lags excitation by 10–40 ms |
| **Hill-type muscle** | contractile element (force–length–velocity) + passive elastic element + series tendon |
| **DOF / COM / GRF** | degree of freedom / centre of mass / ground reaction force (in body weights) |
| **Step / episode** | one control interval (10 ms PPO, 25 ms DEP-RL) / one attempt until fall or time limit |
| **Termination / truncation** | episode ends because of a fall / because of the time limit |
| **Policy (actor) / critic** | network producing actions / network predicting future reward |
| **Advantage, GAE** | how much better an action was than expected; PPO's smoothed multi-step estimate |
| **Clipped objective** | PPO's rule limiting how far one update can move the policy |
| **Replay buffer** | MPO's memory of the last 1e6 transitions |
| **Target network** | slowly-updated network copy giving stable learning targets (MPO) |
| **DEP** | Differential Extrinsic Plasticity, the self-organising exploration controller |
| **KL divergence** | how much the policy changed in an update |
| **Explained variance** | how well the critic predicts returns (1 = perfect) |
| **Cost of transport** / `effort_per_m` | effort per distance; lower = more economical |
| **Checkpoint** | saved weights + normaliser (`step_N.pt`, `best.pt`, `latest.pt`) |
| **`.sto` file** | SCONE storage file: a recorded simulation that SCONE Studio plays back in 3-D |
| **CMA-ES** | the evolutionary optimiser SCONE uses for its own reflex controllers |

### References

* J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* 2017. J. Schulman et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
* P. Schumacher, D. Häufle, D. Büchler, S. Schmitt, G. Martius. *DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems.* ICLR 2023.
* A. Abdolmaleki et al. *Maximum a Posteriori Policy Optimisation.* ICLR 2018. R. Der, G. Martius. *Novel plasticity rule can explain the development of sensorimotor intelligence.* PNAS 2015.
* M. Millard, T. Uchida, A. Seth, S. Delp. *Flexing computational muscle.* J. Biomech. Eng. 2013. K. H. Hunt, F. R. E. Crossley. J. Appl. Mech. 1975.
* H. Geyer, H. Herr. *A muscle-reflex model that encodes principles of legged mechanics.* IEEE TNSRE 2010.
* T. Geijtenbeek. *SCONE: Open Source Software for Predictive Simulation of Biological Motion.* JOSS 2019. Hyfydy user manual (`../hyfydy-user-manual.pdf`).
