"""Gait-imitation environment: H2190 muscle model tracks a measured reference gait.

Built on sconegym's GaitGym (same Hyfydy model loading, muscle observations, fall detection and
.sto writing as the walking envs), with:

* Reference State Initialisation: every episode starts at a random frame of the reference motion
  (pose + velocities of a real subject), as in DeepMimic (Peng et al. 2018).
* a tracking reward in the form of Choi et al. 2026 (arXiv 2604.09431), plus a ground reaction
  force term (Liu et al. 2026, arXiv 2603.12408: kinematics alone does not give realistic kinetics):

      r = w_pos    * exp(-k_pos    * sum_j  c_j (q_j  - q*_j)^2)            joint angles
        + w_vel    * exp(-k_vel    * sum_j  c_j (dq_j - dq*_j)^2)           joint velocities
        + w_root   * exp(-k_root   * |p_pelvis - p*_pelvis|^2)              pelvis position
        + w_ee     * exp(-k_ee     * sum_e |(p_e - p_pelvis) - (p*_e - p*_pelvis)|^2)  feet, torso
        + w_torque * exp(-k_torque * sum_j ((tau_muscle_j - tau*_j) / m)^2) hip / knee / ankle moments
        + w_grf    * exp(-k_grf    * sum_f |(F_f - F*_f) / (m g)|^2)        per-foot ground force
        + w_cop    * mean_f exp(-k_cop * |(c_f - p_calcn_f) - (c*_f - p*_calcn_f)|^2)  centre of pressure
                                                                    (horizontal, relative to the foot)
        - w_effort * mean(a^2)                                               muscle effort
        - w_smooth * mean((e_t - e_{t-1})^2)                                 excitation smoothness
      r = max(r, min_reward)          (min_reward = 0: falling early can never pay off)

  Kinetics options (defaults = the original AB07 run; the "kinetic" fine-tune config switches them):
      torque_mode = "per_joint":  w_torque * mean_j exp(-k_torque * ((tau_j - tau*_j)/m)^2)
                                  (one noisy joint can no longer zero the whole term)
      grf_mode    = "per_foot":   w_grf * mean_f exp(-k_grf * |(F_f - F*_f)/(m g)|^2)
      tau_lowpass_hz > 0:         reference moments low-pass filtered (4th-order Butterworth,
                                  zero-lag, per segment) to remove inverse-dynamics spikes
      w_cop > 0:                  centre-of-pressure term (Liu et al. 2026), on feet that are in
                                  stance in the reference (a foot in the air in the simulation scores 0)

  (* = reference). Every positive term is in [0, w], so the per-step reward is in [0, 1.1]
  with the defaults.
* early termination: fall (sconegym), pelvis more than `max_root_dev` m away from the reference,
  end of the reference segment, or `max_episode_steps`.
* observation = sconegym 3-D observation (muscle lengths / velocities / forces / excitations,
  head orientation, feet, DoF positions / velocities, activations)
  + reference deltas q*(t + k*step) - q(t) for the next `n_future` control steps (all 21 DoFs)
  + fraction of the episode time left (so the critic knows when the episode will end).

All coefficients are attributes, so a deprl config can override them through `env_args`.
"""
import os

import numpy as np

from sconegym.gaitgym import GaitGym
from sconetools import sconepy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MIMIC_ARGS = dict(
    # reward weights (Choi et al. 2026 base policy, + GRF term)
    w_pos=0.25, w_vel=0.10, w_root=0.15, w_ee=0.25, w_torque=0.25, w_grf=0.10,
    w_effort=0.05, w_smooth=0.1, w_cop=0.0,
    # the per-step reward is clipped to >= min_reward. With early termination a negative step
    # reward would teach the policy to fall on purpose (DEP's exploration makes the smoothness
    # penalty large), so like DeepMimic every step is worth >= 0.
    min_reward=0.0,
    # reward sharpness (larger = stricter)
    k_pos=2.0, k_vel=0.05, k_root=500.0, k_ee=80.0, k_torque=2.0, k_grf=5.0, k_cop=500.0,
    # kinetics options (see module docstring)
    torque_mode="sum", grf_mode="sum", tau_lowpass_hz=0.0,
    cop_min_force=0.1,              # body weights: below this a foot does not count as loaded
    # per-DoF weight inside the pose / velocity terms; subtalar is often outside the model range
    subtalar_weight=0.2,
    # episodes
    max_episode_steps=300,          # 300 x 0.025 s = 7.5 s
    max_root_dev=0.4,               # m, early termination (Choi et al. 2026)
    n_future=5,                     # reference frames in the observation
    init_pos_noise=0.02,            # rad / m, added to the reference start pose
    init_vel_noise=0.05,
)
TORQUE_DOFS = ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle"]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


class Reference:
    """Reference motion from scripts/mimic_build_subject.py (<subject>_reference.npz)."""

    def __init__(self, path):
        r = np.load(resolve(path), allow_pickle=False)
        self.path = resolve(path)
        self.dof_names = [str(x) for x in r["dof_names"]]
        self.model_file = resolve(str(r["model_file"]))
        self.mass = float(r["mass"])
        self.dt = float(r["dt"])
        self.q, self.dq = r["q"], r["dq"]
        self.tau_clipped = r["tau_clipped"]
        self.tau = self.tau_clipped
        self.tau_filter_hz = 0.0
        self.cop = r["cop"]                     # (N, 2, 3) world, model frame
        self.grf = r["grf"]                     # (N, 2, 3) order calcn_r, calcn_l
        self.contact = r["contact"]
        self.ee_names = [str(x) for x in r["ee_names"]]
        self.ee = r["ee_pos"]                   # (N, n_ee, 3) world, model-consistent
        self.seg_start, self.seg_len = r["seg_start"], r["seg_len"]
        self.belt_speed = r["belt_speed"]

    def set_tau_lowpass(self, hz):
        """Reference moments = clipped inverse-dynamics moments, low-pass filtered per segment."""
        hz = float(hz or 0.0)
        if hz == self.tau_filter_hz:
            return
        self.tau_filter_hz = hz
        if hz <= 0:
            self.tau = self.tau_clipped
            return
        from scipy.signal import butter, filtfilt
        b, a = butter(4, hz / (0.5 / self.dt))
        out = self.tau_clipped.copy()
        for s0, n in zip(self.seg_start, self.seg_len):
            out[s0:s0 + n] = filtfilt(b, a, self.tau_clipped[s0:s0 + n], axis=0)
        self.tau = out

    def segment_of(self, i):
        k = np.searchsorted(self.seg_start, i, side="right") - 1
        return int(self.seg_start[k]), int(self.seg_start[k] + self.seg_len[k])


class SconeMimicGym(GaitGym):
    """Imitation environment (the class name must contain 'scone' so deprl uses its SconeWrapper)."""

    def __init__(self, reference="data/mimic/AB07/AB07_reference.npz", *args, **kwargs):
        for k, v in DEFAULT_MIMIC_ARGS.items():
            setattr(self, k, v)
        self.ref = Reference(reference)
        super().__init__(self.ref.model_file, *args, **kwargs)
        self.step_size = 0.025
        self._max_episode_steps = self.max_episode_steps
        model_dofs = [d.name() for d in self.model.dofs()]
        if model_dofs != self.ref.dof_names:
            raise ValueError(f"reference DoFs {self.ref.dof_names} != model DoFs {model_dofs}")
        self.dofs = list(self.model.dofs())
        self.bodies = {b.name(): b for b in self.model.bodies()}
        self.pelvis = self.bodies["pelvis"]
        self.ee_bodies = [self.bodies[n] for n in self.ref.ee_names]
        self.i_root = [model_dofs.index(n) for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
        self.i_track = [i for i, n in enumerate(model_dofs) if i not in self.i_root]
        self.i_torque = [i for i, n in enumerate(model_dofs) if n.rsplit("_", 1)[0] in TORQUE_DOFS]
        self.track_w = np.array([self.subtalar_weight if model_dofs[i].startswith("subtalar") else 1.0
                                 for i in self.i_track])
        legs = {lg.name(): lg for lg in self.model.legs()}
        self.feet_legs = [legs[n] for n in sorted(legs, key=lambda n: 0 if n.endswith("r") else 1)]
        assert self.ref.ee_names[:2] == ["calcn_r", "calcn_l"]
        self.weight = self.mass * 9.81
        self.ref_start = 0
        self.ref_end = 0
        self.prev_excs = self.model.muscle_excitation_array()
        self._setup_action_observation_spaces()   # observation size includes the reference part

    # ------------------------------------------------------------------ helpers
    def apply_args(self):
        self._max_episode_steps = int(self.max_episode_steps)

    def _ref_index(self, t=None):
        t = self.model.time() if t is None else t
        return min(self.ref_start + int(round(t / self.ref.dt)), self.ref_end - 1)

    @property
    def horizon(self):
        return int(self.max_episode_steps)

    # ------------------------------------------------------------------ reset (RSI)
    def reset(self, *, seed=None, return_info=False, options=None):
        self.model.reset()
        self.has_reset = True
        self.time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.fall_time = -1.0
        self.model.set_store_data(self.store_next)
        self.ref.set_tau_lowpass(self.tau_lowpass_hz)   # env_args are applied after __init__

        n_ep = int(round(self.max_episode_steps * self.step_size / self.ref.dt))
        k = np.random.randint(len(self.ref.seg_start))
        s0, L = int(self.ref.seg_start[k]), int(self.ref.seg_len[k])
        latest = max(s0, s0 + L - n_ep - (self.n_future + 1) * 5)   # leave room for a full episode
        self.ref_start = np.random.randint(s0, latest + 1)
        self.ref_end = s0 + L

        q = self.ref.q[self.ref_start] + np.random.normal(0, self.init_pos_noise, len(self.dofs))
        dq = self.ref.dq[self.ref_start] + np.random.normal(0, self.init_vel_noise, len(self.dofs))
        self.model.set_dof_positions(q)
        self.model.set_dof_velocities(dq)
        acts = np.full(len(self.model.muscles()), self.init_activations_mean)
        self.model.init_muscle_activations(acts)
        self.model.init_state_from_dofs()
        self.prev_acts = acts
        self.prev_excs = self.model.muscle_excitation_array()
        obs = self._get_obs()
        return (obs, (obs, {})) if return_info else obs

    # ------------------------------------------------------------------ observation
    def _get_obs(self):
        base = self._get_obs_3d()
        if not hasattr(self, "ref") or not hasattr(self, "dofs"):   # during GaitGym.__init__
            return base
        q = self.model.dof_position_array()
        i0 = self._ref_index()
        step = int(round(self.step_size / self.ref.dt))
        fut = [self.ref.q[min(i0 + (k + 1) * step, self.ref_end - 1)] - q for k in range(self.n_future)]
        t_left = 1.0 - self.steps / max(1, self.max_episode_steps)
        return np.concatenate([base, np.concatenate(fut), [t_left]]).astype(np.float32)

    # ------------------------------------------------------------------ reward
    def _update_rwd_dict(self):
        i = self._ref_index()
        R = self.ref
        q = self.model.dof_position_array()
        dq = self.model.dof_velocity_array()
        it = self.i_track
        e_pos = np.sum(self.track_w * (q[it] - R.q[i, it]) ** 2)
        e_vel = np.sum(self.track_w * (dq[it] - R.dq[i, it]) ** 2)
        e_root = np.sum((q[self.i_root] - R.q[i, self.i_root]) ** 2)

        p_pel = np.array(self.pelvis.pos().array())
        p_pel_ref = R.q[i, self.i_root]
        ee = np.array([b.pos().array() for b in self.ee_bodies]) - p_pel
        ee_ref = R.ee[i] - p_pel_ref
        e_ee = np.sum((ee - ee_ref) ** 2)

        tau = np.array([self.dofs[j].muscle_moment() for j in self.i_torque])
        e_tau_j = ((tau - R.tau[i, self.i_torque]) / self.mass) ** 2
        if self.torque_mode == "per_joint":
            s_torque = np.mean(np.exp(-self.k_torque * e_tau_j))
        else:
            s_torque = np.exp(-self.k_torque * np.sum(e_tau_j))

        grf = np.array([lg.contact_force().array() for lg in self.feet_legs])
        e_grf_f = np.sum(((grf - R.grf[i]) / self.weight) ** 2, axis=1)
        if self.grf_mode == "per_foot":
            s_grf = np.mean(np.exp(-self.k_grf * e_grf_f))
        else:
            s_grf = np.exp(-self.k_grf * np.sum(e_grf_f))

        acts = self.model.muscle_activation_array()
        excs = self.model.muscle_excitation_array()
        self.rwd_dict = {
            "pos": self.w_pos * np.exp(-self.k_pos * e_pos),
            "vel": self.w_vel * np.exp(-self.k_vel * e_vel),
            "root": self.w_root * np.exp(-self.k_root * e_root),
            "ee": self.w_ee * np.exp(-self.k_ee * e_ee),
            "torque": self.w_torque * s_torque,
            "grf": self.w_grf * s_grf,
            "effort": -self.w_effort * np.mean(acts ** 2),
            "smooth": -self.w_smooth * np.mean((excs - self.prev_excs) ** 2),
        }
        if self.w_cop > 0:
            self.rwd_dict["cop"] = self.w_cop * self._cop_score(i, grf)
        self._root_dev = float(np.sqrt(e_root))
        return self.rwd_dict

    def cop_errors(self, i, grf):
        """Per foot: (ref loaded, sim loaded, horizontal COP-minus-foot error vector in m or None)."""
        R, thr = self.ref, self.cop_min_force * self.weight
        out = []
        for f, lg in enumerate(self.feet_legs):
            ref_on = R.grf[i, f, 1] > thr
            sim_on = grf[f, 1] > thr
            d = None
            if ref_on and sim_on:
                c = np.array(lg.contact_pos().array()) - np.array(self.ee_bodies[f].pos().array())
                c_ref = R.cop[i, f] - R.ee[i, f]
                d = (c - c_ref)[[0, 2]]
            out.append((ref_on, sim_on, d))
        return out

    def _cop_score(self, i, grf):
        scores = [np.exp(-self.k_cop * np.sum(d ** 2)) if d is not None else 0.0
                  for ref_on, sim_on, d in self.cop_errors(i, grf) if ref_on]
        return float(np.mean(scores)) if scores else 1.0

    def max_step_reward(self):
        return sum(getattr(self, k) for k in ("w_pos", "w_vel", "w_root", "w_ee", "w_torque", "w_grf", "w_cop"))

    def custom_reward(self):
        self._update_rwd_dict()
        return max(float(np.sum(list(self.rwd_dict.values()))), self.min_reward)

    # ------------------------------------------------------------------ termination
    def _get_done(self):
        if super()._get_done():                       # fall (COM / head height)
            return True
        if getattr(self, "_root_dev", 0.0) > self.max_root_dev:
            return True
        if self.steps >= self.max_episode_steps:
            return True
        return self._ref_index() >= self.ref_end - 1

    def tracking_errors(self):
        """Per-step diagnostics for evaluation scripts (degrees, N.m/kg, body weights)."""
        i = self._ref_index()
        q = self.model.dof_position_array()
        tau = np.array([self.dofs[j].muscle_moment() for j in self.i_torque])
        grf = np.array([lg.contact_force().array() for lg in self.feet_legs])
        cop = np.full((2, 2), np.nan)          # horizontal COP error per foot (m), NaN = not both loaded
        for f, (_, _, d) in enumerate(self.cop_errors(i, grf)):
            if d is not None:
                cop[f] = d
        return dict(i=i, q=q.copy(), q_ref=self.ref.q[i].copy(), cop_err=cop,
                    tau=tau / self.mass, tau_ref=self.ref.tau[i, self.i_torque] / self.mass,
                    grf=grf / self.weight, grf_ref=self.ref.grf[i] / self.weight)
