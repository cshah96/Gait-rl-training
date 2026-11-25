"""
Environment creation + the energy-efficiency reward wrapper.

Layering:
    Hyfydy (physics)  ->  SCONE / sconepy (model API)  ->  sconegym GaitGym (gym env)
                      ->  EnergyWalkEnv (scone_ppo/envs.py: effort penalty, time limit, episode stats)
"""
import gym
import numpy as np

import sconegym  # noqa: F401  (registers the "scone*" environments with gym)


class EnergyWalkEnv:
    """
    Thin wrapper around a sconegym GaitGym environment.

    Adds four things:
      1. An effort (energy) penalty to the reward, so the policy learns to walk
         with as little muscle activation as possible.
      2. Optionally, a *constant-speed* velocity reward (speed_mode="constant") that
         replaces sconegym's "at least target speed" reward.
      3. A time limit (sconegym envs never end on their own unless the model falls).
         Hitting the limit is a *truncation*, not a failure: info["truncated"] = True.
      4. Per-episode statistics in info["episode"] when an episode ends.

    It uses the old gym 0.13 API that sconegym is built on:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    """

    def __init__(self, env_id, max_episode_steps=1000, effort_type="volume_squared",
                 effort_coeff=10.0, use_delayed_sensors=False, speed_mode="min",
                 target_speed=1.2, speed_sigma=0.25, speed_avg_time=0.5):
        # target_vel overrides the value registered in sconegym/__init__.py
        self.env = gym.make(env_id, use_delayed_sensors=use_delayed_sensors, target_vel=target_speed)
        self.model = self.env.model                     # the sconepy Model
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = max_episode_steps
        self.effort_type = effort_type
        self.effort_coeff = effort_coeff
        self.dt = self.env.step_size                     # 0.01 s per env step
        if speed_mode not in ("min", "constant"):
            raise ValueError(f"unknown speed_mode {speed_mode}")
        self.speed_mode = speed_mode
        self.target_speed = target_speed
        self.speed_sigma = speed_sigma
        self.speed_avg_time = speed_avg_time

        # Muscle volumes [m^3] (from max force, fiber length, specific tension).
        # Used to weight effort: activating the big vasti costs more than tib_ant.
        vol = np.array([m.volume() for m in self.model.muscles()])
        self.volume_weights = vol / vol.sum()
        self.muscle_names = [m.name() for m in self.model.muscles()]

    # ------------------------------------------------------------------
    def effort(self):
        """Instantaneous effort from current muscle activations (0 = relaxed)."""
        a = self.model.muscle_activation_array()
        if self.effort_type == "volume_squared":
            return float(np.sum(self.volume_weights * a ** 2))
        if self.effort_type == "squared":
            return float(np.mean(a ** 2))
        if self.effort_type == "cubed":
            return float(np.mean(a ** 3))
        raise ValueError(f"unknown effort_type {self.effort_type}")

    def velocity_reward(self):
        """
        Constant-speed reward: a bell curve centred on target_speed.
            r = vel_coeff * exp(-((v_avg - target) / sigma)^2)
        v_avg is the forward COM velocity averaged over ~speed_avg_time seconds
        (exponential moving average). Within each stride the COM speeds up and slows
        down by roughly +-0.1-0.2 m/s; averaging means natural gait oscillation is
        not punished, only drifting away from the target speed.
        """
        err = (self._v_avg - self.target_speed) / self.speed_sigma
        return self.env.vel_coeff * float(np.exp(-err ** 2))

    def reset(self):
        obs = self.env.reset()
        self._steps = 0
        self._ep_return = 0.0
        self._ep_effort = 0.0
        self._ep_terms = {}
        self._start_x = self.model.com_pos().x
        self._v_avg = self.model.com_vel().x      # start the average at the initial speed
        self._ep_speed_err = 0.0
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)
        self._steps += 1
        terms = dict(self.env.get_rwd_dict())

        # running average of forward speed (time constant speed_avg_time)
        alpha = min(1.0, self.dt / self.speed_avg_time)
        self._v_avg += alpha * (self.model.com_vel().x - self._v_avg)
        self._ep_speed_err += abs(self._v_avg - self.target_speed)

        if self.speed_mode == "constant":
            # swap sconegym's plateau velocity term for the constant-speed one
            vel = self.velocity_reward()
            base_reward = base_reward - terms["gaussian_vel"] + vel
            terms["gaussian_vel"] = vel

        effort = self.effort()
        effort_reward = -self.effort_coeff * effort
        reward = base_reward + effort_reward

        # bookkeeping for logging
        self._ep_return += reward
        self._ep_effort += effort
        terms["effort"] = effort_reward
        for k, v in terms.items():
            self._ep_terms[k] = self._ep_terms.get(k, 0.0) + float(v)

        truncated = False
        if not done and self._steps >= self.max_episode_steps:
            done, truncated = True, True

        info = dict(info)
        info["truncated"] = truncated
        if done:
            duration = self._steps * self.dt
            distance = self.model.com_pos().x - self._start_x
            info["episode"] = {
                "return": self._ep_return,
                "length": self._steps,
                "duration_s": duration,
                "distance_m": distance,
                "speed_ms": distance / duration,
                # mean |averaged speed - target| [m/s]: lower = better speed tracking
                "speed_error": self._ep_speed_err / self._steps,
                "mean_effort": self._ep_effort / self._steps,
                # effort per meter travelled: lower = more economical gait
                # (undefined if the model barely moved, e.g. fell on the spot)
                "effort_per_m": self._ep_effort * self.dt / distance if distance > 0.1 else float("nan"),
                "fell": not truncated,
                **{f"r_{k}": v for k, v in self._ep_terms.items()},
            }
        return obs, reward, done, info

    # pass-throughs used for evaluation / storing results for SCONE Studio
    def store_next_episode(self):
        self.env.store_next_episode()

    def write_now(self):
        self.env.write_now()

    def set_output_dir(self, name):
        self.env.set_output_dir(name)

    def close(self):
        self.env.close()


def make_env(cfg):
    """Create one EnergyWalkEnv from a Config."""
    return EnergyWalkEnv(
        cfg.env_id,
        max_episode_steps=cfg.max_episode_steps,
        effort_type=cfg.effort_type,
        effort_coeff=cfg.effort_coeff,
        use_delayed_sensors=cfg.use_delayed_sensors,
        speed_mode=cfg.speed_mode,
        target_speed=cfg.target_speed,
        speed_sigma=cfg.speed_sigma,
        speed_avg_time=cfg.speed_avg_time,
    )
