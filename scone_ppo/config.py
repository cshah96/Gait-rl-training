"""
All training settings in one place.

Every field can be overridden from the command line, e.g.
    python scripts/ppo_train.py --total_steps 5000000 --effort_coeff 5 --num_workers 12

The defaults reproduce the run that learned to walk (gait-training run
"20261105_101425_const_1p2_full", now in results/ppo/): constant-speed walking at
1.2 m/s, volume-weighted squared-activation effort cost, 20M steps.
"""
from dataclasses import dataclass, asdict, fields
import argparse
import json


@dataclass
class Config:
    # ---------------- experiment ----------------
    run_name: str = "const_1p2"        # appended to the run folder name
    seed: int = 0
    total_steps: int = 20_000_000        # total environment steps (1 step = 10 ms of simulation)
    device: str = "cpu"                  # policy is tiny; CPU is fine
    torch_threads: int = 4               # CPU threads for the neural network

    # ---------------- environment ----------------
    env_id: str = "sconewalk_h0918-v1"   # 2D, 9 dof, 18 muscles, Hyfydy
    num_workers: int = 16                # parallel processes (leave a few CPU cores free)
    envs_per_worker: int = 4             # simulations per process -> num_envs = 16 x 4 = 64
    max_episode_steps: int = 1000        # 1000 x 10 ms = 10 s of walking per episode
    use_delayed_sensors: bool = False    # True = observe neurally-delayed sensors only

    # ---------------- speed objective ----------------
    # speed_mode:
    #   "min"      : sconegym default. Velocity reward saturates once speed >= target_speed
    #                ("walk at least this fast"); going faster is not penalized.
    #   "constant" : velocity reward is a bell curve that peaks AT target_speed and drops
    #                when walking slower OR faster -> trains constant-speed walking.
    speed_mode: str = "constant"         # default = setting of the proven walking run
    target_speed: float = 1.2            # [m/s]
    speed_sigma: float = 0.25            # constant mode: bell-curve width [m/s] (smaller = stricter)
    speed_avg_time: float = 0.5          # constant mode: compare the speed averaged over ~0.5 s

    # ---------------- energy-efficiency reward ----------------
    # Extra penalty added on top of sconegym's built-in reward:
    #   r_effort = -effort_coeff * effort
    # effort_type:
    #   "volume_squared" : sum(volume_i * a_i^2) / sum(volume_i)   (big muscles cost more)
    #   "squared"        : mean(a_i^2)
    #   "cubed"          : mean(a_i^3)  (punishes high activations harder, fatigue-like)
    effort_type: str = "volume_squared"
    effort_coeff: float = 10.0

    # ---------------- PPO ----------------
    n_steps: int = 128                   # steps collected per env per update -> batch = 128 x 64 = 8192
    n_epochs: int = 10                   # passes over each batch
    minibatch_size: int = 2048
    learning_rate: float = 3e-4
    anneal_lr: bool = True               # linearly decay LR to 0 over training
    gamma: float = 0.99                  # discount factor (~1 s horizon at 10 ms steps)
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03              # stop epoch loop early if policy changes too much

    # ---------------- network ----------------
    hidden_size: int = 256
    init_log_std: float = -1.5           # initial exploration noise std = exp(-1.5) ~ 0.22
    init_action_mean: float = 0.1        # initial mean muscle excitation

    # ---------------- logging / checkpoints ----------------
    log_interval: int = 1                # print every N updates
    save_interval: int = 25              # checkpoint every N updates (25 x 8192 = 204,800 steps)
    runs_dir: str = ""                   # "" = <repo>/results/ppo

    @property
    def num_envs(self):
        return self.num_workers * self.envs_per_worker

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path):
        with open(path) as f:
            return Config(**json.load(f))


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_config(argv=None) -> Config:
    """Build a Config from defaults + command-line overrides."""
    parser = argparse.ArgumentParser(description="PPO training for sconegym walking")
    for f in fields(Config):
        arg_type = _str2bool if f.type in (bool, "bool") else type(f.default)
        parser.add_argument(f"--{f.name}", type=arg_type, default=f.default)
    args = parser.parse_args(argv)
    return Config(**vars(args))
