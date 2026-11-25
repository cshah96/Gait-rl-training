"""Verify that every piece of the training stack is installed and working.

    python scripts/check_install.py

Checks: Python version, sconepy import, Hyfydy license/engine, the
sconewalk_h0918-v1 gym environment, the PPO package, depRL import, and PyTorch + CUDA.
"""
import os
import sys
import time

OK = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = 0


def check(name, fn):
    global failures
    try:
        msg = fn()
        print(f"[{OK}] {name}" + (f": {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"[{FAIL}] {name}: {type(e).__name__}: {e}")


def py_version():
    v = sys.version_info
    if (v.major, v.minor) != (3, 9):
        raise RuntimeError(
            f"Python {v.major}.{v.minor} found; sconepy only ships a cp39 build"
        )
    return sys.version.split()[0]


def sconepy_check():
    from sconetools import sconepy

    hfd = sconepy.is_supported("ModelHyfydy")
    if not hfd:
        raise RuntimeError(
            "Hyfydy NOT available - check the license in SCONE Studio "
            "(Tools -> Preferences -> Hyfydy)"
        )
    return f"SCONE {sconepy.version()}, Hyfydy supported, results dir = {sconepy.scone_results_dir()}"


def env_check():
    import gym
    import sconegym  # noqa: F401  (registers the envs)

    env = gym.make("sconewalk_h0918-v1")
    env.reset()
    t = time.time()
    n = 500
    for _ in range(n):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
    sps = n / (time.time() - t)
    return (
        f"obs {env.observation_space.shape}, act {env.action_space.shape}, "
        f"{len(env.model.muscles())} muscles, ~{sps:,.0f} steps/s (1 core)"
    )


def ppo_check():
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scone_ppo.config import Config
    from scone_ppo.envs import make_env

    env = make_env(Config())
    env.reset()
    env.step(env.action_space.sample())
    env.close()
    return "scone_ppo imports, EnergyWalkEnv steps OK"


def mimic_check():
    import gym
    import scone_mimic  # noqa: F401  (registers sconemimic_h2190-v1)

    ref = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "mimic", "AB07", "AB07_reference.npz")
    if not os.path.exists(ref):
        return "scone_mimic imports (no reference yet - run scripts/mimic_prepare_subject.sh)"
    env = gym.make("sconemimic_h2190-v1", reference=ref)
    obs = env.reset()
    env.step(env.action_space.sample() * 0.1)
    return f"sconemimic_h2190-v1 OK, obs {obs.shape}, reference {len(env.unwrapped.ref.q)} frames"


def deprl_check():
    import deprl

    return f"deprl {getattr(deprl, '__version__', '?')} from {deprl.__file__}"


def torch_check():
    import torch

    if not torch.cuda.is_available():
        return f"torch {torch.__version__} - CUDA NOT available (training will run on CPU)"
    x = torch.randn(256, 256, device="cuda")
    (x @ x).sum().item()
    return f"torch {torch.__version__}, CUDA OK on {torch.cuda.get_device_name(0)}"


check("Python 3.9", py_version)
check("sconepy + Hyfydy", sconepy_check)
check("gym env sconewalk_h0918-v1", env_check)
check("PPO package (scone_ppo)", ppo_check)
check("imitation env (scone_mimic)", mimic_check)
check("depRL", deprl_check)
check("PyTorch", torch_check)

print()
print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
sys.exit(1 if failures else 0)
