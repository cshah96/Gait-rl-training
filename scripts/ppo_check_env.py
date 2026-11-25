"""
PPO environment inspector: what does the EnergyWalkEnv look like (obs layout, reward terms, sim speed)?

    python scripts/ppo_check_env.py

Prints the SCONE/Hyfydy versions, the muscles, the degrees of freedom, the layout
of the 121-number observation vector, a breakdown of the reward, and the
simulation speed. Run this first whenever something seems broken.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from scone_ppo.config import Config  # noqa: E402
from scone_ppo.envs import make_env  # noqa: E402


def main():
    from sconetools import sconepy
    print(f"SCONE {sconepy.version()} | Hyfydy available: {sconepy.is_supported('ModelHyfydy')}")

    cfg = Config()
    env = make_env(cfg)
    m = env.model
    n_mus, n_dof = len(m.muscles()), len(m.dofs())

    print(f"\nModel: {m.name()}  mass={m.mass():.1f} kg  bodies={[b.name() for b in m.bodies()]}")
    print(f"\nACTION: {env.action_space.shape[0]} muscle excitations (0..1, clipped to 0..0.5 by sconegym)")
    for i, (mus, w) in enumerate(zip(m.muscles(), env.volume_weights)):
        print(f"  [{i:2d}] {mus.name():14s} Fmax={mus.max_isometric_force():6.0f} N  "
              f"volume weight={w:.3f}")

    print(f"\nDOFs ({n_dof}): {[d.name() for d in m.dofs()]}")

    layout = [("muscle fiber length", n_mus), ("muscle fiber velocity", n_mus), ("muscle force", n_mus),
              ("muscle excitation", n_mus), ("head orientation (quaternion)", 4),
              ("head angular velocity", 3), ("feet position rel. to pelvis (l, r)", 6),
              ("dof positions (pelvis_tx zeroed)", n_dof), ("dof velocities", n_dof),
              ("muscle activation", n_mus)]
    obs = env.reset()
    print(f"\nOBSERVATION: {obs.shape[0]} values")
    i = 0
    for name, n in layout:
        print(f"  [{i:3d}:{i + n:3d}] {name:38s} e.g. {np.round(obs[i:i + min(n, 3)], 3)}")
        i += n
    assert i == obs.shape[0] or cfg.use_delayed_sensors, "observation layout mismatch"

    print("\nREWARD for one step with all muscles at 0.1 excitation:")
    obs, r, done, info = env.step(np.full(env.action_space.shape, 0.1))
    for k, v in env.env.get_rwd_dict().items():
        print(f"  {k:16s} {v:8.3f}")
    print(f"  {'effort':16s} {-cfg.effort_coeff * env.effort():8.3f}   (added by EnergyWalkEnv)")
    print(f"  {'TOTAL':16s} {r:8.3f}")

    print("\nRandom-action episodes (expect falls within ~1 s):")
    t, n = time.time(), 0
    for ep in range(3):
        env.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(env.action_space.sample())
            n += 1
        e = info["episode"]
        print(f"  episode {ep}: {e['duration_s']:.2f} s, return {e['return']:.1f}, fell={e['fell']}")
    print(f"\nSpeed: {n / (time.time() - t):.0f} env steps/s per process "
          f"(= {n / (time.time() - t) * 0.01:.0f}x real time)")
    env.close()


if __name__ == "__main__":
    main()
