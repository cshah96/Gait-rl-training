"""Replay a trained PPO policy, print gait statistics and write .sto files for SCONE Studio.

    python scripts/ppo_evaluate.py --latest                                   # newest run, latest checkpoint
    python scripts/ppo_evaluate.py results/ppo/<run>/best.pt
    python scripts/ppo_evaluate.py results/ppo/<run>/checkpoints/step_19996672.pt --episodes 5
    python scripts/ppo_evaluate.py results/ppo/<run>/best.pt --seconds 60 --no_store   # long stability test

.sto files go next to the checkpoint's run folder, like depRL's layout:
    results/ppo/<run>/run_checkpoint_<N>/<episode>_<return>.sto   (for checkpoints/step_<N>.pt)
    results/ppo/<run>/run_best/...                                (for best.pt / latest.pt)

SCONE Studio also runs this script when you double-click a PPO step_<N>.pt
(through bin/studio-python).
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scone_ppo.config import Config  # noqa: E402
from scone_ppo.envs import make_env  # noqa: E402
from scone_ppo.networks import ActorCritic, to_env_action  # noqa: E402
from scone_ppo.normalization import ObsNormalizer  # noqa: E402


def newest_run():
    from _paths import PPO_RESULTS

    runs = sorted(d for d in glob.glob(os.path.join(PPO_RESULTS, "*")) if os.path.isfile(os.path.join(d, "config.json")))
    if not runs:
        sys.exit(f"No PPO runs in {PPO_RESULTS}")
    return runs[-1]


def output_folder(ckpt_path):
    """Run folder + sub-folder name for the .sto files of this checkpoint."""
    real = os.path.realpath(ckpt_path)
    m = re.search(r"step_(\d+)\.pt$", real)
    if m and os.path.basename(os.path.dirname(real)) == "checkpoints":
        return os.path.dirname(os.path.dirname(real)), f"run_checkpoint_{m.group(1)}"
    return os.path.dirname(real), "run_" + os.path.splitext(os.path.basename(real))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="a PPO .pt file (best.pt, latest.pt or checkpoints/step_N.pt)")
    p.add_argument("--latest", action="store_true", help="use latest.pt of the newest run")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=None,
                   help="episode time limit in seconds (default: the training limit, 10 s)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--no_store", action="store_true", help="don't write .sto files")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if args.latest:
        args.checkpoint = os.path.join(newest_run(), "latest.pt")
    if not args.checkpoint:
        p.error("give a checkpoint path or --latest")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # older checkpoints lack newer fields -> fall back to Config defaults for those
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in Config.__dataclass_fields__})
    if args.seconds:
        cfg.max_episode_steps = int(round(args.seconds / 0.01))
    np.random.seed(args.seed)

    env = make_env(cfg)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    obs_norm = ObsNormalizer(obs_dim)
    obs_norm.rms.load_state_dict(ckpt["obs_norm"])

    run_dir, sub = output_folder(args.checkpoint)
    out_repo = os.path.join(run_dir, sub)
    if not args.no_store:
        from _paths import ensure_scone_link, scone_relative

        ensure_scone_link()
        env.set_output_dir(scone_relative(out_repo))   # "scone-rl/ppo/<run>/<sub>" inside ~/SCONE/results
    print(f"Checkpoint: {args.checkpoint}  (update {ckpt['update']}, {ckpt['global_step']:,} steps)", flush=True)

    n_ok = 0
    for ep in range(args.episodes):
        if not args.no_store:
            # sconegym names files <episode>_<return>.sto but only counts episodes that end in
            # a fall, so set the counter explicitly to get 00000, 00001, ... for every episode
            env.env.unwrapped.episode = ep
            env.store_next_episode()   # record every frame of the next episode (slower)
        obs = env.reset()
        done, info = False, {}
        while not done:
            x = torch.as_tensor(obs_norm(obs[None], update=False))
            with torch.no_grad():
                action, _, _ = agent.act(x, deterministic=not args.stochastic)
            obs, reward, done, info = env.step(to_env_action(action[0].numpy()))
        if not args.no_store and info.get("truncated"):
            env.write_now()            # sconegym only auto-writes on falls; write timeouts manually
        e = info["episode"]
        n_ok += not e["fell"]
        print(f"episode {ep}: {'FELL' if e['fell'] else 'ok  '} | {e['duration_s']:5.2f} s | "
              f"distance {e['distance_m']:5.2f} m | speed {e['speed_ms']:4.2f} m/s (err {e['speed_error']:.2f}) | "
              f"mean effort {e['mean_effort']:.4f} | effort/m {e['effort_per_m']:.4f} | return {e['return']:.1f}",
              flush=True)
    print(f"\n{n_ok}/{args.episodes} episodes walked the full {cfg.max_episode_steps * 0.01:.0f} s without falling.")

    if not args.no_store:
        stos = sorted(glob.glob(os.path.join(out_repo, "*.sto")))
        print(f".sto files written to {out_repo}:")
        for s in stos:
            print("   ", os.path.basename(s))
        print("Open them in SCONE Studio: Optimization Results -> scone-rl -> ppo -> <run> -> " + sub)
    env.close()


if __name__ == "__main__":
    main()
