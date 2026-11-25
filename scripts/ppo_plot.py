"""
Plot PPO training curves from a run's progress.csv.

    python scripts/ppo_plot.py --latest                    # newest run in results/ppo/
    python scripts/ppo_plot.py results/ppo/<run>           # saves <run>/training.png + reward_terms.png
    python scripts/ppo_plot.py results/ppo/<run> --show    # also opens a window
"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--latest", action="store_true", help="plot the newest run in results/ppo/")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if args.latest or not args.run_dir:
        import glob
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ppo")
        runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "progress.csv")))
        if not runs:
            p.error(f"no PPO runs in {root}")
        args.run_dir = runs[-1]
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.run_dir, "progress.csv")) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    steps = col("global_step") / 1e6
    cfg_path = os.path.join(args.run_dir, "config.json")
    target = json.load(open(cfg_path)).get("target_speed", 1.2) if os.path.exists(cfg_path) else 1.2

    panels = [
        ("ep_return", "Episode return", None),
        ("ep_length", "Episode length (steps, 1000 = 10 s)", None),
        ("ep_speed_ms", f"Walking speed [m/s] (target {target})", target),
        ("ep_speed_error", "Speed error |v_avg - target| [m/s]", None),
        ("ep_fell", "Fraction of episodes ending in a fall", None),
        ("ep_mean_effort", "Mean muscle effort per step", None),
        ("ep_effort_per_m", "Effort per meter (lower = more economical)", None),
        ("explained_var", "Critic explained variance", None),
        ("action_std", "Policy exploration std", None),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    for ax, (key, title, ref) in zip(axes.flat, panels):
        if key not in rows[0]:          # older runs may not have every column
            ax.set_visible(False)
            continue
        ax.plot(steps, col(key), lw=1.5)
        if ref is not None:
            ax.axhline(ref, ls="--", c="gray", lw=1)
        ax.set_title(title)
        ax.set_xlabel("environment steps [millions]")
        ax.grid(alpha=0.3)
    fig.suptitle(os.path.basename(os.path.normpath(args.run_dir)))
    fig.tight_layout()
    out = os.path.join(args.run_dir, "training.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")

    # reward-term breakdown
    terms = [k for k in rows[0] if k.startswith("ep_r_")]
    fig2, ax = plt.subplots(figsize=(10, 5))
    for k in terms:
        ax.plot(steps, col(k), label=k[5:])
    ax.set_title("Reward terms (summed per episode)")
    ax.set_xlabel("environment steps [millions]")
    ax.legend()
    ax.grid(alpha=0.3)
    out2 = os.path.join(args.run_dir, "reward_terms.png")
    fig2.savefig(out2, dpi=110)
    print(f"Saved {out2}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
