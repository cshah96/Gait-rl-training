"""Plot training curves from a depRL run's log.csv.

    python scripts/deprl_plot.py --experiment sconewalk_h0918_v1      # newest run
    python scripts/deprl_plot.py --run results/deprl/sconewalk_h0918_v1/<run>
    python scripts/deprl_plot.py --experiment sconewalk_h0918_v1 --show

Writes <run>/training_curves.png (and optionally opens a window).
"""
import argparse
import glob
import os
import sys

import pandas as pd


def newest_run(experiment):
    from _paths import DEPRL_RESULTS

    root = os.path.join(DEPRL_RESULTS, experiment)
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "log.csv")))
    if not runs:
        sys.exit(f"No runs with log.csv in {root}")
    return runs[-1]


PANELS = [
    ("test/episode_score/mean", "Test episode return"),
    ("test/episode_length/mean", "Test episode length (steps, max 1000)"),
    ("train/episode_score/mean", "Train episode return"),
    ("train/steps_per_second", "Env steps / second"),
]


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--experiment")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = os.path.abspath(os.path.expanduser(args.run)) if args.run else newest_run(args.experiment)
    df = pd.read_csv(os.path.join(run, "log.csv"))
    x = df["train/steps"] / 1e6

    panels = [(c, t) for c, t in PANELS if c in df.columns]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else axes
    for ax, (col, title) in zip(axes, panels):
        ax.plot(x, df[col], lw=1.5)
        std = col.replace("/mean", "/std")
        if std in df.columns:
            ax.fill_between(x, df[col] - df[std], df[col] + df[std], alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Environment steps (millions)")
    fig.suptitle(os.path.relpath(run, os.path.dirname(os.path.dirname(run))), fontsize=10)
    fig.tight_layout()
    out = os.path.join(run, "training_curves.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    last = df.iloc[-1]
    for c, t in panels:
        print(f"  {t:<40s} {last[c]:,.2f}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
