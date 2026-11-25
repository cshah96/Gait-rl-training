"""Roll out a trained policy and write SCONE .sto files for SCONE Studio.

This runs exactly what SCONE Studio runs when you double-click a ``step_*.pt``
checkpoint (``python -m deprl.play --checkpoint_file ... --num_episodes N``),
so you can do it from the terminal / over SSH / for many checkpoints at once.

Examples
--------
    # newest checkpoint of the newest sconewalk_h0918_v1 run
    python scripts/deprl_evaluate.py --experiment sconewalk_h0918_v1

    # a specific checkpoint, 3 episodes
    python scripts/deprl_evaluate.py --checkpoint results/deprl/sconewalk_h0918_v1/<run>/checkpoints/step_50000000.pt -n 3

    # every checkpoint of a run (e.g. to compare learning progress)
    python scripts/deprl_evaluate.py --run results/deprl/sconewalk_h0918_v1/<run> --all

Output
------
    <run>/run_checkpoint_<N>/<episode>_<score>.sto   (+ .par/.zml side files)
Open the .sto in SCONE Studio to view the 3-D animation.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

from _paths import DEPRL_RESULTS, to_scone_path


def step_of(path):
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def newest_run(experiment):
    root = os.path.join(DEPRL_RESULTS, experiment)
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "config.yaml")))
    if not runs:
        sys.exit(f"No runs found in {root}")
    return runs[-1]


def checkpoints_of(run):
    ck = sorted(glob.glob(os.path.join(run, "checkpoints", "step_*.pt")), key=step_of)
    if not ck:
        sys.exit(f"No checkpoints in {run}/checkpoints")
    return ck


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="path to a step_*.pt file")
    g.add_argument("--run", help="path to a run folder (contains config.yaml + checkpoints/)")
    g.add_argument("--experiment", help="experiment name; uses its newest run")
    p.add_argument("--all", action="store_true", help="evaluate every checkpoint of the run")
    p.add_argument("-n", "--num-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noisy", action="store_true", help="sample from the stochastic policy instead of the mean")
    args = p.parse_args()

    if args.checkpoint:
        ckpts = [os.path.abspath(os.path.expanduser(args.checkpoint))]
    else:
        run = os.path.abspath(os.path.expanduser(args.run)) if args.run else newest_run(args.experiment)
        all_ck = checkpoints_of(run)
        ckpts = all_ck if args.all else [all_ck[-1]]

    for ck in ckpts:
        # deprl.play needs the ~/SCONE/results/... form of the path to put
        # run_checkpoint_<N>/ inside the run folder (see _paths.to_scone_path)
        ck = to_scone_path(ck)
        cmd = [
            sys.executable, "-m", "deprl.play",
            "--checkpoint_file", ck,
            "--num_episodes", str(args.num_episodes),
            "--seed", str(args.seed),
        ]
        if args.noisy:
            cmd.append("--noisy")
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        out = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(ck)), f"run_checkpoint_{step_of(ck)}"))
        stos = sorted(glob.glob(os.path.join(out, "*.sto")))
        print(f"\n.sto files written to {out}:")
        for s in stos:
            print("   ", os.path.basename(s))
        print()


if __name__ == "__main__":
    main()
