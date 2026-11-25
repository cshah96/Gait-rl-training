"""List all training runs (PPO and DEP-RL) in scone-rl/results/, with their checkpoints.

    python scripts/list_runs.py            # everything
    python scripts/list_runs.py const      # only runs whose path contains 'const'
"""
import glob
import os
import re
import sys

from _paths import DEPRL_RESULTS, PPO_RESULTS


def step_of(path):
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def show(run, root, extra=None):
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "step_*.pt")), key=step_of)
    evals = sorted(os.path.basename(e) for e in glob.glob(os.path.join(run, "run_*")) if os.path.isdir(e))
    print(f"  {os.path.relpath(run, root)}")
    print(f"      path        : {run}")
    if ckpts:
        print(f"      checkpoints : {len(ckpts)} (latest: {os.path.basename(ckpts[-1])})")
    for name in extra or []:
        if os.path.exists(os.path.join(run, name)):
            print(f"      {name:<12}: {os.path.join(run, name)}")
    if evals:
        print(f"      rollouts    : {', '.join(evals)}  (open the .sto files in SCONE Studio)")


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    ppo = [os.path.dirname(c) for c in sorted(glob.glob(os.path.join(PPO_RESULTS, "*", "config.json"))) if filt in c]
    dep = [os.path.dirname(c) for c in sorted(glob.glob(os.path.join(DEPRL_RESULTS, "*", "*", "config.yaml"))) if filt in c]

    print(f"PPO runs ({PPO_RESULTS}):")
    for r in ppo:
        show(r, PPO_RESULTS, extra=["best.pt", "latest.pt"])
    if not ppo:
        print("  none")
    print(f"\nDEP-RL runs ({DEPRL_RESULTS}):")
    for r in dep:
        show(r, DEPRL_RESULTS)
    if not dep:
        print("  none")


if __name__ == "__main__":
    main()
