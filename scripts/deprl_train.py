"""Launch (or resume) DEP-RL (MPO + DEP) training on a sconegym environment.

Thin wrapper around ``python -m deprl.main <config.yaml>`` that lets you
override common settings from the command line without editing the YAML.
The fully-resolved config is written next to the source config as
``configs/.resolved/<name>.yaml`` and depRL also copies it into the run folder.

Examples
--------
    # full training run with the reference hyper-parameters
    python scripts/deprl_train.py configs/deprl/sconewalk_h0918_v1.yaml

    # quick pipeline test (~5-10 min)
    python scripts/deprl_train.py configs/deprl/sconewalk_h0918_v1_smoke.yaml

    # 10-minute demo run on the 90-muscle model (stops cleanly + saves a checkpoint)
    python scripts/deprl_train.py configs/deprl/sconewalk_h2190_v1.yaml --minutes 10 \
        --save-steps 2e5 --name sconewalk_h2190_v1_10min --fresh

    # a second seed, saved under its own name, starting fresh
    python scripts/deprl_train.py configs/deprl/sconewalk_h0918_v1.yaml --seed 1 \
        --name sconewalk_h0918_v1_seed1 --fresh

Where results go
----------------
    scone-rl/results/deprl/<name>/<YYMMDD.HHMMSS>.<model>/
    (depRL writes to ~/SCONE/results/scone-rl/deprl/<name>; ~/SCONE/results/scone-rl
     is a symlink to scone-rl/results, so the data lives in this repo)
        config.yaml          the exact config used
        log.csv              training curves (one row per epoch)
        checkpoints/step_<N>.pt   policy checkpoints (open these in SCONE Studio)
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a depRL yaml config")
    p.add_argument("--name", help="experiment name (results sub-folder)")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=float, help="total env steps, e.g. 1e8")
    p.add_argument("--save-steps", type=float, help="checkpoint interval in env steps")
    p.add_argument("--parallel", type=int, help="number of worker processes")
    p.add_argument("--sequential", type=int, help="envs per worker process")
    p.add_argument("--minutes", type=float,
                   help="wall-clock limit: stop after this many minutes and save a final checkpoint")
    p.add_argument("--no-buffer", action="store_true",
                   help="do not save/load the replay buffer (~6 GB for H2190); optimiser + normalisers are still saved")
    p.add_argument("--fresh", action="store_true", help="do not resume an existing run with the same name")
    p.add_argument("--cpu", action="store_true", help="force PyTorch to run on CPU")
    p.add_argument("--dry-run", action="store_true", help="only print the resolved config")
    return p.parse_args()


def replace_trainer_arg(trainer: str, key: str, value: float) -> str:
    """Replace e.g. steps=int(5e8) inside the trainer expression string."""
    import re

    # (?<![\w]) so that 'steps' does not also match 'epoch_steps' / 'save_steps'
    new, n = re.subn(rf"(?<![\w]){key}=int\([^)]*\)", f"{key}=int({value:g})", trainer)
    if n == 0:
        raise ValueError(f"could not find '{key}=int(...)' in trainer string: {trainer}")
    return new


def install_time_limit(minutes):
    """Make Trainer.max_steps read as 0 once the deadline passes, so the trainer's own
    end-of-training branch runs: it saves a final checkpoint and closes the workers."""
    import time
    from deprl import custom_trainer

    deadline = time.time() + minutes * 60

    def get(self):
        return 0 if time.time() > deadline else self._max_steps

    def set_(self, v):
        self._max_steps = v

    custom_trainer.Trainer.max_steps = property(get, set_)
    print(f" time limit : {minutes:g} min (final checkpoint saved when it is reached)", flush=True)


def disable_buffer_saving():
    """Skip only the replay buffer in checkpoints. full_save: 0 would also drop the
    optimiser state and observation normaliser, which are small and needed to resume well.
    On resume the buffer starts empty and refills during the warm-up steps."""
    from deprl.vendor.tonic.torch.agents.agent import Agent

    Agent.save_buffer = lambda self, path: None
    Agent.load_buffer = lambda self, path, *a, **k: None
    print(" replay buffer: NOT saved (policy, optimiser, normalisers are)", flush=True)


def trust_own_checkpoints():
    """PyTorch >= 2.6 loads with weights_only=True by default, which rejects the numpy arrays in
    depRL's obs_norm.pt: resuming then silently printed "Not loading observation normalizer" and
    restarted the input normalisation. These files are written by this repo's own runs, so load
    them fully."""
    import torch

    if getattr(torch.load, "_scone_rl_trusted", False):
        return
    orig = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    load._scone_rl_trusted = True
    torch.load = load


def tolerate_missing_buffer():
    """Resuming from a checkpoint saved with --no-buffer: there are no buffer files, and
    tonic would then also skip loading the normalisers (same try block). Skip the buffer
    instead, so the rest of the training state still loads; the buffer starts empty."""
    from deprl.vendor.tonic.torch.agents.agent import Agent

    original = Agent.load_buffer

    def load_buffer(self, path, *a, **k):
        if not os.path.exists(self.get_path(path, "buffer_0")):
            print(" no saved replay buffer in checkpoint - starting with an empty buffer", flush=True)
            # still restore the small adaptive-effort state (action_cost, score_avg, ...)
            import torch

            skip = {"buffers", "index", "size", "num_workers", "max_size"}
            for field in getattr(self.replay, "checkpoint_fields", []):
                f = self.get_path(path, field)
                if field not in skip and os.path.exists(f):
                    setattr(self.replay, field, torch.load(f))
            return
        return original(self, path, *a, **k)

    Agent.load_buffer = load_buffer


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    t = config["tonic"]

    if args.name:
        t["name"] = args.name
    if args.seed is not None:
        t["seed"] = args.seed
    if args.parallel:
        t["parallel"] = args.parallel
    if args.sequential:
        t["sequential"] = args.sequential
    if args.fresh:
        t["resume"] = False
    if args.cpu:
        t["cpu_override"] = True
    if args.steps:
        t["trainer"] = replace_trainer_arg(t["trainer"], "steps", args.steps)
    if args.save_steps:
        t["trainer"] = replace_trainer_arg(t["trainer"], "save_steps", args.save_steps)

    from _paths import DEPRL_RESULTS, deprl_tonic_name, ensure_scone_link

    short_name = t["name"].split("/")[-1]
    t["name"] = deprl_tonic_name(short_name)   # "scone-rl/deprl/<name>" relative to ~/SCONE/results
    results = os.path.join(DEPRL_RESULTS, short_name)
    out_dir = os.path.join(ROOT, "configs", ".resolved")
    os.makedirs(out_dir, exist_ok=True)
    resolved = os.path.join(out_dir, f"{short_name}.yaml")
    with open(resolved, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=1000)

    print("=" * 72)
    print(f" experiment : {short_name}")
    print(f" env        : {t['environment']}")
    print(f" workers    : {t['parallel']} x {t['sequential']} = {t['parallel'] * t['sequential']} envs")
    print(f" trainer    : {t['trainer']}")
    print(f" seed       : {t['seed']}   resume: {t['resume']}")
    print(f" config     : {resolved}")
    print(f" results    : {results}/<date>.<model>/checkpoints/step_*.pt")
    print(f"              (SCONE/Studio: ~/SCONE/results/{t['name']})")
    print("=" * 72, flush=True)
    if args.dry_run:
        return
    ensure_scone_link()

    if args.minutes:
        install_time_limit(args.minutes)
    trust_own_checkpoints()
    tolerate_missing_buffer()
    if args.no_buffer:
        disable_buffer_saving()
    from deprl import main as deprl_main

    sys.argv = [sys.argv[0], resolved]
    deprl_main.main()


if __name__ == "__main__":
    main()
