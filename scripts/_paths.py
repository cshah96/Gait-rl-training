"""Path helpers: keep ALL output inside scone-rl/results/.

sconepy always writes .sto files into the SCONE results directory
(``sconepy.scone_results_dir()``, here ~/SCONE/results). It has no API to change
that and treats absolute paths as relative to it. depRL does the same for its
training runs. To keep everything inside this repo without touching SCONE's
global settings, there is ONE symlink:

    ~/SCONE/results/scone-rl  ->  <repo>/results

Code then writes to "scone-rl/ppo/..." or "scone-rl/deprl/..." relative to the
SCONE results dir, and the bytes land in <repo>/results/ppo/... or
<repo>/results/deprl/.... SCONE Studio shows everything under a single
"scone-rl" folder in its Optimization Results pane.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_RESULTS = os.path.join(ROOT, "results")
PPO_RESULTS = os.path.join(REPO_RESULTS, "ppo")
DEPRL_RESULTS = os.path.join(REPO_RESULTS, "deprl")
LINK_NAME = "scone-rl"


def scone_results_dir():
    from sconetools import sconepy

    return sconepy.scone_results_dir()


def ensure_scone_link():
    """Create ~/SCONE/results/scone-rl -> <repo>/results (idempotent). Returns the link path."""
    for d in (PPO_RESULTS, DEPRL_RESULTS):
        os.makedirs(d, exist_ok=True)
    link = os.path.join(scone_results_dir(), LINK_NAME)
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(REPO_RESULTS):
            raise RuntimeError(f"{link} points to {os.path.realpath(link)}, expected {REPO_RESULTS}")
        return link
    if os.path.exists(link):
        raise RuntimeError(f"{link} exists and is not a symlink; move it away and re-run")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(REPO_RESULTS, link)
    print(f"Linked {link} -> {REPO_RESULTS}")
    return link


def scone_relative(repo_path):
    """<repo>/results/x/y -> 'scone-rl/x/y' (for sconegym's set_output_dir)."""
    rel = os.path.relpath(os.path.realpath(repo_path), os.path.realpath(REPO_RESULTS))
    if rel.startswith(".."):
        raise ValueError(f"{repo_path} is not inside {REPO_RESULTS}")
    return os.path.join(LINK_NAME, rel)


def to_scone_path(path):
    """Map <repo>/results/... to ~/SCONE/results/scone-rl/... (same file, via the link).

    deprl.play decides where to write .sto files by checking whether the
    checkpoint path starts with the SCONE results dir, so it must be given
    the ~/SCONE/results/... form of the path.
    """
    path = os.path.abspath(os.path.expanduser(path))
    real = os.path.realpath(path)
    repo = os.path.realpath(REPO_RESULTS)
    if real.startswith(repo + os.sep):
        return os.path.join(scone_results_dir(), LINK_NAME, os.path.relpath(real, repo))
    return path


def deprl_tonic_name(name):
    """depRL experiment name as seen from the SCONE results dir."""
    return f"{LINK_NAME}/deprl/{name}"
