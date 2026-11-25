"""Point SCONE Studio at this repo's Python so double-clicking a .pt works.

SCONE Studio evaluates depRL checkpoints by running
    <python> -m deprl.play --checkpoint_file <step_N.pt> --num_episodes <n>
where <python> comes from Tools -> Preferences -> Sconegym -> "Name of the
Python executable used for evaluation" (default: "python", which is the system
Python 3.12 and has neither deprl nor a sconepy build for it).

This script sets that preference (stored in ~/.config/SCONE/studio-settings.zml)
to this repo's bin/studio-python wrapper, which runs the .venv Python and sends
PPO checkpoints to scripts/ppo_evaluate.py and depRL checkpoints to deprl.play. A backup of the original file is kept as
studio-settings.zml.bak. Close SCONE Studio before running it.

    python scripts/configure_studio.py              # use bin/studio-python
    python scripts/configure_studio.py --episodes 3
"""
import argparse
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.expanduser("~/.config/SCONE/studio-settings.zml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=os.path.join(ROOT, "bin", "studio-python"))
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    text = open(SETTINGS).read() if os.path.exists(SETTINGS) else ""
    if text and not os.path.exists(SETTINGS + ".bak"):
        shutil.copy(SETTINGS, SETTINGS + ".bak")

    block = f'sconegym {{\n\tpython = "{args.python}"\n\tnum_episodes = {args.episodes}\n}}\n'
    pattern = re.compile(r"^sconegym\s*\{[^{}]*\}\s*\n?", re.M)
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        # keep 'version = ...' as the last line, like SCONE does
        m = re.search(r"^version\s*=.*$", text, re.M)
        text = text[: m.start()] + block + text[m.start():] if m else text + block

    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    with open(SETTINGS, "w") as f:
        f.write(text)
    print(f"Updated {SETTINGS}:")
    print(block)


if __name__ == "__main__":
    main()
