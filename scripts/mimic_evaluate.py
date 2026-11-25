"""Evaluate a gait-imitation policy against the measured reference: kinematics, kinetics, GRF.

    python scripts/mimic_evaluate.py --experiment sconemimic_h2190_ab07            # newest run, last ckpt
    python scripts/mimic_evaluate.py --checkpoint results/deprl/<exp>/<run>/checkpoints/step_N.pt -n 10

For every episode (deterministic policy, starting from a reference frame without noise) it records
the simulated and reference joint angles, hip/knee/ankle muscle moments vs inverse-dynamics
moments, and per-foot ground reaction forces, then writes into <run>/mimic_eval_step_<N>/:
    report.txt      survival, speed, RMSE + Pearson r per joint (deg, N.m/kg, body weights)
    tracking.png    simulated vs measured curves for the first (longest) episode
    *.sto           the episodes, for SCONE Studio (same folder layout as deprl.play)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PLOT_DOFS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r", "hip_flexion_l", "knee_angle_l", "ankle_angle_l"]


def newest_checkpoint(args):
    if args.checkpoint:
        return os.path.abspath(args.checkpoint)
    run = args.run or sorted(glob.glob(os.path.join(ROOT, "results", "deprl", args.experiment, "*/")))[-1]
    ck = sorted(glob.glob(os.path.join(run, "checkpoints", "step_*.pt")),
                key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not ck:
        sys.exit(f"no checkpoints in {run}")
    return os.path.abspath(ck[-1])


def build(ckpt):
    import deprl  # noqa: F401
    from deprl import env_wrappers
    from deprl.vendor.tonic import environments  # noqa: F401

    run_dir = os.path.dirname(os.path.dirname(ckpt))
    config = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    exec(config["tonic"]["header"], globals())
    env = eval(config["tonic"]["environment"], globals())
    env = env_wrappers.apply_wrapper(env)
    env.merge_args(config.get("env_args") or {})
    agent = eval(config["tonic"]["agent"], globals())
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space, seed=0)
    agent.load(ckpt[:-3], only_checkpoint=True)
    return env, agent, run_dir


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--run")
    g.add_argument("--experiment")
    ap.add_argument("-n", "--num-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-sto", action="store_true")
    a = ap.parse_args()

    ckpt = newest_checkpoint(a)
    step = re.search(r"step_(\d+)", ckpt).group(1)
    env, agent, run_dir = build(ckpt)
    u = env.unwrapped
    u.init_pos_noise = u.init_vel_noise = 0.0
    out = os.path.join(run_dir, f"mimic_eval_step_{step}")
    os.makedirs(out, exist_ok=True)
    rel = os.path.relpath(out, u.results_dir) if out.startswith(os.path.realpath(u.results_dir)) else None
    if rel is None:   # results/ is reached through ~/SCONE/results/scone-rl (symlink)
        rel = os.path.join("scone-rl", os.path.relpath(out, os.path.join(ROOT, "results")))
    u.set_output_dir(rel)
    # model files next to the .sto so SCONE Studio can play them (config.scone = the scaled model)
    import shutil
    mdir = os.path.dirname(u.ref.model_file)
    for f in os.listdir(mdir):
        if f.endswith((".hfd", ".zml")):
            shutil.copy(os.path.join(mdir, f), out)
    shutil.copy(u.ref.model_file, os.path.join(out, "config.scone"))
    np.random.seed(a.seed)

    dof_names = u.ref.dof_names
    tq_names = [dof_names[j] for j in u.i_torque]
    episodes = []
    for ep in range(a.num_episodes):
        if not a.no_sto:
            u.store_next = True
        obs = env.reset()
        rec = {k: [] for k in ("q", "q_ref", "tau", "tau_ref", "grf", "grf_ref", "cop_err", "terms", "x")}
        done, R = False, 0.0
        while not done:
            act = agent.test_step(obs, muscle_states=env.muscle_states, steps=1e6)
            if act.ndim > 1:
                act = act[0]
            obs, r, done, _ = env.step(act)
            R += r
            e = u.tracking_errors()
            for k in ("q", "q_ref", "tau", "tau_ref", "grf", "grf_ref", "cop_err"):
                rec[k].append(e[k])
            rec["terms"].append(dict(u.rwd_dict))
            rec["x"].append(u.model.com_pos().x)
        if not a.no_sto:
            env.write_now()
        rec = {k: (np.array(v) if k != "terms" else v) for k, v in rec.items()}
        rec["return"], rec["steps"] = R, len(rec["x"])
        rec["fell"] = rec["steps"] < u.max_episode_steps and u.model.com_pos().y < u.min_com_height + 0.05
        episodes.append(rec)
        print(f"episode {ep}: {rec['steps']} steps ({rec['steps'] * u.step_size:.2f} s), return {R:.2f}")

    # ---- report ------------------------------------------------------------
    L = np.array([e["steps"] for e in episodes]) * u.step_size
    lines = [f"checkpoint {ckpt}", f"reference  {u.ref.path}",
             f"episodes {len(episodes)}: duration mean {L.mean():.2f} s (max possible {u.max_episode_steps * u.step_size:.1f} s), "
             f"return mean {np.mean([e['return'] for e in episodes]):.2f} (max ~{u.max_episode_steps * u.max_step_reward():.0f})",
             f"reward settings: torque_mode {u.torque_mode}, grf_mode {u.grf_mode}, "
             f"reference moments low-pass {u.tau_lowpass_hz or 'off'} Hz, w_cop {u.w_cop}"]
    spd = [(e["x"][-1] - e["x"][0]) / max(1e-6, (e["steps"] - 1) * u.step_size) for e in episodes if e["steps"] > 4]
    if spd:
        lines.append(f"COM speed {np.mean(spd):.2f} m/s (reference belt {np.mean(u.ref.belt_speed):.2f} m/s)")
    terms = {k: np.mean([t[k] for e in episodes for t in e["terms"]]) for k in episodes[0]["terms"][0]}
    lines.append("mean reward terms per step: " + ", ".join(f"{k} {v:.3f}" for k, v in terms.items()))
    lines.append("as % of each term's maximum: " + ", ".join(
        f"{k} {100 * v / getattr(u, 'w_' + k):.0f}%" for k, v in terms.items() if v >= 0 and getattr(u, "w_" + k, 0) > 0))
    cat = {k: np.concatenate([e[k] for e in episodes]) for k in ("q", "q_ref", "tau", "tau_ref", "grf", "grf_ref", "cop_err")}
    lines.append(f"\n{'joint angle':<20}{'RMSE deg':>9}{'r':>7}")
    for j, n in enumerate(dof_names):
        if n.startswith("pelvis_t"):
            continue
        lines.append(f"{n:<20}{np.degrees(rmse(cat['q'][:, j], cat['q_ref'][:, j])):>9.1f}{corr(cat['q'][:, j], cat['q_ref'][:, j]):>7.2f}")
    lines.append(f"\n{'joint moment':<20}{'RMSE N.m/kg':>12}{'r':>7}")
    for j, n in enumerate(tq_names):
        lines.append(f"{n:<20}{rmse(cat['tau'][:, j], cat['tau_ref'][:, j]):>12.2f}{corr(cat['tau'][:, j], cat['tau_ref'][:, j]):>7.2f}")
    lines.append(f"\n{'GRF':<20}{'RMSE BW':>9}{'r':>7}")
    for f, foot in enumerate(("right", "left")):
        for c, ax in enumerate("xyz"):
            lines.append(f"{foot + ' ' + ax:<20}{rmse(cat['grf'][:, f, c], cat['grf_ref'][:, f, c]):>9.2f}"
                         f"{corr(cat['grf'][:, f, c], cat['grf_ref'][:, f, c]):>7.2f}")
    lines.append(f"\n{'centre of pressure':<20}{'RMSE cm':>9}   (horizontal, relative to the foot, frames where both are loaded)")
    for f, foot in enumerate(("right", "left")):
        d = cat["cop_err"][:, f]
        d = d[~np.isnan(d[:, 0])]
        if len(d):
            lines.append(f"{foot + ' fore-aft':<20}{100 * np.sqrt(np.mean(d[:, 0] ** 2)):>9.1f}")
            lines.append(f"{foot + ' medio-lateral':<20}{100 * np.sqrt(np.mean(d[:, 1] ** 2)):>9.1f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "report.txt"), "w").write(txt + "\n")

    # ---- plot the longest episode ------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = max(episodes, key=lambda x: x["steps"])
    t = np.arange(e["steps"]) * u.step_size
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for k, n in enumerate(PLOT_DOFS[:3]):
        j = dof_names.index(n)
        axs[0, k].plot(t, np.degrees(e["q_ref"][:, j]), "k", lw=2, label="measured (AB07)")
        axs[0, k].plot(t, np.degrees(e["q"][:, j]), "C1", label="simulated")
        axs[0, k].set_title(f"{n} [deg]")
        jt = tq_names.index(n)
        axs[1, k].plot(t, e["tau_ref"][:, jt], "k", lw=2)
        axs[1, k].plot(t, e["tau"][:, jt], "C0")
        axs[1, k].set_title(f"{n} moment [N.m/kg]")
    for f, foot in enumerate(("right", "left")):
        axs[2, f].plot(t, e["grf_ref"][:, f, 1], "k", lw=2)
        axs[2, f].plot(t, e["grf"][:, f, 1], "C2")
        axs[2, f].set_title(f"vertical GRF {foot} [BW]")
    axs[2, 2].plot(t, [sum(v for k, v in tt.items() if v > 0) for tt in e["terms"]], "C3")
    axs[2, 2].set_title(f"tracking reward per step (max {u.max_step_reward():.2f})")
    for ax in axs.flat:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    axs[0, 0].legend()
    fig.suptitle(f"{os.path.basename(run_dir)}  step {step}: longest episode {e['steps'] * u.step_size:.2f} s")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tracking.png"), dpi=90)
    print(f"wrote {out}/report.txt, tracking.png" + ("" if a.no_sto else ", *.sto"))


if __name__ == "__main__":
    main()
