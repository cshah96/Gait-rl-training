"""Train a walking policy for sconewalk_h0918-v1 with PPO (scone_ppo package).

    python scripts/ppo_train.py                                   # proven defaults: 20M steps, ~1 h
    python scripts/ppo_train.py --total_steps 2000000 --run_name quick
    python scripts/ppo_train.py --effort_coeff 20 --run_name effort20

Every field of scone_ppo/config.py can be overridden as --<field> <value>.

Outputs go to results/ppo/<YYYYMMDD_HHMMSS>_<run_name>/:
    config.json                 settings used for this run
    progress.csv                one row per PPO update (plot with scripts/ppo_plot.py)
    latest.pt                   most recent checkpoint (overwritten every update)
    best.pt                     checkpoint with the highest average episode return so far
    checkpoints/step_<N>.pt     periodic snapshots (N = env steps); double-click these in SCONE Studio
"""
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scone_ppo.config import parse_config  # noqa: E402
from scone_ppo.networks import ActorCritic, to_env_action  # noqa: E402
from scone_ppo.normalization import ObsNormalizer, RewardScaler  # noqa: E402
from scone_ppo.ppo import RolloutBuffer, ppo_update  # noqa: E402
from scone_ppo.vec_env import SubprocVecEnv  # noqa: E402

EPISODE_KEYS = ["return", "length", "distance_m", "speed_ms", "speed_error", "mean_effort", "effort_per_m", "fell",
                "r_gaussian_vel", "r_grf", "r_smooth", "r_number_muscles", "r_constr", "r_self_contact",
                "r_effort"]


def save_checkpoint(path, agent, obs_norm, cfg, update, global_step, score):
    torch.save({
        "model": agent.state_dict(),
        "obs_norm": obs_norm.rms.state_dict(),
        "config": vars(cfg),
        "update": update,
        "global_step": global_step,
        "score": score,
    }, path)


def main():
    cfg = parse_config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)   # a few cores for the network; the rest run simulations

    runs_dir = cfg.runs_dir or os.path.join(ROOT, "results", "ppo")
    run_dir = os.path.join(runs_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg.to_json(os.path.join(run_dir, "config.json"))
    print(f"Run folder: {run_dir}")

    # ---- environments ----
    envs = SubprocVecEnv(cfg)
    obs_dim, act_dim = envs.obs_shape[0], envs.act_shape[0]
    print(f"{cfg.num_envs} x {cfg.env_id}: obs_dim={obs_dim}, act_dim={act_dim} ({', '.join(envs.muscle_names)})")

    # ---- agent ----
    device = torch.device(cfg.device)
    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size, cfg.init_log_std, cfg.init_action_mean).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer(obs_dim)
    rew_scaler = RewardScaler(cfg.num_envs, cfg.gamma)
    buf = RolloutBuffer(cfg.n_steps, cfg.num_envs, obs_dim, act_dim, device)

    batch = cfg.n_steps * cfg.num_envs
    num_updates = max(1, cfg.total_steps // batch)
    print(f"{num_updates} updates of {batch} steps = {num_updates * batch:,} steps "
          f"({num_updates * batch * 0.01 / 3600:.1f} h of simulated walking)", flush=True)

    # ---- logging ----
    csv_file = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
    fieldnames = ["update", "global_step", "time_s", "fps", "lr", "episodes", "rollout_s", "update_s",
                  *[f"ep_{k}" for k in EPISODE_KEYS],
                  "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
                  "explained_var", "epochs_done", "action_std"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    recent = deque(maxlen=50)      # last 50 finished episodes
    best_score = -np.inf
    total_episodes = 0

    obs = obs_norm(envs.reset())
    global_step, t_start = 0, time.time()
    update = 0
    try:
        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                lr = cfg.learning_rate * (1.0 - (update - 1) / num_updates)
                optimizer.param_groups[0]["lr"] = lr

            # ---------------- 1. collect rollout ----------------
            t_rollout = time.time()
            for t in range(cfg.n_steps):
                obs_t = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    action, log_prob, value = agent.act(obs_t)
                raw = action.cpu().numpy()
                next_obs, rewards, dones, infos = envs.step(to_env_action(raw))
                global_step += cfg.num_envs

                scaled = rew_scaler(rewards, dones)
                for i, info in enumerate(infos):
                    # Time-limit truncation is not a failure: bootstrap with the value of the
                    # state we were cut off in, so the agent isn't punished for "running out of time".
                    if info.get("truncated"):
                        term = torch.as_tensor(obs_norm(info["terminal_obs"][None], update=False), device=device)
                        with torch.no_grad():
                            scaled[i] += cfg.gamma * agent.value(term).item()
                    if "episode" in info:
                        recent.append(info["episode"])
                        total_episodes += 1

                buf.obs[t] = obs_t
                buf.actions[t] = action
                buf.log_probs[t] = log_prob
                buf.values[t] = value
                buf.rewards[t] = torch.as_tensor(scaled, device=device)
                buf.dones[t] = torch.as_tensor(dones, dtype=torch.float32, device=device)
                obs = obs_norm(next_obs)

            t_rollout = time.time() - t_rollout

            # ---------------- 2. advantages ----------------
            t_update = time.time()
            with torch.no_grad():
                last_value = agent.value(torch.as_tensor(obs, device=device))
            buf.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

            # ---------------- 3. PPO update ----------------
            stats = ppo_update(agent, optimizer, buf, cfg)
            t_update = time.time() - t_update

            # ---------------- logging ----------------
            elapsed = time.time() - t_start
            row = {"update": update, "global_step": global_step, "time_s": round(elapsed, 1),
                   "fps": int(global_step / elapsed), "lr": optimizer.param_groups[0]["lr"],
                   "episodes": total_episodes, "rollout_s": round(t_rollout, 2), "update_s": round(t_update, 2),
                   "action_std": agent.actor_log_std.exp().mean().item(), **stats}
            for k in EPISODE_KEYS:
                vals = [e[k] for e in recent if not np.isnan(e[k])]
                row[f"ep_{k}"] = float(np.mean(vals)) if vals else float("nan")
            writer.writerow(row)
            csv_file.flush()

            if update % cfg.log_interval == 0:
                print(f"upd {update:4d}/{num_updates} | steps {global_step:>10,} | fps {row['fps']:5d} | "
                      f"return {row['ep_return']:8.1f} | len {row['ep_length']:6.1f} | "
                      f"speed {row['ep_speed_ms']:5.2f}+-{row['ep_speed_error']:.2f} m/s | effort {row['ep_mean_effort']:.4f} | "
                      f"fell {row['ep_fell']:.2f} | kl {stats['approx_kl']:.4f} | ev {stats['explained_var']:.2f} | "
                      f"t {t_rollout:.1f}+{t_update:.1f}s",
                      flush=True)

            # ---------------- checkpoints ----------------
            score = row["ep_return"]
            save_checkpoint(os.path.join(run_dir, "latest.pt"), agent, obs_norm, cfg, update, global_step, score)
            if len(recent) >= 10 and score > best_score:
                best_score = score
                save_checkpoint(os.path.join(run_dir, "best.pt"), agent, obs_norm, cfg, update, global_step, score)
            if update % cfg.save_interval == 0 or update == num_updates:
                save_checkpoint(os.path.join(ckpt_dir, f"step_{global_step}.pt"),
                                agent, obs_norm, cfg, update, global_step, score)
    except KeyboardInterrupt:
        print("\nInterrupted - saving latest.pt")
        save_checkpoint(os.path.join(run_dir, "latest.pt"), agent, obs_norm, cfg, update, global_step, float("nan"))
    finally:
        csv_file.close()
        envs.close()
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":   # required for multiprocessing (spawn)
    main()
