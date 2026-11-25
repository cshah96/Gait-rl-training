"""
Proximal Policy Optimization (PPO), written out plainly.

One training iteration =
  1. collect a rollout: run the current policy in all envs for n_steps
  2. compute advantages with GAE (how much better each action was than expected)
  3. update actor + critic for n_epochs over shuffled minibatches,
     with the PPO "clip" keeping the new policy close to the old one
"""
import numpy as np
import torch
import torch.nn as nn


class RolloutBuffer:
    """Stores one rollout of shape [n_steps, num_envs, ...]."""

    def __init__(self, n_steps, num_envs, obs_dim, act_dim, device):
        def z(*shape):
            return torch.zeros((n_steps, num_envs) + shape, device=device)
        self.obs = z(obs_dim)
        self.actions = z(act_dim)
        self.log_probs = z()
        self.rewards = z()
        self.dones = z()      # 1 if the episode ended after this step
        self.values = z()
        self.advantages = z()
        self.returns = z()
        self.n_steps = n_steps

    def compute_gae(self, last_value, gamma, lam):
        """
        Generalized Advantage Estimation.
          delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
        returns_t = A_t + V(s_t) is the target for the critic.
        """
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values


def ppo_update(agent, optimizer, buf, cfg):
    """Run the PPO optimization on one filled RolloutBuffer. Returns a dict of stats."""
    obs = buf.obs.reshape(-1, buf.obs.shape[-1])
    actions = buf.actions.reshape(-1, buf.actions.shape[-1])
    old_log_probs = buf.log_probs.reshape(-1)
    advantages = buf.advantages.reshape(-1)
    returns = buf.returns.reshape(-1)
    old_values = buf.values.reshape(-1)
    batch_size = obs.shape[0]

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    stop = False
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(batch_size)
        for start in range(0, batch_size, cfg.minibatch_size):
            idx = perm[start:start + cfg.minibatch_size]
            new_log_prob, entropy, new_value = agent.evaluate(obs[idx], actions[idx])

            # probability ratio between new and old policy for the taken actions
            log_ratio = new_log_prob - old_log_probs[idx]
            ratio = log_ratio.exp()

            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # clipped surrogate objective (the core of PPO)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            # critic: regress to the GAE returns (with value clipping)
            v_clipped = old_values[idx] + torch.clamp(new_value - old_values[idx], -cfg.clip_coef, cfg.clip_coef)
            value_loss = 0.5 * torch.max((new_value - returns[idx]) ** 2,
                                         (v_clipped - returns[idx]) ** 2).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop = True
                break
        if stop:
            break

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["epochs_done"] = epoch + 1
    # explained variance: 1 = critic predicts returns perfectly, 0 = useless
    var_ret = returns.var().item()
    out["explained_var"] = float(1 - (returns - old_values).var().item() / var_ret) if var_ret > 0 else 0.0
    return out
