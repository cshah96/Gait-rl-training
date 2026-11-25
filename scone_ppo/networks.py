"""
The neural networks: an actor (policy) and a critic (value function).

    actor : normalized observation (121) -> mean muscle excitation (18)
            + a learned, state-independent log standard deviation (18)
    critic: normalized observation (121) -> expected future reward (1 number)

They are separate networks (no shared layers), which is the most robust choice
for continuous control with PPO.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

_HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


def _layer(n_in, n_out, gain=np.sqrt(2)):
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, init_log_std=-1.5, init_action_mean=0.1):
        super().__init__()
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, gain=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, act_dim, gain=0.01),   # tiny weights: start near the bias
        )
        # start every muscle at a small excitation instead of 0 (0 would be clipped half the time)
        nn.init.constant_(self.actor_mean[-1].bias, init_action_mean)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs, deterministic=False):
        """
        Sample an action. Returns (raw_action, log_prob, value).
        Same math as distribution(obs).sample()/log_prob(), written out by hand because
        it is called every simulation step and torch.distributions adds ~1 ms overhead.
        """
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std
        action = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        log_prob = (-0.5 * ((action - mean) / log_std.exp()) ** 2 - log_std - _HALF_LOG_2PI).sum(-1)
        return action, log_prob, self.value(obs)

    def evaluate(self, obs, action):
        """Log-prob, entropy and value of given actions (used in the PPO update)."""
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs)


def to_env_action(raw_action):
    """
    The policy outputs an unbounded Gaussian sample; muscles need excitation in [0, 1].
    We clip here (sconegym additionally clips walking envs to [0, 0.5]).
    The *raw* sample is what PPO stores, so log-probabilities stay exact.
    """
    return np.clip(raw_action, 0.0, 1.0)
