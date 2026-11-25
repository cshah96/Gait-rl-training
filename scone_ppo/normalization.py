"""
Running normalization of observations and rewards.

The 121 observation values have wildly different scales (muscle forces ~1,
joint velocities ~10 rad/s, ...). Neural networks learn much better when every
input is roughly zero-mean / unit-variance, so we keep running statistics and
normalize on the fly. These statistics are saved in each checkpoint, because a
trained policy only works with the same normalization it was trained with.
"""
import numpy as np


class RunningMeanStd:
    """Tracks mean and variance of a stream of batches (parallel Welford algorithm)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ObsNormalizer:
    def __init__(self, shape, clip=10.0):
        self.rms = RunningMeanStd(shape)
        self.clip = clip

    def __call__(self, obs, update=True):
        if update:
            self.rms.update(obs)
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8),
                       -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    """
    Divides rewards by the running std of the discounted return.
    Keeps value-function targets in a sane range without changing which
    behaviour is optimal (scaling a reward does not change the best policy).
    """

    def __init__(self, num_envs, gamma):
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(num_envs)
        self.gamma = gamma

    def __call__(self, rewards, dones):
        self.ret = self.ret * self.gamma + rewards
        self.rms.update(self.ret)
        self.ret[dones] = 0.0
        return (rewards / np.sqrt(self.rms.var + 1e-8)).astype(np.float32)
