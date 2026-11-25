"""
Run many simulations in parallel.

Layout: num_workers processes, each owning envs_per_worker EnergyWalkEnv instances
(each its own Hyfydy model). Per step, the main process sends every worker its slice
of the action batch; each worker steps its envs one after another and sends back
the results.

Why several envs per worker instead of one? Two measured costs dominate on Windows:
  * inter-process messaging (~0.1-0.2 ms per message), and
  * waiting for the slowest worker: Hyfydy uses a variable-step integrator, so a step
    with a foot impact takes much longer than a swing-phase step.
Stepping several envs per worker halves the messages per env and averages out
those slow steps, which roughly triples throughput compared to one env per process.

Workers reset envs automatically when an episode ends; the observation returned is
then the first observation of the *new* episode, and the last observation of the
finished episode is kept in info["terminal_obs"] (needed for truncation bootstrapping).
"""
import multiprocessing as mp

import numpy as np


def _worker(remote, cfg, seed):
    # Imported inside the worker so each process loads its own sconepy/Hyfydy.
    from .envs import make_env

    np.random.seed(seed)   # sconegym randomizes initial states with np.random
    envs = [make_env(cfg) for _ in range(cfg.envs_per_worker)]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, reward, done, info = env.step(action)
                    if done:
                        info["terminal_obs"] = obs
                        obs = env.reset()
                    out.append((obs, reward, done, info))
                remote.send(out)
            elif cmd == "reset":
                remote.send([env.reset() for env in envs])
            elif cmd == "spaces":
                e = envs[0]
                remote.send((e.observation_space.shape, e.action_space.shape, e.muscle_names))
            elif cmd == "close":
                break
    finally:
        for env in envs:
            env.close()
        remote.close()


class SubprocVecEnv:
    def __init__(self, cfg):
        ctx = mp.get_context("spawn")  # required on Windows, safe everywhere
        self.k = cfg.envs_per_worker
        self.num_envs = cfg.num_workers * self.k
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(cfg.num_workers)])
        self.procs = []
        for i, wr in enumerate(work_remotes):
            p = ctx.Process(target=_worker, args=(wr, cfg, cfg.seed * 1000 + i), daemon=True)
            p.start()
            wr.close()
            self.procs.append(p)
        self.remotes[0].send(("spaces", None))
        self.obs_shape, self.act_shape, self.muscle_names = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([o for r in self.remotes for o in r.recv()])

    def step(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.k:(i + 1) * self.k]))
        results = [res for r in self.remotes for res in r.recv()]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), list(infos)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
