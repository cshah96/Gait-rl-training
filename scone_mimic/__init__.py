"""Gait imitation (mimic) environments for sconegym / depRL.

    import gym, scone_mimic
    env = gym.make("sconemimic_h2190-v1", reference="data/mimic/AB07/AB07_reference.npz")

The reference .npz (from scripts/mimic_prepare_subject.sh) also names the scaled model it was
built for, so the subject's model is loaded automatically.
"""
from gym.envs.registration import register, registry

from .env import DEFAULT_MIMIC_ARGS, Reference, SconeMimicGym  # noqa: F401

_H2190_KWARGS = dict(
    obs_type="3D",
    left_leg_idxs=[6, 7, 8, 9, 10, 11],
    right_leg_idxs=[12, 13, 14, 15, 16, 17],
    clip_actions=False,
    leg_switch=False,
    init_activations_mean=0.05,
    init_activations_std=0.0,
)

if "sconemimic_h2190-v1" not in getattr(registry, "env_specs", registry):
    register(id="sconemimic_h2190-v1", entry_point="scone_mimic.env:SconeMimicGym", kwargs=_H2190_KWARGS)
