"""Adaptadores da engine para interfaces padrão de RL (PettingZoo, Gymnasium) e texto."""

from .action_space import DiscreteActionSpace
from .baselines import GreedyPolicy, Policy, RandomPolicy
from .encoding import ObservationEncoder
from .rewards import RewardFn, win_loss_reward
from .text import describe_action, render_text

ENV_VERSION = 1
ENV_VERSION_V2 = 2

__all__ = [
    "DiscreteActionSpace",
    "ENV_VERSION",
    "ENV_VERSION_V2",
    "GreedyPolicy",
    "MonopolyGymEnv",
    "MonopolyGymEnvV2",
    "ObservationEncoder",
    "ObservationEncoderV2",
    "Policy",
    "RandomPolicy",
    "RewardFn",
    "StructuredActionSpace",
    "describe_action",
    "render_text",
    "win_loss_reward",
]


def __getattr__(name: str):
    """Carrega o adaptador Gymnasium somente quando ele é solicitado."""
    if name == "MonopolyGymEnv":
        from .gymnasium_env import MonopolyGymEnv

        return MonopolyGymEnv
    if name == "MonopolyGymEnvV2":
        from .gymnasium_env_v2 import MonopolyGymEnvV2

        return MonopolyGymEnvV2
    if name == "ObservationEncoderV2":
        from .observation_v2 import ObservationEncoderV2

        return ObservationEncoderV2
    if name == "StructuredActionSpace":
        from .action_space_v2 import StructuredActionSpace

        return StructuredActionSpace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
