"""Adaptadores da engine para interfaces padrão de RL (PettingZoo, Gymnasium) e texto."""

from .action_space import DiscreteActionSpace
from .baselines import GreedyPolicy, Policy, RandomPolicy
from .encoding import ObservationEncoder
from .rewards import RewardFn, win_loss_reward
from .text import describe_action, render_text

ENV_VERSION = 1

__all__ = [
    "DiscreteActionSpace",
    "ENV_VERSION",
    "GreedyPolicy",
    "MonopolyGymEnv",
    "ObservationEncoder",
    "Policy",
    "RandomPolicy",
    "RewardFn",
    "describe_action",
    "render_text",
    "win_loss_reward",
]


def __getattr__(name: str):
    """Carrega o adaptador Gymnasium somente quando ele é solicitado."""
    if name == "MonopolyGymEnv":
        from .gymnasium_env import MonopolyGymEnv

        return MonopolyGymEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
