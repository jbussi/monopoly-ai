"""Adaptadores da engine para interfaces padrão de RL (PettingZoo, Gymnasium) e texto."""

from .action_space import DiscreteActionSpace
from .baselines import GreedyPolicy, Policy, RandomPolicy
from .encoding import ObservationEncoder
from .gymnasium_env import MonopolyGymEnv
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
