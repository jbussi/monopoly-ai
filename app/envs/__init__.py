"""Adaptadores da engine para interfaces padrão de RL (PettingZoo, Gymnasium) e texto."""

from importlib import import_module

from .action_space import DiscreteActionSpace
from .baselines import GreedyPolicy, Policy, RandomPolicy
from .cash_flow_forecast import (
    CashFlowForecast,
    CashFlowHorizonForecast,
    forecast_cash_flow,
)
from .encoding import ObservationEncoder
from .rewards import RewardFn, win_loss_reward
from .text import describe_action, render_text

ENV_VERSION = 1
ENV_VERSION_V2 = 2

__all__ = [
    "DiscreteActionSpace",
    "CashFlowForecast",
    "CashFlowHorizonForecast",
    "CandidateSelectionEnv",
    "ENV_VERSION",
    "ENV_VERSION_V2",
    "GreedyPolicy",
    "GymEpisodeResult",
    "MonopolyGymEnv",
    "MonopolyGymEnvV2",
    "PettingZooEpisodeResult",
    "ObservationEncoder",
    "ObservationEncoderV2",
    "Policy",
    "RandomPolicy",
    "RewardFn",
    "StructuredActionSpace",
    "describe_action",
    "forecast_cash_flow",
    "render_text",
    "run_gymnasium_episode",
    "run_pettingzoo_episode",
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
    if name == "CandidateSelectionEnv":
        from .candidate_env import CandidateSelectionEnv

        return CandidateSelectionEnv
    if name == "ObservationEncoderV2":
        from .observation_v2 import ObservationEncoderV2

        return ObservationEncoderV2
    if name == "StructuredActionSpace":
        from .action_space_v2 import StructuredActionSpace

        return StructuredActionSpace
    if name in {
        "GymEpisodeResult",
        "PettingZooEpisodeResult",
        "run_gymnasium_episode",
        "run_pettingzoo_episode",
    }:
        return getattr(import_module(".rollout", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
