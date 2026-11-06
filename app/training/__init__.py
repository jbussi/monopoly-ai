"""Treinamento experimental de políticas para os ambientes de RL."""

from .ppo import (
    PPOTrainingConfig,
    evaluate_ppo_matchups,
    load_ppo_checkpoint,
    train_ppo,
)
from .league import (
    CallableTrainingLeague,
    LeagueConfig,
    MatchTable,
    PlayerRating,
    PromotionGateState,
    TrainingLeague,
    placement_ranks,
    sample_num_players,
    schedule_table,
)

__all__ = [
    "PPOTrainingConfig",
    "TrainingLeague",
    "CallableTrainingLeague",
    "LeagueConfig",
    "MatchTable",
    "PlayerRating",
    "PromotionGateState",
    "placement_ranks",
    "sample_num_players",
    "schedule_table",
    "evaluate_ppo_matchups",
    "load_ppo_checkpoint",
    "train_ppo",
]
