"""Policies and model adapters for game agents.

Exports are loaded lazily so lightweight agent protocols can be used by the
environment without importing analytics and forecast modules as a side effect.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .feature_policy import (
        CandidateEvaluator,
        CandidateScorer,
        CandidateSelector,
        FeatureCandidate,
        FeaturePolicy,
        MaxScoreSelector,
    )
    from .protocols import Agent, SeedableAgent
    from .rl_agent import CandidateRLAgent, RLAgent, ResettableRLAgent
    from .trade_candidates import LegalCandidateProvider, TradeCandidateProvider

__all__ = [
    "Agent",
    "CandidateEvaluator",
    "CandidateRLAgent",
    "CandidateScorer",
    "CandidateSelector",
    "FeatureCandidate",
    "FeaturePolicy",
    "LegalCandidateProvider",
    "MaxScoreSelector",
    "RLAgent",
    "ResettableRLAgent",
    "SeedableAgent",
    "TradeCandidateProvider",
]


def __getattr__(name: str):
    if name in {"Agent", "SeedableAgent"}:
        return getattr(import_module(".protocols", __name__), name)
    if name in {"CandidateRLAgent", "RLAgent", "ResettableRLAgent"}:
        return getattr(import_module(".rl_agent", __name__), name)
    if name in {"LegalCandidateProvider", "TradeCandidateProvider"}:
        return getattr(import_module(".trade_candidates", __name__), name)
    if name in {
        "CandidateEvaluator",
        "CandidateScorer",
        "CandidateSelector",
        "FeatureCandidate",
        "FeaturePolicy",
        "MaxScoreSelector",
    }:
        return getattr(import_module(".feature_policy", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
