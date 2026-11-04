"""Protocols for agents that consume RL observations and action masks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class RLAgent(Protocol):
    """Observation-based actor compatible with an environment action space."""

    def act(
        self,
        observation: Mapping[str, Any],
        action_mask: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> Any:
        """Choose an action from an environment observation and legal mask."""
        ...


class CandidateRLAgent(Protocol):
    """Actor for candidate-index environments; returns a legal row index."""

    def act(
        self,
        observation: Mapping[str, Any],
        action_mask: Any,
        *,
        deterministic: bool = False,
    ) -> int:
        """Choose an active candidate row using the supplied legality mask."""
        ...


class ResettableRLAgent(RLAgent, Protocol):
    """An RL actor with episode-local randomness or recurrent state."""

    def reset(self, seed: int | None = None) -> None:
        """Reset recurrent state and episode randomness for this actor."""
        ...
