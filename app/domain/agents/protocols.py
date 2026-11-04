"""Structural interfaces shared by heuristic and learned game agents."""

from __future__ import annotations

from typing import Protocol

from app.domain.game.actions import GameAction
from app.domain.game.game_loop import GameLoop


class Agent(Protocol):
    """A policy that chooses one engine action for the current decision."""

    def __call__(self, loop: GameLoop) -> GameAction:
        """Return the action chosen for the current engine decision."""
        ...


class SeedableAgent(Agent, Protocol):
    """An agent whose private randomness or episode state can be reset."""

    def reset(self, seed: int | None = None) -> None:
        """Reset episode state and seed the agent's private randomness."""
        ...
