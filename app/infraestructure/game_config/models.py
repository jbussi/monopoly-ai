"""Modelos da configuração externa do jogo."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Configuração carregada de uma variante do Monopoly."""

    board: list[dict[str, Any]]
    cards: dict[str, Any]
    settings: dict[str, Any]
    variant: str