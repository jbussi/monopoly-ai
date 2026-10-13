from abc import ABC, abstractmethod

from actions import GameAction
from game_state import GameState


class BaseController(ABC):
    """Contrato base para todos os agentes que tomam decisões no jogo."""

    @abstractmethod
    def get_decision(
        self,
        game_state: GameState,
        phase: str,
    ) -> GameAction:
        """
        Analisa o estado atual e retorna a ação escolhida pelo agente.

        Args:
            game_state: Estado atual da partida.
            phase: Fase do jogo em que a decisão será tomada.

        Returns:
            A ação escolhida, representada por GameAction.
        """
        raise NotImplementedError
