from abc import ABC, abstractmethod
from src.engine.game_state import GameState
from src.engine.actions import GameAction

class BaseController(ABC):
    """Interface Abstrata que todo decisor (Humano ou IA) deve herdar."""
    
    @abstractmethod
    def get_decision(self, game_state: GameState, phase: str) -> GameAction:
        """Analisa o estado e retorna um comando estruturado GameAction."""
        pass