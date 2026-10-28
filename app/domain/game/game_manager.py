from collections.abc import Mapping
from uuid import UUID

from .actions import GameAction
from .game_loop import GameLoop
from .game_state import GameState
from .base_controller import BaseController


class GameManager:
    def __init__(
        self,
        game_state: GameState,
        engine_loop: GameLoop,
        controllers: Mapping[UUID, BaseController],
    ) -> None:
        self.game_state = game_state
        self.engine_loop = engine_loop
        self.controllers = dict(controllers)

        self._validate_controllers()

    def _validate_controllers(self) -> None:
        """Verifica se todos os jogadores possuem um controlador."""

        player_ids = set(self.game_state.players.keys())
        controller_ids = set(self.controllers.keys())

        missing = player_ids - controller_ids

        if missing:
            raise ValueError(
                f"Não existem controladores para os jogadores: {missing}"
            )

    def step(self) -> dict:
        """Executa uma única ação na partida."""

        if self.game_state.game_over:
            raise RuntimeError("A partida já terminou.")

        # Durante um leilão, outro jogador pode precisar agir.
        player = self.engine_loop.current_decision_player
        controller = self.controllers[player.player_id]

        action = controller.get_decision(
            self.game_state,
            phase=self.engine_loop.turn_phase,
        )

        if not isinstance(action, GameAction):
            raise TypeError(
                "O controlador deve retornar uma instância de GameAction."
            )

        return self.engine_loop.process_action(action)

    def run_game(self, max_actions: int = 100_000) -> GameState:
        """Executa a partida até o fim ou até atingir o limite de ações."""

        if max_actions < 1:
            raise ValueError("max_actions deve ser maior que zero.")

        actions_executed = 0

        while not self.game_state.game_over:
            if actions_executed >= max_actions:
                raise RuntimeError(
                    "A partida excedeu o limite de ações. "
                    "Verifique se algum controlador está repetindo ações inválidas."
                )

            self.step()
            actions_executed += 1

        return self.game_state