from typing import List, Any
from src.engine.game_state import GameState
from src.engine.game_loop import GameLoop
from src.engine.actions import GameAction
from src.controllers.base_controller import BaseController

class GameManager:
    def __init__(self, player_names: List[str], controllers: List[BaseController]):
        self.game_state = GameState(player_names)
        self.engine_loop = GameLoop(self.game_state)
        self.controllers = controllers 

    def run_game(self):
        while not self.game_state.game_over:
            self.step_turn()
        print(f"🏆 Fim de Jogo! Vencedor: {self.game_state.winner.name}")

    def step_turn(self):
        active_player = self.game_state.current_player
        player_id = active_player.player_id
        controller = self.controllers[player_id]

        # --- FASE 1: ANTES DA ROLAGEM ---
        while self.engine_loop.turn_phase == "BEFORE_ROLL":
            action: GameAction = controller.get_decision(self.game_state, phase="BEFORE_ROLL")
            if action.action_type == "ROLL_DICE":
                self.engine_loop.process_action(action)
                break
            else:
                self.engine_loop.process_action(action)

        # --- FASE 2: RESOLUÇÃO DE TERRENO ---
        if self.engine_loop.turn_phase == "AWAITING_PROPERTY_DECISION":
            action: GameAction = controller.get_decision(self.game_state, phase="AWAITING_PROPERTY_DECISION")
            if action.action_type in ["BUY_PROPERTY", "DECLINE_PROPERTY"]:
                self.engine_loop.process_action(action)
            else:
                self.engine_loop.process_action(GameAction("DECLINE_PROPERTY"))

        # --- FASE 3: PÓS-ROLAGEM ---
        while self.engine_loop.turn_phase == "AFTER_ROLL":
            action: GameAction = controller.get_decision(self.game_state, phase="AFTER_ROLL")
            if action.action_type == "END_TURN":
                self.engine_loop.process_action(action)
                break
            else:
                self.engine_loop.process_action(action)