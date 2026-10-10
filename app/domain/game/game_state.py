from typing import List, Dict, Any, Optional
from src.engine.player import Player
from src.engine.board import Board
from src.engine.dice import Dice

class GameState:
    def __init__(self, player_names: List[str]):
        self.players: List[Player] = [Player(i, name) for i, name in enumerate(player_names)]
        self.board: Board = Board()
        self.dice: Dice = Dice()
        
        self.current_player_idx: int = 0
        self.game_over: bool = False
        self.winner: Optional[Player] = None
        
        # Telemetria para análise estatística e simulação
        self.history_logs: List[Dict[str, Any]] = []
        self.turn_counter: int = 1

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def next_turn(self):
        """Passa o controle para o próximo jogador ativo na mesa."""
        if self.game_over:
            return

        # Reseta o contador de duplos se o turno mudou de jogador
        self.dice.reset_counters()

        attempts = 0
        while attempts < len(self.players):
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if self.current_player_idx == 0:
                self.turn_counter += 1
                
            if not self.players[self.current_player_idx].is_bankrupt:
                return
            attempts += 1

    def force_bankruptcy(self, player: Player, creditor_id: Optional[int]):
        """Aplica a falência imediata e redistribui ou liquida os bens."""
        player.is_bankrupt = True
        self.log_event("BANKRUPTCY_TRIGGERED", {"player_id": player.player_id, "creditor_id": creditor_id})
        
        if creditor_id is not None:
            creditor = self.players[creditor_id]
            # Transfere as propriedades para o credor
            for tile_id in player.properties:
                tile = self.board.get_tile(tile_id)
                if hasattr(tile, 'owner_id'):
                    tile.owner_id = creditor_id
                creditor.properties.append(tile_id)
        else:
            # Propriedades voltam a leilão/banco
            for tile_id in player.properties:
                tile = self.board.get_tile(tile_id)
                if hasattr(tile, 'owner_id'):
                    tile.owner_id = None
                    tile.houses = 0
                    tile.is_mortgaged = False

        player.reset_assets_on_bankruptcy()
        self._check_victory_condition()

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Grava telemetria estruturada que alimentará nossa análise de dados."""
        log_entry = {
            "turn": self.turn_counter,
            "current_player_id": self.current_player_idx,
            "event": event_type,
            "data": payload
        }
        self.history_logs.append(log_entry)

    def _check_victory_condition(self):
        active_players = [p for p in self.players if not p.is_bankrupt]
        if len(active_players) == 1:
            self.game_over = True
            self.winner = active_players[0]
            self.log_event("GAME_FINISHED", {"winner_id": self.winner.player_id})