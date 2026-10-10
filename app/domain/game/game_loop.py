from typing import Dict, Any
from src.engine.game_state import GameState
from src.engine.board import PropertyTile, SpecialTile, TaxTile
from src.engine.actions import GameAction

class GameLoop:
    def __init__(self, game_state: GameState):
        self.game: GameState = game_state
        # Fases: 'BEFORE_ROLL', 'AWAITING_PROPERTY_DECISION', 'AFTER_ROLL'
        self.turn_phase: str = "BEFORE_ROLL" 

    def process_action(self, action: GameAction) -> Dict[str, Any]:
        """Processa a ação do jogador atual, valida as regras e atualiza o estado."""
        player = self.game.current_player
        
        if player.is_bankrupt or self.game.game_over:
            return {"status": "REJECTED", "reason": "Game over or player bankrupt"}

        # --- AÇÃO: USAR CARTA DE PRISÃO ---
        if action.action_type == "USE_JAIL_CARD":
            if player.is_in_jail and player.has_jail_card > 0:
                player.has_jail_card -= 1
                player.is_in_jail = False
                player.jail_turns = 0
                self.game.log_event("JAIL_ESCAPE_CARD", {"player_id": player.player_id})
                return {"status": "SUCCESS", "phase": self.turn_phase}
            return {"status": "REJECTED", "reason": "Cannot use jail card now"}

        # --- AÇÃO: ROLAR DADOS ---
        elif action.action_type == "ROLL_DICE":
            if self.turn_phase != "BEFORE_ROLL":
                return {"status": "REJECTED", "reason": "Dice already rolled this turn"}
                
            if player.is_in_jail:
                return self._execute_jail_roll(player)
            else:
                return self._execute_standard_movement(player)

        # --- AÇÃO: COMPRAR PROPRIEDADE ---
        elif action.action_type == "BUY_PROPERTY":
            if self.turn_phase != "AWAITING_PROPERTY_DECISION":
                return {"status": "REJECTED", "reason": "No property available for purchase"}
                
            tile = self.game.board.get_tile(player.position)
            if isinstance(tile, PropertyTile) and tile.owner_id is None and player.money >= tile.price:
                player.buy_property(tile.tile_id, tile.price)
                tile.owner_id = player.player_id
                self.game.log_event("PROPERTY_PURCHASE", {"tile_id": tile.tile_id, "price": tile.price})
                self.turn_phase = "AFTER_ROLL"
                return {"status": "SUCCESS", "phase": self.turn_phase}
            return {"status": "REJECTED", "reason": "Ineligible to buy property"}

        # --- AÇÃO: DECLINAR COMPRA ---
        elif action.action_type == "DECLINE_PROPERTY":
            if self.turn_phase != "AWAITING_PROPERTY_DECISION":
                return {"status": "REJECTED", "reason": "No active property decision pending"}
            self.game.log_event("PROPERTY_PASSED", {"tile_id": player.position})
            self.turn_phase = "AFTER_ROLL"
            return {"status": "SUCCESS", "phase": self.turn_phase}

        # --- AÇÃO: ENCERRAR TURNO ---
        elif action.action_type == "END_TURN":
            if self.turn_phase not in ["BEFORE_ROLL", "AFTER_ROLL"]:
                return {"status": "REJECTED", "reason": "Cannot end turn with pending decisions"}
            self.game.next_turn()
            self.turn_phase = "BEFORE_ROLL"
            return {"status": "SUCCESS", "phase": self.turn_phase}

        return {"status": "REJECTED", "reason": "Unknown action type"}

    def _execute_standard_movement(self, player: Any) -> Dict[str, Any]:
        self.game.dice.roll()
        dice_total = self.game.dice.total
        is_double = self.game.dice.is_double()

        if is_double and self.game.dice.double_count == 3:
            player.position = 10
            player.is_in_jail = True
            self.game.log_event("GO_TO_JAIL_DOUBLES", {"player_id": player.player_id})
            self.game.next_turn()
            self.turn_phase = "BEFORE_ROLL"
            return {"status": "SUCCESS", "event": "SENT_TO_JAIL_DOUBLES", "phase": self.turn_phase}

        old_position = player.position
        new_position = (old_position + dice_total) % 40
        player.position = new_position

        if new_position < old_position and old_position != 0:
            player.receive_money(200)
            self.game.log_event("PASSED_GO", {"salary": 200})

        current_tile = self.game.board.get_tile(new_position)
        return self._evaluate_tile_arrival(player, current_tile, is_double)

    def _evaluate_tile_arrival(self, player: Any, tile: Any, is_double: bool) -> Dict[str, Any]:
        if isinstance(tile, PropertyTile):
            if tile.owner_id is None:
                self.turn_phase = "AWAITING_PROPERTY_DECISION"
                return {"status": "SUCCESS", "event": "AWAITING_BUY_OR_DECLINE", "tile_id": tile.tile_id}
            
            elif tile.owner_id != player.player_id and not tile.is_mortgaged:
                rent = tile.current_rent
                if player.money < rent:
                    self.game.force_bankruptcy(player, creditor_id=tile.owner_id)
                    self.turn_phase = "BEFORE_ROLL"
                    return {"status": "SUCCESS", "event": "BANKRUPTCY", "phase": self.turn_phase}
                else:
                    player.pay_money(rent)
                    self.game.players[tile.owner_id].receive_money(rent)
                    self.game.log_event("RENT_PAYMENT", {"tile_id": tile.tile_id, "amount": rent, "to_player_id": tile.owner_id})

        elif isinstance(tile, TaxTile):
            if player.money < tile.tax_amount:
                self.game.force_bankruptcy(player, creditor_id=None)
                self.turn_phase = "BEFORE_ROLL"
                return {"status": "SUCCESS", "event": "BANKRUPTCY", "phase": self.turn_phase}
            else:
                player.pay_money(tile.tax_amount)
                self.game.log_event("TAX_PAYMENT", {"tile_id": tile.tile_id, "amount": tile.tax_amount})

        elif isinstance(tile, SpecialTile):
            if tile.tile_type == "GO_TO_JAIL":
                player.position = 10
                player.is_in_jail = True
                self.game.log_event("GO_TO_JAIL_TILE", {"player_id": player.player_id})
                self.game.next_turn()
                self.turn_phase = "BEFORE_ROLL"
                return {"status": "SUCCESS", "event": "SENT_TO_JAIL", "phase": self.turn_phase}

        if is_double and not player.is_in_jail and not player.is_bankrupt:
            self.turn_phase = "BEFORE_ROLL" 
        else:
            self.turn_phase = "AFTER_ROLL"

        return {"status": "SUCCESS", "phase": self.turn_phase}

    def _execute_jail_roll(self, player: Any) -> Dict[str, Any]:
        player.jail_turns += 1
        self.game.dice.roll()
        
        if self.game.dice.is_double():
            player.is_in_jail = False
            player.jail_turns = 0
            self.game.log_event("JAIL_ESCAPE_DOUBLE", {"player_id": player.player_id})
            return self._execute_standard_movement(player)
            
        if player.jail_turns >= 3:
            if player.money >= 50:
                player.pay_money(50)
                player.is_in_jail = False
                player.jail_turns = 0
                self.game.log_event("JAIL_FORCED_PAY", {"player_id": player.player_id})
                return self._execute_standard_movement(player)
            else:
                self.game.force_bankruptcy(player, creditor_id=None)
                self.turn_phase = "BEFORE_ROLL"
                return {"status": "SUCCESS", "event": "BANKRUPTCY"}
                
        self.game.next_turn()
        self.turn_phase = "BEFORE_ROLL"
        return {"status": "SUCCESS", "event": "STAYED_IN_JAIL", "phase": self.turn_phase}