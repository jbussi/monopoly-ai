import copy
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .actions import ActionType, GameAction
from .board import PropertyKind, PropertyTile, SpecialTile, TaxTile
from .card_actions import (
    CardAction,
    CardActionRegistry,
    CardExecutionResult,
    CardLike,
)
from .game_state import HOTEL_LEVEL, Debt, GameState

# Incrementos sobre o maior lance que `legal_actions` oferece num leilão.
# `process_action` aceita qualquer lance inteiro válido; a lista existe para
# dar um conjunto finito de opções a agentes.
AUCTION_BID_INCREMENTS: tuple[int, ...] = (1, 10, 50, 100)

MANAGEMENT_ACTIONS = (
    ActionType.BUILD_HOUSE,
    ActionType.SELL_HOUSE,
    ActionType.MORTGAGE,
    ActionType.UNMORTGAGE,
)


class GameLoop:
    """Coordena ações, fases do turno e aplicação das regras."""

    BEFORE_ROLL = "BEFORE_ROLL"
    AWAITING_PROPERTY_DECISION = "AWAITING_PROPERTY_DECISION"
    AUCTION = "AUCTION"
    AFTER_ROLL = "AFTER_ROLL"
    RESOLVE_DEBT = "RESOLVE_DEBT"
    GAME_OVER = "GAME_OVER"

    PHASE_ACTIONS: dict[str, tuple[str, ...]] = {
        BEFORE_ROLL: (
            ActionType.ROLL_DICE,
            ActionType.PAY_JAIL_FINE,
            ActionType.USE_JAIL_CARD,
            *MANAGEMENT_ACTIONS,
        ),
        AWAITING_PROPERTY_DECISION: (
            ActionType.BUY_PROPERTY,
            ActionType.DECLINE_PROPERTY,
            *MANAGEMENT_ACTIONS,
        ),
        AFTER_ROLL: (ActionType.END_TURN, *MANAGEMENT_ACTIONS),
        AUCTION: (ActionType.AUCTION_BID, ActionType.AUCTION_WITHDRAW),
        RESOLVE_DEBT: (
            ActionType.SELL_HOUSE,
            ActionType.MORTGAGE,
            ActionType.DECLARE_BANKRUPTCY,
        ),
        GAME_OVER: (),
    }

    def __init__(self, game_state: GameState) -> None:
        self.game = game_state
        self.rules = game_state.rules
        self.turn_phase = self.BEFORE_ROLL

        self.double_count = 0
        self._pending_extra_roll = False
        self._pending_tile_id: int | None = None
        self._resume_phase = self.AFTER_ROLL

        # Estado temporário do leilão.
        self._auction_tile_id: int | None = None
        self._auction_active: list[UUID] = []
        self._auction_pending: list[UUID] = []
        self._auction_highest_bid = 0
        self._auction_highest_bidder: UUID | None = None

    # ================================================================ consulta

    @property
    def current_decision_player(self):
        """Jogador que deve fornecer a próxima ação."""
        return self.game.players[self.current_decision_player_id]

    @property
    def current_decision_player_id(self) -> UUID:
        if self.turn_phase == self.AUCTION and self._auction_pending:
            return self._auction_pending[0]

        if self.turn_phase == self.RESOLVE_DEBT and self.game.debts:
            return self.game.debts[0].debtor_id

        return self.game.current_player_id

    @property
    def pending_tile_id(self) -> int | None:
        """Imóvel em decisão de compra ou em leilão."""
        if self.turn_phase == self.AUCTION:
            return self._auction_tile_id
        return self._pending_tile_id

    @property
    def auction_highest_bid(self) -> int:
        return self._auction_highest_bid

    @property
    def auction_highest_bidder(self) -> UUID | None:
        return self._auction_highest_bidder

    def auction_bid_amount(self, increment: int) -> int:
        """Valor de um lance `increment` acima do maior lance atual."""
        return max(self.rules.auction_min_bid, self._auction_highest_bid + increment)

    def unmortgage_cost(self, tile: PropertyTile) -> int:
        if tile.unmortgage_cost:
            return tile.unmortgage_cost
        return tile.mortgage_value + -(-tile.mortgage_value // 10)

    # ========================================================== ações legais

    def validate(self, action: GameAction) -> str | None:
        """Retorna None se a ação é legal para `current_decision_player`; senão o motivo."""
        if self.game.game_over:
            return "A partida já terminou."

        kind = action.action_type
        if kind not in self.PHASE_ACTIONS[self.turn_phase]:
            return f"A ação {kind} não é permitida na fase {self.turn_phase}."

        player_id = self.current_decision_player_id
        state = self.game.player_states[player_id]

        if kind == ActionType.PAY_JAIL_FINE:
            if not state.is_in_jail:
                return "O jogador não está preso."
            if state.money < self.rules.jail_fine:
                return "Saldo insuficiente para pagar a multa."

        elif kind == ActionType.USE_JAIL_CARD:
            if not state.is_in_jail or not state.jail_cards:
                return "O jogador não pode usar uma carta de prisão."

        elif kind == ActionType.BUY_PROPERTY:
            tile = self.game.board.get_tile_by_id(self._pending_tile_id)
            if state.money < tile.price:
                return "Saldo insuficiente para comprar."

        elif kind == ActionType.AUCTION_BID:
            amount = action.payload.get("amount")
            if (
                not isinstance(amount, int)
                or isinstance(amount, bool)
                or amount <= self._auction_highest_bid
                or amount < self.rules.auction_min_bid
            ):
                return "O lance deve ser um inteiro maior que o lance atual."
            if amount > state.money:
                return "Saldo insuficiente para esse lance."

        elif kind == ActionType.AUCTION_WITHDRAW:
            if player_id == self._auction_highest_bidder:
                return "O maior lance atual não pode ser retirado."

        elif kind == ActionType.DECLARE_BANKRUPTCY:
            amount_owed = self.game.amount_owed(player_id)
            recoverable = state.money + self.game.liquidation_value(player_id)
            if recoverable >= amount_owed:
                return "Ainda há recursos suficientes para pagar as dívidas."

        elif kind in MANAGEMENT_ACTIONS:
            return self._validate_management(player_id, kind, action.payload)

        return None

    def _validate_management(
        self, player_id: UUID, kind: str, payload: dict[str, Any]
    ) -> str | None:
        tile_id = payload.get("tile_id")
        if tile_id not in self.game.property_states:
            return "Informe o tile_id de um imóvel."

        tile = self.game.board.get_tile_by_id(tile_id)
        prop = self.game.property_states[tile_id]
        if prop.owner_id != player_id:
            return "O imóvel não pertence ao jogador."

        group = self.game.board.group_members(tile.group)
        levels = [self.game.property_states[t.tile_id].houses for t in group]
        money = self.game.player_states[player_id].money

        if kind == ActionType.BUILD_HOUSE:
            if tile.kind != PropertyKind.STREET:
                return "Só é possível construir em ruas."
            if not self.game.owns_group(player_id, tile.group):
                return "É preciso possuir o grupo completo."
            if any(self.game.property_states[t.tile_id].is_mortgaged for t in group):
                return "O grupo possui imóvel hipotecado."
            if prop.houses >= HOTEL_LEVEL:
                return "O imóvel já possui hotel."
            if prop.houses > min(levels):
                return "A construção deve ser uniforme no grupo."
            if money < tile.house_cost:
                return "Saldo insuficiente para construir."
            if prop.houses == HOTEL_LEVEL - 1 and self.game.hotels_in_bank < 1:
                return "Não há hotéis disponíveis no banco."
            if prop.houses < HOTEL_LEVEL - 1 and self.game.houses_in_bank < 1:
                return "Não há casas disponíveis no banco."

        elif kind == ActionType.SELL_HOUSE:
            if prop.houses == 0:
                return "Não há construções para vender."
            if prop.houses < max(levels):
                return "A venda deve ser uniforme no grupo."
            if prop.houses == HOTEL_LEVEL and self.game.houses_in_bank < HOTEL_LEVEL - 1:
                return "Não há casas no banco para desfazer o hotel."

        elif kind == ActionType.MORTGAGE:
            if prop.is_mortgaged:
                return "O imóvel já está hipotecado."
            if any(levels):
                return "Venda as construções do grupo antes de hipotecar."

        elif kind == ActionType.UNMORTGAGE:
            if not prop.is_mortgaged:
                return "O imóvel não está hipotecado."
            if money < self.unmortgage_cost(tile):
                return "Saldo insuficiente para levantar a hipoteca."

        return None

    def legal_actions(self) -> list[GameAction]:
        """Todas as ações legais (com payloads concretos) para `current_decision_player`."""
        candidates: list[GameAction] = []

        for kind in self.PHASE_ACTIONS[self.turn_phase]:
            if kind in MANAGEMENT_ACTIONS:
                for tile in self.game.owned_tiles(self.current_decision_player_id):
                    candidates.append(GameAction(kind, {"tile_id": tile.tile_id}))
            elif kind == ActionType.AUCTION_BID:
                amounts = sorted({self.auction_bid_amount(inc) for inc in AUCTION_BID_INCREMENTS})
                candidates.extend(GameAction(kind, {"amount": a}) for a in amounts)
            else:
                candidates.append(GameAction(kind))

        return [action for action in candidates if self.validate(action) is None]

    # ============================================================ processamento

    def process_action(self, action: GameAction) -> dict[str, Any]:
        reason = self.validate(action)
        if reason is not None:
            return self._rejected(reason)

        player_id = self.current_decision_player_id
        handler = getattr(self, f"_handle_{action.action_type.lower()}")
        result = handler(player_id, action)
        self._settle()

        result["phase"] = self.turn_phase
        return result

    def _handle_roll_dice(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        state = self.game.player_states[player_id]
        roll = self.game.dice.roll()
        self.game.last_roll = roll
        self._pending_extra_roll = False

        self.game.log_event(
            "DICE_ROLLED",
            {"player_id": player_id, "die1": roll.die1, "die2": roll.die2},
        )

        if state.is_in_jail:
            state.jail_turns += 1

            if roll.is_double:
                self._leave_jail(player_id, "JAIL_ESCAPE_DOUBLE")
                # A rolagem que tira da prisão movimenta a peça, sem rolagem extra.
                self._move_by(player_id, roll.total)
            elif state.jail_turns >= self.rules.max_jail_attempts:
                self._leave_jail(player_id, "JAIL_FORCED_PAY")
                self._charge(player_id, self.rules.jail_fine, None, "JAIL_FINE")
                self._move_by(player_id, roll.total)
            else:
                self.game.log_event(
                    "STAYED_IN_JAIL",
                    {"player_id": player_id, "attempt": state.jail_turns},
                )

            return self._finish_move()

        if roll.is_double:
            self.double_count += 1
            if self.double_count >= self.rules.max_consecutive_doubles:
                self._send_to_jail(player_id, "GO_TO_JAIL_DOUBLES")
                return self._finish_move()
            self._pending_extra_roll = True

        self._move_by(player_id, roll.total)
        return self._finish_move()

    def _handle_pay_jail_fine(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        self.game.player_states[player_id].money -= self.rules.jail_fine
        self._leave_jail(player_id, "JAIL_ESCAPE_FINE")
        return self._success(event="JAIL_ESCAPE_FINE")

    def _handle_use_jail_card(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        deck_name, card = self.game.player_states[player_id].jail_cards.pop()
        self.game.decks[deck_name].return_card(card)
        self._leave_jail(player_id, "JAIL_ESCAPE_CARD")
        return self._success(event="JAIL_ESCAPE_CARD")

    def _handle_buy_property(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(self._pending_tile_id)
        self.game.player_states[player_id].money -= tile.price
        self.game.property_states[tile.tile_id].owner_id = player_id
        self._pending_tile_id = None

        self.game.log_event(
            "PROPERTY_PURCHASED",
            {"player_id": player_id, "tile_id": tile.tile_id, "price": tile.price},
        )
        return self._finish_property_resolution()

    def _handle_decline_property(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile_id = self._pending_tile_id
        self._pending_tile_id = None
        self.game.log_event(
            "PROPERTY_PURCHASE_DECLINED",
            {"player_id": player_id, "tile_id": tile_id},
        )

        if self.rules.auction_after_declined_purchase():
            self._start_auction(tile_id)
            return self._success(event="AUCTION_STARTED")

        return self._finish_property_resolution()

    def _handle_end_turn(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        self._next_turn()
        return self._success(event="TURN_ENDED")

    def _handle_declare_bankruptcy(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        self._bankrupt(player_id, self._bankruptcy_creditor(player_id))
        return self._success(event="BANKRUPTCY")

    def _handle_build_house(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(action.payload["tile_id"])
        prop = self.game.property_states[tile.tile_id]
        self.game.player_states[player_id].money -= tile.house_cost

        if prop.houses == HOTEL_LEVEL - 1:
            self.game.hotels_in_bank -= 1
            self.game.houses_in_bank += HOTEL_LEVEL - 1
        else:
            self.game.houses_in_bank -= 1
        prop.houses += 1

        self.game.log_event(
            "BUILDING_BOUGHT",
            {"player_id": player_id, "tile_id": tile.tile_id, "level": prop.houses},
        )
        return self._success(event="BUILDING_BOUGHT")

    def _handle_sell_house(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(action.payload["tile_id"])
        prop = self.game.property_states[tile.tile_id]

        if prop.houses == HOTEL_LEVEL:
            self.game.hotels_in_bank += 1
            self.game.houses_in_bank -= HOTEL_LEVEL - 1
        else:
            self.game.houses_in_bank += 1
        prop.houses -= 1
        self.game.player_states[player_id].money += tile.house_cost // 2

        self.game.log_event(
            "BUILDING_SOLD",
            {"player_id": player_id, "tile_id": tile.tile_id, "level": prop.houses},
        )
        return self._success(event="BUILDING_SOLD")

    def _handle_mortgage(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(action.payload["tile_id"])
        self.game.property_states[tile.tile_id].is_mortgaged = True
        self.game.player_states[player_id].money += tile.mortgage_value
        self.game.log_event("MORTGAGED", {"player_id": player_id, "tile_id": tile.tile_id})
        return self._success(event="MORTGAGED")

    def _handle_unmortgage(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(action.payload["tile_id"])
        self.game.player_states[player_id].money -= self.unmortgage_cost(tile)
        self.game.property_states[tile.tile_id].is_mortgaged = False
        self.game.log_event("UNMORTGAGED", {"player_id": player_id, "tile_id": tile.tile_id})
        return self._success(event="UNMORTGAGED")

    # ================================================================== leilão

    def _start_auction(self, tile_id: int) -> None:
        self._auction_tile_id = tile_id
        # Começa pelo jogador da vez e segue a ordem da mesa
        order = self.game.active_player_ids
        start = order.index(self.game.current_player_id)
        self._auction_active = order[start:] + order[:start]
        self._auction_pending = self._auction_active.copy()
        self._auction_highest_bid = 0
        self._auction_highest_bidder = None
        self.turn_phase = self.AUCTION

    def _handle_auction_bid(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        amount = action.payload["amount"]
        self._auction_highest_bid = amount
        self._auction_highest_bidder = player_id
        self.game.log_event(
            "AUCTION_BID",
            {"player_id": player_id, "tile_id": self._auction_tile_id, "amount": amount},
        )

        # Todos os outros ainda ativos voltam a decidir, na ordem da mesa
        idx = self._auction_active.index(player_id)
        rotated = self._auction_active[idx + 1:] + self._auction_active[:idx]
        self._auction_pending = rotated
        return self._advance_auction()

    def _handle_auction_withdraw(self, player_id: UUID, action: GameAction) -> dict[str, Any]:
        self._auction_pending.remove(player_id)
        self._auction_active.remove(player_id)
        self.game.log_event(
            "AUCTION_WITHDRAW",
            {"player_id": player_id, "tile_id": self._auction_tile_id},
        )
        return self._advance_auction()

    def _advance_auction(self) -> dict[str, Any]:
        if self._auction_pending:
            return self._success(
                event="AUCTION_CONTINUES",
                next_bidder_id=self._auction_pending[0],
                highest_bid=self._auction_highest_bid,
            )
        return self._finish_auction()

    def _finish_auction(self) -> dict[str, Any]:
        tile_id = self._auction_tile_id
        winner_id = self._auction_highest_bidder
        amount = self._auction_highest_bid

        if winner_id is not None:
            # O lance nunca excede o saldo, e o saldo não muda durante o leilão
            self.game.player_states[winner_id].money -= amount
            self.game.property_states[tile_id].owner_id = winner_id
            self.game.log_event(
                "AUCTION_WON",
                {"player_id": winner_id, "tile_id": tile_id, "amount": amount},
            )
        else:
            self.game.log_event("AUCTION_NO_BIDS", {"tile_id": tile_id})

        self._auction_tile_id = None
        self._auction_active = []
        self._auction_pending = []
        self._auction_highest_bid = 0
        self._auction_highest_bidder = None
        return self._finish_property_resolution()

    # =============================================================== movimento

    def _move_by(self, player_id: UUID, steps: int) -> None:
        if self.game.player_states[player_id].is_bankrupt:
            return

        board_size = self.game.board.size
        unwrapped = self.game.position_of(player_id) + steps
        if unwrapped >= board_size:
            self._collect_salary(player_id)

        self.game.set_position(player_id, unwrapped % board_size)
        self._land(player_id)

    def _move_to(
        self,
        player_id: UUID,
        position: int,
        collect_salary: bool = True,
        rent_multiplier: int | None = None,
        reroll_utility: bool = False,
    ) -> None:
        if self.game.player_states[player_id].is_bankrupt:
            return

        if collect_salary and position < self.game.position_of(player_id):
            self._collect_salary(player_id)

        self.game.set_position(player_id, position)
        self._land(player_id, rent_multiplier, reroll_utility)

    def _collect_salary(self, player_id: UUID) -> None:
        self.game.player_states[player_id].money += self.rules.go_salary
        self.game.log_event(
            "PASSED_GO",
            {"player_id": player_id, "salary": self.rules.go_salary},
        )

    def _land(
        self,
        player_id: UUID,
        rent_multiplier: int | None = None,
        reroll_utility: bool = False,
    ) -> None:
        if self.game.position_of(player_id) == self.rules.go_to_jail_position:
            self._send_to_jail(player_id, "GO_TO_JAIL_TILE")
            return

        tile = self.game.board.get_tile(self.game.position_of(player_id))

        if isinstance(tile, PropertyTile):
            owner_id = self.game.property_states[tile.tile_id].owner_id

            if owner_id is None:
                self._pending_tile_id = tile.tile_id
            elif owner_id != player_id:
                if tile.kind == PropertyKind.UTILITY and reroll_utility:
                    # Carta: "jogue os dados e pague N vezes o valor"
                    roll = self.game.dice.roll()
                    rent = (rent_multiplier or 1) * roll.total
                else:
                    rent = self.game.rent_for(tile) * (rent_multiplier or 1)
                self._charge(player_id, rent, owner_id, "RENT_PAYMENT", tile_id=tile.tile_id)

        elif isinstance(tile, TaxTile):
            amount = tile.tax_amount
            if tile.percent_option is not None and self.rules.income_tax_choice:
                percent = self.game.net_worth(player_id) * tile.percent_option // 100
                amount = min(amount, percent)
            self._charge(player_id, amount, None, "TAX_PAYMENT", tile_id=tile.tile_id)

        elif isinstance(tile, SpecialTile):
            if tile.tile_type in self.game.decks:
                self._draw_card(player_id, tile.tile_type)

    def _draw_card(self, player_id: UUID, deck_name: str) -> None:
        card = self.game.decks[deck_name].draw()
        self.game.log_event(
            "CARD_DRAWN",
            {"player_id": player_id, "deck": deck_name, "card_id": card.id},
        )
        CARD_ACTIONS.execute(card, CardContext(self, player_id, deck_name))

        if card.action != CardAction.GET_OUT_OF_JAIL:
            self.game.decks[deck_name].return_card(card)

    def _send_to_jail(self, player_id: UUID, event: str) -> None:
        state = self.game.player_states[player_id]
        self.game.set_position(player_id, self.rules.jail_position)
        state.is_in_jail = True
        state.jail_turns = 0
        self.double_count = 0
        self._pending_extra_roll = False
        self.game.log_event(event, {"player_id": player_id})

    def _leave_jail(self, player_id: UUID, event: str) -> None:
        state = self.game.player_states[player_id]
        state.is_in_jail = False
        state.jail_turns = 0
        self.game.log_event(event, {"player_id": player_id})

    def _finish_move(self) -> dict[str, Any]:
        if self.game.game_over:
            return self._success()
        if self._pending_tile_id is not None:
            self.turn_phase = self.AWAITING_PROPERTY_DECISION
            return self._success(event="AWAITING_PROPERTY_DECISION", tile_id=self._pending_tile_id)
        return self._finish_property_resolution()

    def _finish_property_resolution(self) -> dict[str, Any]:
        state = self.game.player_states[self.game.current_player_id]

        if self._pending_extra_roll and not state.is_in_jail and not state.is_bankrupt:
            self.turn_phase = self.BEFORE_ROLL
        else:
            self._pending_extra_roll = False
            self.turn_phase = self.AFTER_ROLL

        return self._success()

    def _next_turn(self) -> None:
        self.double_count = 0
        self._pending_extra_roll = False
        self._pending_tile_id = None

        max_turns = self.rules.max_turns
        if max_turns is not None and self.game.turn_counter >= max_turns:
            self.game.game_over = True
            self.game.truncated = True
            self.turn_phase = self.GAME_OVER
            self.game.log_event("GAME_TRUNCATED", {"turn": self.game.turn_counter})
            return

        order = self.game.turn_order
        idx = order.index(self.game.current_player_id)
        for step in range(1, len(order) + 1):
            candidate = order[(idx + step) % len(order)]
            if not self.game.player_states[candidate].is_bankrupt:
                break

        self.game.current_player_id = candidate
        self.game.turn_counter += 1
        self.turn_phase = self.BEFORE_ROLL

    # ======================================================= dívida e falência

    def _charge(
        self,
        player_id: UUID,
        amount: int,
        creditor_id: UUID | None,
        event: str,
        **details: Any,
    ) -> None:
        """Cobra `amount`; se faltar dinheiro, abre uma dívida a ser resolvida."""
        state = self.game.player_states[player_id]
        if amount <= 0 or self.game.game_over or state.is_bankrupt:
            return

        self.game.log_event(
            event,
            {"player_id": player_id, "amount": amount, "creditor_id": creditor_id, **details},
        )

        if self.game.amount_owed(player_id) == 0 and state.money >= amount:
            self._transfer(player_id, creditor_id, amount)
            return

        self.game.debts.append(Debt(player_id, creditor_id, amount))
        owed = self.game.amount_owed(player_id)
        if state.money + self.game.liquidation_value(player_id) < owed:
            # Nem vendendo tudo o jogador consegue pagar
            self._bankrupt(player_id, self._bankruptcy_creditor(player_id))

    def _transfer(self, debtor_id: UUID, creditor_id: UUID | None, amount: int) -> None:
        self.game.player_states[debtor_id].money -= amount
        if creditor_id is not None and not self.game.player_states[creditor_id].is_bankrupt:
            self.game.player_states[creditor_id].money += amount

    def _bankruptcy_creditor(self, player_id: UUID) -> UUID | None:
        """Herda os bens o jogador credor, se for o único; senão, o banco."""
        creditors = {d.creditor_id for d in self.game.debts if d.debtor_id == player_id}
        return creditors.pop() if len(creditors) == 1 else None

    def _bankrupt(self, player_id: UUID, creditor_id: UUID | None) -> None:
        game = self.game
        state = game.player_states[player_id]
        game.log_event("BANKRUPTCY", {"player_id": player_id, "creditor_id": creditor_id})
        game.debts = [d for d in game.debts if d.debtor_id != player_id]

        if player_id == game.current_player_id:
            self._pending_tile_id = None

        for tile in game.owned_tiles(player_id):
            prop = game.property_states[tile.tile_id]
            if prop.houses:
                # Construções voltam ao banco pela metade do custo
                state.money += prop.houses * (tile.house_cost // 2)
                if prop.houses == HOTEL_LEVEL:
                    game.hotels_in_bank += 1
                else:
                    game.houses_in_bank += prop.houses
                prop.houses = 0

            if creditor_id is None:
                prop.owner_id = None
                prop.is_mortgaged = False
            else:
                prop.owner_id = creditor_id

        if creditor_id is None:
            for deck_name, card in state.jail_cards:
                game.decks[deck_name].return_card(card)
        else:
            game.player_states[creditor_id].money += state.money
            game.player_states[creditor_id].jail_cards.extend(state.jail_cards)

        state.jail_cards = []
        state.money = 0
        state.is_in_jail = False
        state.is_bankrupt = True

        if player_id in self._auction_active:
            self._auction_active.remove(player_id)
        if player_id in self._auction_pending:
            self._auction_pending.remove(player_id)

        active = game.active_player_ids
        if self.rules.is_game_over(len(active)):
            game.game_over = True
            game.winner_id = active[0] if len(active) == 1 else None
            game.debts = []
            self.turn_phase = self.GAME_OVER
            game.log_event("GAME_FINISHED", {"winner_id": game.winner_id})

    def _settle(self) -> None:
        """Quita as dívidas que já podem ser pagas e ajusta a fase."""
        game = self.game
        if game.game_over:
            self.turn_phase = self.GAME_OVER
            return

        while game.debts:
            debt = game.debts[0]
            if game.player_states[debt.debtor_id].money < debt.amount:
                break
            self._transfer(debt.debtor_id, debt.creditor_id, debt.amount)
            game.debts.pop(0)

        if game.debts:
            if self.turn_phase != self.RESOLVE_DEBT:
                self._resume_phase = self.turn_phase
                self.turn_phase = self.RESOLVE_DEBT
            return

        if self.turn_phase == self.RESOLVE_DEBT:
            self.turn_phase = self._resume_phase

        if game.player_states[game.current_player_id].is_bankrupt:
            self._next_turn()

    # ============================================================== utilidades

    def clone(self) -> "GameLoop":
        """Cópia independente da partida (estado, dados, baralhos, RNGs).

        Tabuleiro e regras são imutáveis e ficam compartilhados. Útil para MCTS e busca.
        """
        memo = {id(self.game.board): self.game.board, id(self.rules): self.rules}
        return copy.deepcopy(self, memo)

    @staticmethod
    def _success(event: str | None = None, **details: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "SUCCESS"}
        if event is not None:
            result["event"] = event
        result.update(details)
        return result

    @staticmethod
    def _rejected(reason: str) -> dict[str, Any]:
        return {"status": "REJECTED", "reason": reason}


# ============================================================ efeitos de cartas


@dataclass(frozen=True, slots=True)
class CardContext:
    loop: GameLoop
    player_id: UUID
    deck_name: str


def _ok(card: CardLike, message: str = "") -> CardExecutionResult:
    return CardExecutionResult(card.id, card.action, True, message)


def _advance_to(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    ctx.loop._move_to(
        ctx.player_id,
        card.data["position"],
        collect_salary=card.data.get("collectGoSalary", True),
    )
    return _ok(card)


def _advance_to_nearest(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    loop = ctx.loop
    board = loop.game.board
    kind = card.data["type"]
    position = loop.game.position_of(ctx.player_id)
    target = next(
        (position + step) % board.size
        for step in range(1, board.size + 1)
        if isinstance(tile := board.get_tile(position + step), PropertyTile)
        and tile.kind == kind
    )
    loop._move_to(
        ctx.player_id,
        target,
        collect_salary=card.data.get("collectGoSalary", True),
        rent_multiplier=card.data.get("rentMultiplier"),
        reroll_utility=kind == PropertyKind.UTILITY,
    )
    return _ok(card)


def _collect(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    ctx.loop.game.player_states[ctx.player_id].money += card.data["amount"]
    return _ok(card)


def _pay(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    ctx.loop._charge(ctx.player_id, card.data["amount"], None, "CARD_PAYMENT", card_id=card.id)
    return _ok(card)


def _pay_each_player(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    for other in ctx.loop.game.active_player_ids:
        if other != ctx.player_id:
            ctx.loop._charge(ctx.player_id, card.data["amount"], other, "CARD_PAYMENT", card_id=card.id)
    return _ok(card)


def _collect_from_each_player(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    for other in ctx.loop.game.active_player_ids:
        if other != ctx.player_id:
            ctx.loop._charge(other, card.data["amount"], ctx.player_id, "CARD_PAYMENT", card_id=card.id)
    return _ok(card)


def _go_to_jail(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    ctx.loop._send_to_jail(ctx.player_id, "GO_TO_JAIL_CARD")
    return _ok(card)


def _get_out_of_jail(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    ctx.loop.game.player_states[ctx.player_id].jail_cards.append((ctx.deck_name, card))
    return _ok(card)


def _go_back(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    game = ctx.loop.game
    position = (game.position_of(ctx.player_id) - card.data["spaces"]) % game.board.size
    ctx.loop._move_to(ctx.player_id, position, collect_salary=False)
    return _ok(card)


def _repairs(card: CardLike, ctx: CardContext) -> CardExecutionResult:
    game = ctx.loop.game
    cost = 0
    for tile in game.owned_tiles(ctx.player_id):
        level = game.property_states[tile.tile_id].houses
        cost += card.data["hotelCost"] if level == HOTEL_LEVEL else level * card.data["houseCost"]
    ctx.loop._charge(ctx.player_id, cost, None, "CARD_PAYMENT", card_id=card.id)
    return _ok(card)


CARD_ACTIONS = CardActionRegistry()
for _action, _handler in (
    (CardAction.ADVANCE_TO, _advance_to),
    (CardAction.ADVANCE_TO_NEAREST, _advance_to_nearest),
    (CardAction.COLLECT, _collect),
    (CardAction.BANK_ERROR, _collect),
    (CardAction.PAY, _pay),
    (CardAction.PAY_EACH_PLAYER, _pay_each_player),
    (CardAction.COLLECT_FROM_EACH_PLAYER, _collect_from_each_player),
    (CardAction.BIRTHDAY, _collect_from_each_player),
    (CardAction.GO_TO_JAIL, _go_to_jail),
    (CardAction.GET_OUT_OF_JAIL, _get_out_of_jail),
    (CardAction.GO_BACK, _go_back),
    (CardAction.REPAIRS, _repairs),
):
    CARD_ACTIONS.register(_action, _handler)
