
from typing import Any

from .actions import GameAction
from .board import PropertyTile, SpecialTile, TaxTile
from .game_state import GameState


class GameLoop:
    """Coordena ações, fases do turno e aplicação das regras."""

    BEFORE_ROLL = "BEFORE_ROLL"
    AWAITING_PROPERTY_DECISION = "AWAITING_PROPERTY_DECISION"
    AUCTION = "AUCTION"
    AFTER_ROLL = "AFTER_ROLL"

    def __init__(self, game_state: GameState) -> None:
        self.game = game_state
        self.rules = game_state.rules
        self.turn_phase = self.BEFORE_ROLL

        self._pending_extra_roll = False

        # Estado temporário do leilão.
        self._auction_tile_id: int | None = None
        self._auction_active: list[int] = []
        self._auction_pending: list[int] = []
        self._auction_highest_bid = 0
        self._auction_highest_bidder: int | None = None

    @property
    def current_decision_player(self):
        """Jogador que deve fornecer a próxima ação."""
        if self.turn_phase == self.AUCTION and self._auction_pending:
            player_id = self._auction_pending[0]
            return self._player_by_id(player_id)

        return self.game.current_player

    def process_action(self, action: GameAction) -> dict[str, Any]:
        if self.game.game_over:
            return self._rejected("A partida já terminou.")

        if self.turn_phase == self.AUCTION:
            return self._process_auction_action(action)

        player = self.game.current_player

        if player.is_bankrupt:
            return self._rejected("O jogador atual está falido.")

        handlers = {
            "ROLL_DICE": self._handle_roll,
            "USE_JAIL_CARD": self._handle_jail_card,
            "BUY_PROPERTY": self._handle_buy_property,
            "DECLINE_PROPERTY": self._handle_decline_property,
            "END_TURN": self._handle_end_turn,
        }

        handler = handlers.get(action.action_type)

        if handler is None:
            return self._rejected(
                f"Ação desconhecida: {action.action_type}"
            )

        return handler(player, action)

    def _handle_roll(self, player, action: GameAction) -> dict[str, Any]:
        if self.turn_phase != self.BEFORE_ROLL:
            return self._rejected("Os dados não podem ser rolados nesta fase.")

        if player.is_in_jail:
            return self._execute_jail_roll(player)

        return self._execute_standard_movement(player)

    def _handle_jail_card(
        self, player, action: GameAction
    ) -> dict[str, Any]:
        if not player.is_in_jail or player.has_jail_card <= 0:
            return self._rejected("O jogador não pode usar uma carta de prisão.")

        player.has_jail_card -= 1
        player.is_in_jail = False
        player.jail_turns = 0

        self.game.log_event(
            "JAIL_ESCAPE_CARD",
            {"player_id": player.player_id},
        )

        return self._success(event="JAIL_ESCAPE_CARD")

    def _handle_buy_property(
        self, player, action: GameAction
    ) -> dict[str, Any]:
        if self.turn_phase != self.AWAITING_PROPERTY_DECISION:
            return self._rejected("Não há uma compra pendente.")

        tile = self.game.board.get_tile(player.position)

        if not isinstance(tile, PropertyTile):
            return self._rejected("A casa atual não é uma propriedade.")

        if tile.owner_id is not None:
            return self._rejected("A propriedade já possui proprietário.")

        if player.money < tile.price:
            return self._rejected("Saldo insuficiente para comprar.")

        player.buy_property(tile.tile_id, tile.price)
        tile.owner_id = player.player_id

        self.game.log_event(
            "PROPERTY_PURCHASED",
            {
                "player_id": player.player_id,
                "tile_id": tile.tile_id,
                "price": tile.price,
            },
        )

        return self._finish_property_resolution()

    def _handle_decline_property(
        self, player, action: GameAction
    ) -> dict[str, Any]:
        if self.turn_phase != self.AWAITING_PROPERTY_DECISION:
            return self._rejected("Não há uma decisão de compra pendente.")

        tile = self.game.board.get_tile(player.position)

        self.game.log_event(
            "PROPERTY_PURCHASE_DECLINED",
            {
                "player_id": player.player_id,
                "tile_id": tile.tile_id,
            },
        )

        if self.rules.auction_after_declined_purchase():
            self._start_auction(tile)
            return self._success(event="AUCTION_STARTED")

        return self._finish_property_resolution()

    def _handle_end_turn(
        self, player, action: GameAction
    ) -> dict[str, Any]:
        if self.turn_phase not in (self.BEFORE_ROLL, self.AFTER_ROLL):
            return self._rejected(
                "Não é possível encerrar o turno nesta fase."
            )

        self._pending_extra_roll = False
        self.game.next_turn()
        self.turn_phase = self.BEFORE_ROLL

        return self._success(event="TURN_ENDED")

    def _execute_standard_movement(self, player) -> dict[str, Any]:
        self.game.dice.roll()

        dice_total = self.game.dice.total
        is_double = self.game.dice.is_double()

        if (
            is_double
            and self.game.dice.double_count
            >= self.rules.max_consecutive_doubles
        ):
            player.position = 10
            player.is_in_jail = True
            player.jail_turns = 0
            self._pending_extra_roll = False

            self.game.log_event(
                "GO_TO_JAIL_DOUBLES",
                {"player_id": player.player_id},
            )

            self.game.next_turn()
            self.turn_phase = self.BEFORE_ROLL

            return self._success(event="SENT_TO_JAIL_DOUBLES")

        self._pending_extra_roll = is_double

        return self._move_and_resolve(player, dice_total, is_double)

    def _move_and_resolve(
        self, player, dice_total: int, is_double: bool
    ) -> dict[str, Any]:
        old_position = player.position
        board_size = self.game.board.size
        unwrapped_position = old_position + dice_total
        new_position = unwrapped_position % board_size

        player.position = new_position

        if unwrapped_position >= board_size:
            player.receive_money(self.rules.go_salary)

            self.game.log_event(
                "PASSED_GO",
                {
                    "player_id": player.player_id,
                    "salary": self.rules.go_salary,
                },
            )

        tile = self.game.board.get_tile(new_position)
        return self._evaluate_tile_arrival(player, tile, is_double)

    def _evaluate_tile_arrival(
        self, player, tile, is_double: bool
    ) -> dict[str, Any]:
        if isinstance(tile, PropertyTile):
            if tile.owner_id is None:
                if player.money >= tile.price:
                    self.turn_phase = self.AWAITING_PROPERTY_DECISION

                    return self._success(
                        event="AWAITING_PROPERTY_DECISION",
                        tile_id=tile.tile_id,
                    )

                # Sem saldo para comprar: aplica a regra de leilão.
                if self.rules.auction_after_declined_purchase():
                    self._start_auction(tile)
                    return self._success(event="AUCTION_STARTED")

            elif (
                tile.owner_id != player.player_id
                and not tile.is_mortgaged
            ):
                rent = tile.current_rent
                result = self._charge_player(
                    player,
                    rent,
                    creditor_id=tile.owner_id,
                    event="RENT_PAYMENT",
                    tile_id=tile.tile_id,
                )

                if result is not None:
                    return result

        elif isinstance(tile, TaxTile):
            result = self._charge_player(
                player,
                tile.tax_amount,
                creditor_id=None,
                event="TAX_PAYMENT",
                tile_id=tile.tile_id,
            )

            if result is not None:
                return result

        elif isinstance(tile, SpecialTile):
            if tile.tile_type in ("GO_TO_JAIL", "GO TO JAIL"):
                player.position = 10
                player.is_in_jail = True
                player.jail_turns = 0
                self._pending_extra_roll = False

                self.game.log_event(
                    "GO_TO_JAIL_TILE",
                    {"player_id": player.player_id},
                )

                self.game.next_turn()
                self.turn_phase = self.BEFORE_ROLL

                return self._success(event="SENT_TO_JAIL")

        return self._set_post_roll_phase(is_double)

    def _execute_jail_roll(self, player) -> dict[str, Any]:
        player.jail_turns += 1
        self.game.dice.roll()

        dice_total = self.game.dice.total
        is_double = self.game.dice.is_double()

        if is_double:
            player.is_in_jail = False
            player.jail_turns = 0

            self.game.log_event(
                "JAIL_ESCAPE_DOUBLE",
                {"player_id": player.player_id},
            )

            # A rolagem usada para sair da prisão também movimenta a peça.
            # Não concede uma rolagem extra neste turno.
            self._pending_extra_roll = False
            return self._move_and_resolve(
                player,
                dice_total,
                is_double=False,
            )

        if player.jail_turns >= self.rules.max_jail_attempts:
            if player.money >= self.rules.jail_fine:
                player.pay_money(self.rules.jail_fine)
                player.is_in_jail = False
                player.jail_turns = 0

                self.game.log_event(
                    "JAIL_FORCED_PAY",
                    {
                        "player_id": player.player_id,
                        "amount": self.rules.jail_fine,
                    },
                )

                self._pending_extra_roll = False
                return self._move_and_resolve(
                    player,
                    dice_total,
                    is_double=False,
                )

            self.game.force_bankruptcy(player, creditor_id=None)
            self._pending_extra_roll = False
            self.turn_phase = self.BEFORE_ROLL

            return self._success(event="BANKRUPTCY")

        self.game.log_event(
            "STAYED_IN_JAIL",
            {
                "player_id": player.player_id,
                "attempt": player.jail_turns,
            },
        )

        self._pending_extra_roll = False
        self.game.next_turn()
        self.turn_phase = self.BEFORE_ROLL

        return self._success(event="STAYED_IN_JAIL")

    def _charge_player(
        self,
        player,
        amount: int,
        creditor_id: int | None,
        event: str,
        tile_id: int,
    ) -> dict[str, Any] | None:
        if player.money < amount:
            # Provisório: futuramente, tentar venda/negociação antes
            # de aplicar a falência.
            self.game.force_bankruptcy(player, creditor_id=creditor_id)
            self._pending_extra_roll = False
            self.turn_phase = self.BEFORE_ROLL

            return self._success(event="BANKRUPTCY")

        player.pay_money(amount)

        if creditor_id is not None:
            self.game.players[creditor_id].receive_money(amount)

        self.game.log_event(
            event,
            {
                "player_id": player.player_id,
                "tile_id": tile_id,
                "amount": amount,
                "creditor_id": creditor_id,
            },
        )

        return None

    def _start_auction(self, tile) -> None:
        self._auction_tile_id = tile.tile_id
        self._auction_active = [
            player.player_id
            for player in self.game.players
            if not player.is_bankrupt
        ]
        self._auction_pending = self._auction_active.copy()
        self._auction_highest_bid = 0
        self._auction_highest_bidder = None
        self.turn_phase = self.AUCTION

    def _process_auction_action(
        self, action: GameAction
    ) -> dict[str, Any]:
        if not self._auction_pending:
            return self._rejected("Não há um participante para agir no leilão.")

        bidder_id = self._auction_pending[0]
        bidder = self._player_by_id(bidder_id)

        if bidder.is_bankrupt:
            self._auction_pending.pop(0)
            self._auction_active.remove(bidder_id)
            return self._advance_auction()

        if action.action_type == "AUCTION_BID":
            amount = action.payload.get("amount")

            if (
                not isinstance(amount, int)
                or isinstance(amount, bool)
                or amount <= self._auction_highest_bid
            ):
                return self._rejected(
                    "O lance deve ser um inteiro maior que o lance atual."
                )

            if amount > bidder.money:
                return self._rejected("Saldo insuficiente para esse lance.")

            self._auction_highest_bid = amount
            self._auction_highest_bidder = bidder_id

            self.game.log_event(
                "AUCTION_BID",
                {
                    "player_id": bidder_id,
                    "tile_id": self._auction_tile_id,
                    "amount": amount,
                },
            )

            self._auction_pending = [
                player_id
                for player_id in self._auction_active
                if player_id != bidder_id
            ]

            return self._advance_auction()

        if action.action_type == "AUCTION_WITHDRAW":
            # O maior lance atual não pode ser retirado depois de ofertado.
            if bidder_id == self._auction_highest_bidder:
                return self._rejected(
                    "O maior lance atual não pode ser retirado."
                )

            self._auction_pending.pop(0)
            self._auction_active.remove(bidder_id)

            self.game.log_event(
                "AUCTION_WITHDRAW",
                {
                    "player_id": bidder_id,
                    "tile_id": self._auction_tile_id,
                },
            )

            return self._advance_auction()

        return self._rejected(
            "Durante o leilão, use AUCTION_BID ou AUCTION_WITHDRAW."
        )

    def _advance_auction(self) -> dict[str, Any]:
        if self._auction_highest_bidder is not None:
            if len(self._auction_active) <= 1:
                return self._finish_auction()

        if not self._auction_pending:
            if self._auction_highest_bidder is not None:
                return self._finish_auction()

            # Todos passaram sem apresentar um lance.
            return self._finish_auction()

        return self._success(
            event="AUCTION_CONTINUES",
            next_bidder_id=self._auction_pending[0],
            highest_bid=self._auction_highest_bid,
        )

    def _finish_auction(self) -> dict[str, Any]:
        tile = self.game.board.get_tile_by_id(self._auction_tile_id)

        winner_id = self._auction_highest_bidder
        amount = self._auction_highest_bid

        if winner_id is not None:
            winner = self._player_by_id(winner_id)

            if winner.money < amount:
                return self._rejected(
                    "O vencedor não possui saldo para pagar o lance."
                )

            winner.pay_money(amount)
            winner.properties.append(tile.tile_id)
            tile.owner_id = winner_id

            self.game.log_event(
                "AUCTION_WON",
                {
                    "player_id": winner_id,
                    "tile_id": tile.tile_id,
                    "amount": amount,
                },
            )
        else:
            self.game.log_event(
                "AUCTION_NO_BIDS",
                {"tile_id": tile.tile_id},
            )

        self._clear_auction()
        return self._finish_property_resolution()

    def _clear_auction(self) -> None:
        self._auction_tile_id = None
        self._auction_active = []
        self._auction_pending = []
        self._auction_highest_bid = 0
        self._auction_highest_bidder = None

    def _finish_property_resolution(self) -> dict[str, Any]:
        return self._set_post_roll_phase(self._pending_extra_roll)

    def _set_post_roll_phase(self, is_double: bool) -> dict[str, Any]:
        player = self.game.current_player

        if (
            is_double
            and not player.is_in_jail
            and not player.is_bankrupt
        ):
            self.turn_phase = self.BEFORE_ROLL
        else:
            self._pending_extra_roll = False
            self.turn_phase = self.AFTER_ROLL

        return self._success()

    def _player_by_id(self, player_id: int):
        for player in self.game.players:
            if player.player_id == player_id:
                return player

        raise ValueError(f"Jogador não encontrado: {player_id}")

    @staticmethod
    def _success(
        event: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "SUCCESS",
        }

        if event is not None:
            result["event"] = event

        result.update(details)
        return result

    @staticmethod
    def _rejected(reason: str) -> dict[str, Any]:
        return {
            "status": "REJECTED",
            "reason": reason,
        }
