
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from .board import Board, PropertyKind, PropertyTile
from .card_actions import CardLike
from .cards import CardDeck
from .dice import DiceMechanism, DiceRoll
from .pieces import Piece
from .player import Player
from .rules import Rules

HOTEL_LEVEL = 5


@dataclass
class PlayerState:
    """Estado mutável de um participante na partida."""

    money: int
    is_in_jail: bool = False
    jail_turns: int = 0
    is_bankrupt: bool = False
    # Cartas de saída da prisão guardadas: (baralho de origem, carta)
    jail_cards: list[tuple[str, CardLike]] = field(default_factory=list)


@dataclass
class PieceState:
    """Estado mutável de uma peça."""

    position: int = 0


@dataclass
class PropertyState:
    """Estado mutável de uma propriedade do tabuleiro."""

    owner_id: UUID | None = None
    houses: int = 0  # 0 a 4 = casas, 5 = hotel
    is_mortgaged: bool = False


@dataclass(frozen=True, slots=True)
class Debt:
    """Valor que um jogador deve e ainda não conseguiu pagar."""

    debtor_id: UUID
    creditor_id: UUID | None  # None = banco
    amount: int


class GameState:
    """Representa a situação atual de uma partida."""

    def __init__(
        self,
        players: list[Player],
        pieces: list[Piece],
        board: Board,
        rules: Rules,
        dice: DiceMechanism,
        decks: dict[str, CardDeck],
        record_events: bool = True,
    ) -> None:
        if not players:
            raise ValueError("A partida deve possuir ao menos um jogador.")

        if len({player.player_id for player in players}) != len(players):
            raise ValueError("Os jogadores devem possuir IDs únicos.")

        if len({piece.piece_id for piece in pieces}) != len(pieces):
            raise ValueError("As peças devem possuir IDs únicos.")

        self.players: dict[UUID, Player] = {
            player.player_id: player for player in players
        }
        # Ordem de jogo; o índice nesta lista é o "assento" do jogador
        self.turn_order: list[UUID] = [player.player_id for player in players]

        self.player_states: dict[UUID, PlayerState] = {
            player.player_id: PlayerState(money=rules.starting_money)
            for player in players
        }

        self.pieces: dict[UUID, Piece] = {
            piece.piece_id: piece for piece in pieces
        }

        self.piece_states: dict[UUID, PieceState] = {
            piece.piece_id: PieceState(position=piece.position)
            for piece in pieces
        }

        self._piece_of_player: dict[UUID, UUID] = {}
        for piece in pieces:
            for player_id in piece.player_ids:
                self._piece_of_player.setdefault(player_id, piece.piece_id)

        missing = set(self.players) - set(self._piece_of_player)
        if missing:
            raise ValueError(f"Jogadores sem peça associada: {missing}")

        self.board = board
        self.rules = rules
        self.dice = dice
        self.decks = decks

        self.property_states: dict[int, PropertyState] = {
            tile.tile_id: PropertyState()
            for tile in board.property_tiles
        }

        self.houses_in_bank = rules.max_houses_in_bank
        self.hotels_in_bank = rules.max_hotels_in_bank

        self.current_player_id: UUID = players[0].player_id
        self.turn_counter: int = 1
        self.last_roll: DiceRoll | None = None
        self.debts: list[Debt] = []

        self.game_over: bool = False
        self.winner_id: UUID | None = None
        self.truncated: bool = False

        self.record_events = record_events
        self.history_logs: list[dict[str, Any]] = []

    # ------------------------------------------------------------- jogadores

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_id]

    @property
    def active_player_ids(self) -> list[UUID]:
        return [
            player_id
            for player_id in self.turn_order
            if not self.player_states[player_id].is_bankrupt
        ]

    def seat_of(self, player_id: UUID) -> int:
        return self.turn_order.index(player_id)

    def get_player_state(self, player_id: UUID) -> PlayerState:
        try:
            return self.player_states[player_id]
        except KeyError as exc:
            raise ValueError(
                f"Jogador não encontrado: {player_id}"
            ) from exc

    def get_piece_state(self, piece_id: UUID) -> PieceState:
        try:
            return self.piece_states[piece_id]
        except KeyError as exc:
            raise ValueError(
                f"Peça não encontrada: {piece_id}"
            ) from exc

    def get_property_state(self, tile_id: int) -> PropertyState:
        try:
            return self.property_states[tile_id]
        except KeyError as exc:
            raise ValueError(
                f"Propriedade não encontrada: {tile_id}"
            ) from exc

    def position_of(self, player_id: UUID) -> int:
        return self.piece_states[self._piece_of_player[player_id]].position

    def set_position(self, player_id: UUID, position: int) -> None:
        self.piece_states[self._piece_of_player[player_id]].position = position

    # ----------------------------------------------------------- propriedades

    def owned_tiles(self, player_id: UUID) -> list[PropertyTile]:
        return [
            tile
            for tile in self.board.property_tiles
            if self.property_states[tile.tile_id].owner_id == player_id
        ]

    def owns_group(self, player_id: UUID, group: str) -> bool:
        return all(
            self.property_states[tile.tile_id].owner_id == player_id
            for tile in self.board.group_members(group)
        )

    def amount_owed(self, player_id: UUID) -> int:
        return sum(debt.amount for debt in self.debts if debt.debtor_id == player_id)

    def rent_for(self, tile: PropertyTile, dice_total: int | None = None) -> int:
        """Aluguel devido ao cair na propriedade (sem multiplicadores de cartas)."""
        state = self.property_states[tile.tile_id]

        if state.owner_id is None or state.is_mortgaged:
            return 0

        if tile.kind == PropertyKind.STREET:
            if state.houses:
                return tile.rent_for_level(state.houses)
            if self.owns_group(state.owner_id, tile.group):
                return tile.monopoly_rent if tile.monopoly_rent is not None else 2 * tile.rent_values[0]
            return tile.rent_values[0]

        count = sum(
            1
            for member in self.board.group_members(tile.group)
            if self.property_states[member.tile_id].owner_id == state.owner_id
        )
        value = tile.rent_values[min(count, len(tile.rent_values)) - 1]

        if tile.kind == PropertyKind.UTILITY:
            if dice_total is None:
                dice_total = self.last_roll.total if self.last_roll else 0
            return value * dice_total

        return value

    def liquidation_value(self, player_id: UUID) -> int:
        """Quanto o jogador levanta vendendo construções e hipotecando tudo."""
        total = 0
        for tile in self.owned_tiles(player_id):
            state = self.property_states[tile.tile_id]
            total += state.houses * (tile.house_cost // 2)
            if not state.is_mortgaged:
                total += tile.mortgage_value
        return total

    def net_worth(self, player_id: UUID) -> int:
        """Dinheiro + imóveis (preço, ou preço - hipoteca) + construções a preço de custo."""
        player_state = self.player_states[player_id]
        if player_state.is_bankrupt:
            return 0

        total = player_state.money
        for tile in self.owned_tiles(player_id):
            state = self.property_states[tile.tile_id]
            total += tile.price - (tile.mortgage_value if state.is_mortgaged else 0)
            total += state.houses * tile.house_cost
        return total

    # -------------------------------------------------------------- telemetria

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Grava telemetria estruturada para análise e depuração."""
        if not self.record_events:
            return

        self.history_logs.append(
            {
                "turn": self.turn_counter,
                "current_player_id": self.current_player_id,
                "event": event_type,
                "data": payload,
            }
        )
