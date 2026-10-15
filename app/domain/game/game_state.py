
from dataclasses import dataclass
from uuid import UUID

from .player import Player
from .pieces import Piece
from .board import Board, PropertyTile


@dataclass
class PlayerState:
    """Estado mutável de um participante na partida."""

    money: int
    is_in_jail: bool = False
    jail_turns: int = 0
    is_bankrupt: bool = False
    jail_cards: int = 0


@dataclass
class PieceState:
    """Estado mutável de uma peça."""

    position: int = 0


@dataclass
class PropertyState:
    """Estado mutável de uma propriedade do tabuleiro."""

    owner_id: UUID | None = None
    houses: int = 0
    is_mortgaged: bool = False


class GameState:
    """Representa a situação atual de uma partida."""

    def __init__(
        self,
        players: list[Player],
        pieces: list[Piece],
        board: Board,
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

        self.player_states: dict[UUID, PlayerState] = {
            player.player_id: PlayerState() for player in players
        }

        self.pieces: dict[UUID, Piece] = {
            piece.piece_id: piece for piece in pieces
        }

        self.piece_states: dict[UUID, PieceState] = {
            piece.piece_id: PieceState() for piece in pieces
        }

        self.board = board

        self.property_states: dict[int, PropertyState] = {
            tile.tile_id: PropertyState()
            for tile in board.tiles
            if isinstance(tile, PropertyTile)
        }

        self.current_player_id: UUID = players[0].player_id
        self.turn_counter: int = 1

        self.game_over: bool = False
        self.winner_id: UUID | None = None

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_id]

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
