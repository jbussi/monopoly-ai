"""Codificação do estado em vetor float32, do ponto de vista de um jogador.

Os jogadores são *rotacionados*: o slot 0 é sempre o observador, o slot 1 é o próximo
na ordem da mesa, e assim por diante. Há sempre `max_players` slots (vazios ficam
zerados), para que a mesma rede sirva para partidas de 2 a `max_players` jogadores.

Os offsets exatos ficam em `ObservationEncoder.layout`; o significado está em docs/ambiente.md.
"""

from uuid import UUID

import numpy as np

from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import HOTEL_LEVEL

MONEY_SCALE = 1500.0

PHASES = (
    GameLoop.BEFORE_ROLL,
    GameLoop.AWAITING_PROPERTY_DECISION,
    GameLoop.AUCTION,
    GameLoop.AFTER_ROLL,
    GameLoop.RESOLVE_DEBT,
    GameLoop.GAME_OVER,
)


class ObservationEncoder:
    def __init__(self, loop: GameLoop, max_players: int = 4, max_turns: int = 1000) -> None:
        board = loop.game.board
        self.board_size = board.size
        self.tile_ids = tuple(tile.tile_id for tile in board.property_tiles)
        self.max_players = max_players
        self.max_turns = max_turns
        self.total_houses = loop.rules.max_houses_in_bank
        self.total_hotels = loop.rules.max_hotels_in_bank

        self._player_fields = 1 + 1 + self.board_size + 3 + 1
        self._prop_fields = (max_players + 1) + 2
        sections = (
            ("players", max_players * self._player_fields),
            ("properties", len(self.tile_ids) * self._prop_fields),
            ("phase", len(PHASES)),
            ("pending_property", len(self.tile_ids)),
            ("auction", 1 + max_players + 1),
            ("globals", 7),
        )
        self.layout: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, size in sections:
            self.layout[name] = (offset, offset + size)
            offset += size
        self.size = offset

    def encode(self, loop: GameLoop, observer: UUID) -> np.ndarray:
        game = loop.game
        obs = np.zeros(self.size, dtype=np.float32)

        order = game.turn_order
        n = len(order)
        base = order.index(observer)
        slot_of = {order[(base + k) % n]: k for k in range(n)}

        start = self.layout["players"][0]
        for player_id in order:
            state = game.player_states[player_id]
            if state.is_bankrupt:
                continue
            o = start + slot_of[player_id] * self._player_fields
            obs[o] = 1.0
            obs[o + 1] = state.money / MONEY_SCALE
            obs[o + 2 + game.position_of(player_id)] = 1.0
            b = o + 2 + self.board_size
            obs[b] = float(state.is_in_jail)
            obs[b + 1] = state.jail_turns / 3.0
            obs[b + 2] = len(state.jail_cards) / 2.0
            obs[b + 3] = float(player_id == game.current_player_id)

        start = self.layout["properties"][0]
        for i, tile_id in enumerate(self.tile_ids):
            prop = game.property_states[tile_id]
            o = start + i * self._prop_fields
            obs[o + (0 if prop.owner_id is None else 1 + slot_of[prop.owner_id])] = 1.0
            b = o + self.max_players + 1
            obs[b] = prop.houses / HOTEL_LEVEL
            obs[b + 1] = float(prop.is_mortgaged)

        obs[self.layout["phase"][0] + PHASES.index(loop.turn_phase)] = 1.0

        if loop.pending_tile_id is not None:
            obs[self.layout["pending_property"][0] + self.tile_ids.index(loop.pending_tile_id)] = 1.0

        a = self.layout["auction"][0]
        obs[a] = loop.auction_highest_bid / MONEY_SCALE
        bidder = loop.auction_highest_bidder
        obs[a + 1 + (0 if bidder is None else 1 + slot_of[bidder])] = 1.0

        g = self.layout["globals"][0]
        obs[g] = loop.double_count / 3.0
        obs[g + 1] = game.amount_owed(observer) / MONEY_SCALE
        obs[g + 2] = game.houses_in_bank / self.total_houses
        obs[g + 3] = game.hotels_in_bank / self.total_hotels
        obs[g + 4] = game.turn_counter / self.max_turns
        if game.last_roll is not None:
            obs[g + 5] = game.last_roll.die1 / 6.0
            obs[g + 6] = game.last_roll.die2 / 6.0
        return obs
