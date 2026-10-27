"""Espaço de ações discreto e fixo sobre as `GameAction` da engine.

Layout (P = número de imóveis, 28 no tabuleiro clássico):

    0  ROLL_DICE            4  PAY_JAIL_FINE         8..11  AUCTION_BID (+1, +10, +50, +100
    1  BUY_PROPERTY         5  USE_JAIL_CARD                sobre o maior lance)
    2  DECLINE_PROPERTY     6  DECLARE_BANKRUPTCY
    3  END_TURN             7  AUCTION_WITHDRAW
    12      .. 12+P-1    BUILD_HOUSE[p]
    12+P    .. 12+2P-1   SELL_HOUSE[p]
    12+2P   .. 12+3P-1   MORTGAGE[p]
    12+3P   .. 12+4P-1   UNMORTGAGE[p]

Regra de compatibilidade: versões futuras só *acrescentam* blocos no final
(por exemplo, negociação), nunca reordenam os existentes.
"""

import numpy as np

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.board import Board
from app.domain.game.game_loop import AUCTION_BID_INCREMENTS, GameLoop

SIMPLE_ACTIONS: tuple[str, ...] = (
    ActionType.ROLL_DICE,
    ActionType.BUY_PROPERTY,
    ActionType.DECLINE_PROPERTY,
    ActionType.END_TURN,
    ActionType.PAY_JAIL_FINE,
    ActionType.USE_JAIL_CARD,
    ActionType.DECLARE_BANKRUPTCY,
    ActionType.AUCTION_WITHDRAW,
)
PROPERTY_ACTIONS: tuple[str, ...] = (
    ActionType.BUILD_HOUSE,
    ActionType.SELL_HOUSE,
    ActionType.MORTGAGE,
    ActionType.UNMORTGAGE,
)
BID_OFFSET = len(SIMPLE_ACTIONS)
PROPERTY_OFFSET = BID_OFFSET + len(AUCTION_BID_INCREMENTS)


class DiscreteActionSpace:
    def __init__(self, board: Board) -> None:
        self.tile_ids = tuple(tile.tile_id for tile in board.property_tiles)
        self.num_properties = len(self.tile_ids)
        self._prop_index = {tile_id: i for i, tile_id in enumerate(self.tile_ids)}
        self.size = PROPERTY_OFFSET + len(PROPERTY_ACTIONS) * self.num_properties

    def to_game_action(self, index: int, loop: GameLoop) -> GameAction:
        """Converte o índice em GameAction (o lance depende do estado do leilão)."""
        if not 0 <= index < self.size:
            raise ValueError(f"Ação fora do espaço: {index}")

        if index < BID_OFFSET:
            return GameAction(SIMPLE_ACTIONS[index])

        if index < PROPERTY_OFFSET:
            increment = AUCTION_BID_INCREMENTS[index - BID_OFFSET]
            return GameAction(ActionType.AUCTION_BID, {"amount": loop.auction_bid_amount(increment)})

        block, prop = divmod(index - PROPERTY_OFFSET, self.num_properties)
        return GameAction(PROPERTY_ACTIONS[block], {"tile_id": self.tile_ids[prop]})

    def to_index(self, action: GameAction, loop: GameLoop) -> list[int]:
        """Índices que produzem `action` (um lance pode coincidir com vários incrementos)."""
        kind = action.action_type
        if kind in SIMPLE_ACTIONS:
            return [SIMPLE_ACTIONS.index(kind)]

        if kind == ActionType.AUCTION_BID:
            return [
                BID_OFFSET + i
                for i, inc in enumerate(AUCTION_BID_INCREMENTS)
                if loop.auction_bid_amount(inc) == action.payload["amount"]
            ]

        block = PROPERTY_ACTIONS.index(kind)
        return [PROPERTY_OFFSET + block * self.num_properties + self._prop_index[action.payload["tile_id"]]]

    def mask(self, loop: GameLoop) -> np.ndarray:
        mask = np.zeros(self.size, dtype=np.int8)
        for action in loop.legal_actions():
            mask[self.to_index(action, loop)] = 1
        return mask
