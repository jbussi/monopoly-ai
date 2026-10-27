
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class GameAction:
    """Representa um comando enviado à engine."""

    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_type.strip():
            raise ValueError("O tipo da ação não pode ser vazio.")

        if not isinstance(self.payload, dict):
            raise TypeError("O payload da ação deve ser um dicionário.")


class ActionType(StrEnum):
    """Tipos de ação aceitos pela engine."""

    ROLL_DICE = "ROLL_DICE"
    BUY_PROPERTY = "BUY_PROPERTY"
    DECLINE_PROPERTY = "DECLINE_PROPERTY"
    END_TURN = "END_TURN"
    PAY_JAIL_FINE = "PAY_JAIL_FINE"
    USE_JAIL_CARD = "USE_JAIL_CARD"
    DECLARE_BANKRUPTCY = "DECLARE_BANKRUPTCY"
    AUCTION_BID = "AUCTION_BID"            # payload: {"amount": int}
    AUCTION_WITHDRAW = "AUCTION_WITHDRAW"
    BUILD_HOUSE = "BUILD_HOUSE"            # payload: {"tile_id": int}
    SELL_HOUSE = "SELL_HOUSE"              # payload: {"tile_id": int}
    MORTGAGE = "MORTGAGE"                  # payload: {"tile_id": int}
    UNMORTGAGE = "UNMORTGAGE"              # payload: {"tile_id": int}
