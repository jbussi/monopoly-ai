from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from enum import StrEnum


class CardAction(StrEnum):
    ADVANCE_TO = "advance_to"
    ADVANCE_TO_NEAREST = "advance_to_nearest"
    COLLECT = "collect"
    PAY = "pay"
    PAY_EACH_PLAYER = "pay_each_player"
    COLLECT_FROM_EACH_PLAYER = "collect_from_each_player"
    GO_TO_JAIL = "go_to_jail"
    GET_OUT_OF_JAIL = "get_out_of_jail"
    GO_BACK = "go_back"
    REPAIRS = "repairs"
    BANK_ERROR = "bank_error"
    BIRTHDAY = "birthday"

class CardLike(Protocol):
    id: str
    action: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CardExecutionResult:
    card_id: str
    action: str
    success: bool
    message: str


CardActionHandler = Callable[
    [CardLike, Any],
    CardExecutionResult,
]


class CardActionRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, CardActionHandler] = {}

    def register(
        self,
        action: str,
        handler: CardActionHandler,
    ) -> None:
        if not action.strip():
            raise ValueError("O nome da ação não pode ser vazio.")

        if action in self._handlers:
            raise ValueError(
                f"A ação '{action}' já está registrada."
            )

        self._handlers[action] = handler

    def execute(
        self,
        card: CardLike,
        context: Any,
    ) -> CardExecutionResult:
        handler = self._handlers.get(card.action)

        if handler is None:
            raise ValueError(
                f"Ação de carta não registrada: '{card.action}'."
            )

        return handler(card, context)

    @property
    def registered_actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))