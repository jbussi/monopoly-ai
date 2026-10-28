"""Value objects and payload parsing for bilateral trade negotiations."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class TradeOffer:
    """One proposal, expressed from proposer to recipient."""

    negotiation_id: int
    proposer_id: UUID
    recipient_id: UUID
    money_offered: int = 0
    money_requested: int = 0
    properties_offered: tuple[int, ...] = ()
    properties_requested: tuple[int, ...] = ()
    jail_cards_offered: tuple[str, ...] = ()
    jail_cards_requested: tuple[str, ...] = ()
    counteroffer_number: int = 0


@dataclass(slots=True)
class TradeNegotiation:
    current_offer: TradeOffer
    counteroffers_made: int = 0
    is_debt_trade: bool = False


def payload_to_offer(
    payload: dict,
    negotiation_id: int,
    proposer_id: UUID,
    recipient_id: UUID,
    counteroffer_number: int = 0,
) -> TradeOffer:
    money_offered = payload.get("money_offered", 0)
    money_requested = payload.get("money_requested", 0)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (money_offered, money_requested)
    ):
        raise ValueError("Os valores em dinheiro devem ser inteiros não negativos.")

    def ids(key: str, value_type: type) -> tuple:
        values = payload.get(key, ())
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"'{key}' deve ser uma lista.")
        if any(not isinstance(value, value_type) for value in values):
            label = "inteiros" if value_type is int else "textos"
            raise ValueError(f"'{key}' deve conter apenas {label}.")
        if len(values) != len(set(values)):
            raise ValueError(f"'{key}' não pode conter itens repetidos.")
        return tuple(values)

    return TradeOffer(
        negotiation_id=negotiation_id,
        proposer_id=proposer_id,
        recipient_id=recipient_id,
        money_offered=money_offered,
        money_requested=money_requested,
        properties_offered=ids("properties_offered", int),
        properties_requested=ids("properties_requested", int),
        jail_cards_offered=ids("jail_cards_offered", str),
        jail_cards_requested=ids("jail_cards_requested", str),
        counteroffer_number=counteroffer_number,
    )
