
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RentSchedule:
    base: int
    monopoly: int
    one_house: int
    two_houses: int
    three_houses: int
    four_houses: int
    hotel: int


@dataclass(frozen=True, slots=True)
class BoardSpace:
    position: int
    type: str
    name: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PropertyDefinition(BoardSpace):
    color_group: str = ""
    price: int = 0
    house_cost: int = 0
    hotel_cost: int = 0
    mortgage: int = 0
    unmortgage_cost: int = 0
    rent: RentSchedule | None = None


@dataclass(frozen=True, slots=True)
class RailroadDefinition(BoardSpace):
    price: int = 0
    mortgage: int = 0
    unmortgage_cost: int = 0
    rent: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class UtilityDefinition(BoardSpace):
    price: int = 0
    mortgage: int = 0
    unmortgage_cost: int = 0
    rent: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class TaxDefinition(BoardSpace):
    amount: int = 0
    percent_option: int | None = None
    tax_type: str | None = None


@dataclass(frozen=True, slots=True)
class CardDefinition:
    id: str
    text: str
    action: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GameConfig:
    board: tuple[BoardSpace, ...]
    chance_cards: tuple[CardDefinition, ...]
    community_chest_cards: tuple[CardDefinition, ...]
    settings: Mapping[str, Any]
    variant: str



@dataclass(frozen=True, slots=True)
class GameSettings:
    starting_money: int
    go_salary: int

    income_tax_amount: int
    income_tax_percent: int
    income_tax_choice: bool
    luxury_tax_amount: int

    jail_fine: int
    jail_max_turns: int
    go_to_jail_position: int
    jail_position: int

    max_houses_in_bank: int
    max_hotels_in_bank: int

    auction_enabled: bool
    auction_min_bid: int
    auction_timeout_seconds: int

    free_parking_jackpot: bool
    max_players: int
    min_players_to_start: int
    allow_spectators: bool
    trade_enabled: bool
    bankruptcy_to_bank: bool

    player_colors: tuple[dict[str, Any], ...]
    player_tokens: tuple[str, ...]


