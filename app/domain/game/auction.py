"""Public records captured for auctions and bidder decisions."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class AuctionParticipantSnapshot:
    player_id: UUID
    money: int
    net_worth: int
    position: int
    is_in_jail: bool
    amount_owed: int


@dataclass(frozen=True, slots=True)
class AuctionPropertySnapshot:
    tile_id: int
    owner_id: UUID | None
    houses: int
    is_mortgaged: bool


@dataclass(frozen=True, slots=True)
class AuctionSessionRecord:
    auction_id: int
    tile_id: int
    turn: int
    auction_type: str  # BANK, DEBT, or BANKRUPTCY
    seller_id: UUID | None
    opening_bid: int
    participants: tuple[UUID, ...]
    player_snapshots: tuple[AuctionParticipantSnapshot, ...]
    property_snapshots: tuple[AuctionPropertySnapshot, ...]


@dataclass(frozen=True, slots=True)
class AuctionDecisionRecord:
    auction_id: int
    turn: int
    player_id: UUID
    action: str  # BID or WITHDRAW
    bid_amount: int | None
    highest_bid_before: int
    leader_before: UUID | None
    money_before: int
    remaining_bidders: int


@dataclass(frozen=True, slots=True)
class AuctionResultRecord:
    auction_id: int
    tile_id: int
    turn: int
    winner_id: UUID | None
    winning_bid: int
    seller_id: UUID | None
    auction_type: str
