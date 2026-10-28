"""Information-rich observation contract for the complete game engine (v2)."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from gymnasium import spaces

from app.domain.game.board import PropertyKind, PropertyTile, SpecialTile, TaxTile
from app.domain.game.game_loop import GameLoop, MORTGAGE_TRANSFER_INTEREST_PERCENT
from app.domain.game.game_state import HOTEL_LEVEL


PHASES = (
    GameLoop.BEFORE_ROLL,
    GameLoop.AWAITING_PROPERTY_DECISION,
    GameLoop.AUCTION,
    GameLoop.AFTER_ROLL,
    GameLoop.RESOLVE_DEBT,
    GameLoop.GAME_OVER,
    GameLoop.RESOLVE_MORTGAGE_TRANSFER,
)
TILE_KINDS = (PropertyKind.STREET, PropertyKind.RAILROAD, PropertyKind.UTILITY, "tax", "special")


def _one_hot(index: int | None, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        result[index] = 1.0
    return result


class ObservationEncoderV2:
    """Returns public, decision-relevant state with seats relative to the observer."""

    def __init__(self, loop: GameLoop, max_players: int = 12, max_turns: int = 1000,
                 jail_card_ids: tuple[str, ...] = ()):
        self.max_players = max_players
        self.max_turns = max_turns
        self.board_size = loop.game.board.size
        self.tiles = tuple(loop.game.board.tiles)
        self.property_tiles = tuple(loop.game.board.property_tiles)
        self.tile_ids = tuple(tile.tile_id for tile in self.property_tiles)
        self.property_index = {tile_id: i for i, tile_id in enumerate(self.tile_ids)}
        self.jail_card_ids = jail_card_ids
        self.jail_card_index = {card_id: i for i, card_id in enumerate(jail_card_ids)}
        self.group_labels = tuple(sorted({tile.group for tile in self.property_tiles}))
        self.group_index = {group: i for i, group in enumerate(self.group_labels)}
        self.money_scale = float(max(1, loop.rules.starting_money, loop.rules.go_salary))
        self.total_debt_slots = max_players * max_players
        self.debt_row_size = 1 + max_players + (max_players + 1) + 1

        self.observation_space = spaces.Dict({
            "players": spaces.Dict({
                "active": spaces.Box(0, 1, (max_players,), np.float32),
                "money": spaces.Box(0, np.inf, (max_players,), np.float32),
                "position": spaces.Box(0, 1, (max_players, self.board_size), np.float32),
                "in_jail": spaces.Box(0, 1, (max_players,), np.float32),
                "jail_turns": spaces.Box(0, 1, (max_players,), np.float32),
                "jail_cards": spaces.Box(0, 1, (max_players, len(jail_card_ids)), np.float32),
                "current_turn_owner": spaces.Box(0, 1, (max_players,), np.float32),
                "current_actor": spaces.Box(0, 1, (max_players,), np.float32),
            }),
            "properties": spaces.Dict({
                "owner": spaces.Box(0, 1, (len(self.tile_ids), max_players + 1), np.float32),
                "building_level": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
                "mortgaged": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
            }),
            "board_features": spaces.Box(0, np.inf, (self.board_size, 18 + len(self.group_labels)), np.float32),
            "phase": spaces.Box(0, 1, (len(PHASES),), np.float32),
            "pending_tile": spaces.Box(0, 1, (self.board_size,), np.float32),
            "auction": spaces.Dict({
                "highest_bid": spaces.Box(0, np.inf, (1,), np.float32),
                "highest_bidder": spaces.Box(0, 1, (max_players + 1,), np.float32),
                "seller": spaces.Box(0, 1, (max_players + 1,), np.float32),
                "is_debt_auction": spaces.Box(0, 1, (1,), np.float32),
                "is_bankruptcy_auction": spaces.Box(0, 1, (1,), np.float32),
                "active_bidders": spaces.Box(0, 1, (max_players,), np.float32),
                "pending_bidders": spaces.Box(0, 1, (max_players,), np.float32),
            }),
            "debts": spaces.Box(0, np.inf, (self.total_debt_slots, self.debt_row_size), np.float32),
            "debt_resolution": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "debtor": spaces.Box(0, 1, (max_players,), np.float32),
                "negotiations_started": spaces.Box(0, 1, (1,), np.float32),
                "negotiations_closed": spaces.Box(0, 1, (1,), np.float32),
                "approached_players": spaces.Box(0, 1, (max_players,), np.float32),
                "auctioned_properties": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
            }),
            "pending_trade": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "participants": spaces.Box(0, 1, (2, max_players), np.float32),
                "money": spaces.Box(0, np.inf, (2,), np.float32),
                "properties": spaces.Box(0, 1, (2, len(self.tile_ids)), np.float32),
                "jail_cards": spaces.Box(0, 1, (2, len(jail_card_ids)), np.float32),
                "counteroffer_progress": spaces.Box(0, 1, (2,), np.float32),
            }),
            "globals": spaces.Box(0, np.inf, (11,), np.float32),
            "rules": spaces.Box(0, np.inf, (12,), np.float32),
            "mortgage_transfer": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "recipient": spaces.Box(0, 1, (max_players,), np.float32),
                "property": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
                "interest": spaces.Box(0, np.inf, (1,), np.float32),
                "redeem_cost": spaces.Box(0, np.inf, (1,), np.float32),
            }),
        })

    def encode(self, loop: GameLoop, observer: UUID) -> dict:
        game = loop.game
        order = game.turn_order
        observer_seat = order.index(observer)
        relative_ids = [order[(observer_seat + offset) % len(order)] for offset in range(len(order))]
        relative_slot = {player_id: offset for offset, player_id in enumerate(relative_ids)}
        actor = loop.current_decision_player_id

        players = {
            "active": np.zeros(self.max_players, dtype=np.float32),
            "money": np.zeros(self.max_players, dtype=np.float32),
            "position": np.zeros((self.max_players, self.board_size), dtype=np.float32),
            "in_jail": np.zeros(self.max_players, dtype=np.float32),
            "jail_turns": np.zeros(self.max_players, dtype=np.float32),
            "jail_cards": np.zeros((self.max_players, len(self.jail_card_ids)), dtype=np.float32),
            "current_turn_owner": np.zeros(self.max_players, dtype=np.float32),
            "current_actor": np.zeros(self.max_players, dtype=np.float32),
        }
        for player_id in order:
            slot = relative_slot[player_id]
            state = game.player_states[player_id]
            players["active"][slot] = float(not state.is_bankrupt)
            players["money"][slot] = state.money / self.money_scale
            players["position"][slot, game.position_of(player_id)] = 1.0
            players["in_jail"][slot] = float(state.is_in_jail)
            players["jail_turns"][slot] = state.jail_turns / max(1, loop.rules.max_jail_attempts)
            players["current_turn_owner"][slot] = float(player_id == game.current_player_id)
            players["current_actor"][slot] = float(player_id == actor)
            for deck_name, card in state.jail_cards:
                card_key = f"{deck_name}:{card.id}"
                if card_key in self.jail_card_index:
                    players["jail_cards"][slot, self.jail_card_index[card_key]] = 1.0

        properties = {
            "owner": np.zeros((len(self.tile_ids), self.max_players + 1), dtype=np.float32),
            "building_level": np.zeros(len(self.tile_ids), dtype=np.float32),
            "mortgaged": np.zeros(len(self.tile_ids), dtype=np.float32),
        }
        for tile_id, index in self.property_index.items():
            state = game.property_states[tile_id]
            owner_index = 0 if state.owner_id is None else 1 + relative_slot[state.owner_id]
            properties["owner"][index, owner_index] = 1.0
            properties["building_level"][index] = state.houses / HOTEL_LEVEL
            properties["mortgaged"][index] = float(state.is_mortgaged)

        board_features = self._board_features()
        phase = _one_hot(PHASES.index(loop.turn_phase), len(PHASES))
        pending_tile = _one_hot(loop.pending_tile_id, self.board_size)

        auction = {
            "highest_bid": np.asarray([loop.auction_highest_bid / self.money_scale], dtype=np.float32),
            "highest_bidder": np.zeros(self.max_players + 1, dtype=np.float32),
            "seller": np.zeros(self.max_players + 1, dtype=np.float32),
            "is_debt_auction": np.asarray([float(loop._auction_for_debt)], dtype=np.float32),
            "is_bankruptcy_auction": np.asarray(
                [float(loop._auction_for_bankruptcy)], dtype=np.float32
            ),
            "active_bidders": np.zeros(self.max_players, dtype=np.float32),
            "pending_bidders": np.zeros(self.max_players, dtype=np.float32),
        }
        self._set_player_one_hot(auction["highest_bidder"], loop.auction_highest_bidder, relative_slot)
        self._set_player_one_hot(auction["seller"], loop._auction_seller_id, relative_slot)
        for player_id in loop._auction_active:
            auction["active_bidders"][relative_slot[player_id]] = 1.0
        for player_id in loop._auction_pending:
            auction["pending_bidders"][relative_slot[player_id]] = 1.0

        debts = np.zeros((self.total_debt_slots, self.debt_row_size), dtype=np.float32)
        for row, debt in enumerate(game.debts[:self.total_debt_slots]):
            debts[row, 0] = 1.0
            debtor_slot = relative_slot[debt.debtor_id]
            debts[row, 1 + debtor_slot] = 1.0
            creditor_slot = 0 if debt.creditor_id is None else 1 + relative_slot[debt.creditor_id]
            creditor_start = 1 + self.max_players
            debts[row, creditor_start + creditor_slot] = 1.0
            debts[row, -1] = debt.amount / self.money_scale

        debt_resolution = {
            "active": np.zeros(1, dtype=np.float32),
            "debtor": np.zeros(self.max_players, dtype=np.float32),
            "negotiations_started": np.zeros(1, dtype=np.float32),
            "negotiations_closed": np.zeros(1, dtype=np.float32),
            "approached_players": np.zeros(self.max_players, dtype=np.float32),
            "auctioned_properties": np.zeros(len(self.tile_ids), dtype=np.float32),
        }
        resolution = game.debt_resolution
        if resolution is not None:
            debt_resolution["active"][0] = 1.0
            debt_resolution["debtor"][relative_slot[resolution.debtor_id]] = 1.0
            debt_resolution["negotiations_started"][0] = (
                resolution.negotiations_started / max(1, loop.rules.debt_negotiations_per_player)
            )
            debt_resolution["negotiations_closed"][0] = float(resolution.negotiations_closed)
            for player_id in resolution.negotiation_player_ids:
                debt_resolution["approached_players"][relative_slot[player_id]] = 1.0
            for tile_id in resolution.auctioned_property_ids:
                if tile_id in self.property_index:
                    debt_resolution["auctioned_properties"][self.property_index[tile_id]] = 1.0

        pending_trade = self._pending_trade(loop, relative_slot)
        mortgage_transfer = {
            "active": np.zeros(1, dtype=np.float32),
            "recipient": np.zeros(self.max_players, dtype=np.float32),
            "property": np.zeros(len(self.tile_ids), dtype=np.float32),
            "interest": np.zeros(1, dtype=np.float32),
            "redeem_cost": np.zeros(1, dtype=np.float32),
        }
        if loop._mortgage_transfer_queue:
            recipient_id, tile_id = loop._mortgage_transfer_queue[0]
            tile = game.board.get_tile_by_id(tile_id)
            mortgage_transfer["active"][0] = 1.0
            mortgage_transfer["recipient"][relative_slot[recipient_id]] = 1.0
            mortgage_transfer["property"][self.property_index[tile_id]] = 1.0
            mortgage_transfer["interest"][0] = (
                (
                    tile.mortgage_value * MORTGAGE_TRANSFER_INTEREST_PERCENT + 99
                ) // 100 / self.money_scale
            )
            mortgage_transfer["redeem_cost"][0] = (
                loop.unmortgage_cost(tile) / self.money_scale
            )
        last_roll = game.last_roll
        globals_ = np.asarray([
            loop.double_count / max(1, loop.rules.max_consecutive_doubles),
            game.houses_in_bank / max(1, loop.rules.max_houses_in_bank),
            game.hotels_in_bank / max(1, loop.rules.max_hotels_in_bank),
            game.house_shortage / max(1, loop.rules.max_houses_in_bank),
            game.turn_counter / max(1, self.max_turns),
            (last_roll.die1 / 6.0) if last_roll else 0.0,
            (last_roll.die2 / 6.0) if last_roll else 0.0,
            game.amount_owed(observer) / self.money_scale,
            float(loop._trade_window_open),
            float(game.game_over),
            float(game.truncated),
        ], dtype=np.float32)
        rules = np.asarray([
            loop.rules.debt_negotiations_per_player,
            loop.rules.trade_max_counteroffers,
            loop.rules.auction_min_bid / self.money_scale,
            loop.rules.go_salary / self.money_scale,
            loop.rules.jail_fine / self.money_scale,
            loop.rules.starting_money / self.money_scale,
            float(loop.rules.trade_enabled),
            float(loop.rules.debt_resolution_enabled),
            float(loop.rules.auction_after_declined_purchase()),
            float(loop.rules.income_tax_choice),
            float(loop.rules.bankruptcy_to_bank),
            MORTGAGE_TRANSFER_INTEREST_PERCENT / 100.0,
        ], dtype=np.float32)

        return {
            "players": players,
            "properties": properties,
            "board_features": board_features,
            "phase": phase,
            "pending_tile": pending_tile,
            "auction": auction,
            "debts": debts,
            "debt_resolution": debt_resolution,
            "pending_trade": pending_trade,
            "globals": globals_,
            "rules": rules,
            "mortgage_transfer": mortgage_transfer,
        }

    def _set_player_one_hot(self, vector, player_id, relative_slot):
        if player_id is None:
            vector[0] = 1.0
        else:
            vector[1 + relative_slot[player_id]] = 1.0

    def _pending_trade(self, loop, relative_slot):
        values = {
            "active": np.zeros(1, dtype=np.float32),
            "participants": np.zeros((2, self.max_players), dtype=np.float32),
            "money": np.zeros(2, dtype=np.float32),
            "properties": np.zeros((2, len(self.tile_ids)), dtype=np.float32),
            "jail_cards": np.zeros((2, len(self.jail_card_ids)), dtype=np.float32),
            "counteroffer_progress": np.zeros(2, dtype=np.float32),
        }
        negotiation = loop.game.pending_trade
        if negotiation is None:
            return values
        offer = negotiation.current_offer
        values["active"][0] = 1.0
        values["participants"][0, relative_slot[offer.proposer_id]] = 1.0
        values["participants"][1, relative_slot[offer.recipient_id]] = 1.0
        values["money"][:] = (offer.money_offered / self.money_scale, offer.money_requested / self.money_scale)
        for side, property_ids in enumerate((offer.properties_offered, offer.properties_requested)):
            for tile_id in property_ids:
                if tile_id in self.property_index:
                    values["properties"][side, self.property_index[tile_id]] = 1.0
        for side, card_ids in enumerate((offer.jail_cards_offered, offer.jail_cards_requested)):
            for card_id in card_ids:
                if card_id in self.jail_card_index:
                    values["jail_cards"][side, self.jail_card_index[card_id]] = 1.0
        values["counteroffer_progress"][:] = (
            negotiation.counteroffers_made / max(1, loop.rules.trade_max_counteroffers),
            float(negotiation.is_debt_trade),
        )
        return values

    def _board_features(self) -> np.ndarray:
        rows = np.zeros((self.board_size, 18 + len(self.group_labels)), dtype=np.float32)
        for tile in self.tiles:
            features = rows[tile.tile_id]
            if isinstance(tile, PropertyTile):
                kind = tile.kind
                features[TILE_KINDS.index(kind)] = 1.0
                features[5] = tile.price / self.money_scale
                features[6] = tile.mortgage_value / self.money_scale
                features[7] = (tile.unmortgage_cost or 0) / self.money_scale
                features[8] = tile.house_cost / self.money_scale
                features[9] = tile.hotel_cost / self.money_scale
                for i, rent in enumerate(tile.rent_values[:6]):
                    features[10 + i] = rent / self.money_scale
                features[16] = (tile.monopoly_rent or 0) / self.money_scale
                features[18 + self.group_index[tile.group]] = 1.0
            elif isinstance(tile, TaxTile):
                features[3] = 1.0
                features[17] = tile.tax_amount / self.money_scale
            elif isinstance(tile, SpecialTile):
                features[4] = 1.0
        return rows
