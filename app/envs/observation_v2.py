"""Information-rich observation contract for the complete game engine (v2)."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from gymnasium import spaces

from app.domain.game.board import PropertyKind, PropertyTile, SpecialTile, TaxTile
from app.domain.game.actions import ActionType
from app.domain.game.game_loop import GameLoop, MORTGAGE_TRANSFER_INTEREST_PERCENT
from app.domain.game.game_state import HOTEL_LEVEL
from app.domain.analytics.decisions.jail_forecast import (
    DEFAULT_JAIL_FORECAST_SAMPLES,
    empty_jail_forecast,
    forecast_jail_decisions,
)
from app.domain.analytics.economics.stationary_board import (
    calculate_stationary_board_analysis,
    estimate_group_state_income_per_turn,
    evaluate_marginal_purchase,
    evaluate_group_portfolios,
)
from app.envs.cash_flow_forecast import (
    FORECAST_HORIZONS,
    FORECAST_SCENARIOS,
    FORECAST_STAT_NAMES,
    JAIL_OPTIONS,
    DEFAULT_FORECAST_SAMPLES,
    DEFAULT_MANAGEMENT_FORECAST_SAMPLES,
    MANAGEMENT_ACTION_TYPES,
    empty_cash_flow_forecast,
    forecast_property_cash_flows,
    forecast_management_cash_flows,
    forecast_jail_decisions as forecast_jail_cash_flows,
)
from app.infraestructure.game_config.models import GameConfig


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
                 jail_card_ids: tuple[str, ...] = (),
                 jail_forecast_samples: int = DEFAULT_JAIL_FORECAST_SAMPLES,
                 config: GameConfig | None = None,
                 max_trade_history: int = 16):
        self.max_players = max_players
        self.max_turns = max_turns
        self.jail_forecast_samples = jail_forecast_samples
        if isinstance(max_trade_history, bool) or not isinstance(max_trade_history, int) or max_trade_history < 1:
            raise ValueError("max_trade_history deve ser um inteiro positivo.")
        self.max_trade_history = max_trade_history
        self.board_size = loop.game.board.size
        self.tiles = tuple(loop.game.board.tiles)
        self.property_tiles = tuple(loop.game.board.property_tiles)
        self.tile_ids = tuple(tile.tile_id for tile in self.property_tiles)
        self.property_index = {tile_id: i for i, tile_id in enumerate(self.tile_ids)}
        self.jail_card_ids = jail_card_ids
        self.jail_card_index = {card_id: i for i, card_id in enumerate(jail_card_ids)}
        self.deck_card_ids = {
            name: tuple(deck.card_ids) for name, deck in loop.game.decks.items()
        }
        self.group_labels = tuple(sorted({tile.group for tile in self.property_tiles}))
        self.group_index = {group: i for i, group in enumerate(self.group_labels)}
        self.special_labels = tuple(sorted({
            tile.tile_type for tile in self.tiles if isinstance(tile, SpecialTile)
        }))
        self.special_index = {tile_type: i for i, tile_type in enumerate(self.special_labels)}
        self.money_scale = float(max(1, loop.rules.starting_money, loop.rules.go_salary))
        self.config = config
        if config is None:
            raise ValueError("ObservationEncoderV2 precisa da GameConfig para análises econômicas.")
        if tuple(space.position for space in config.board) != tuple(tile.tile_id for tile in self.tiles):
            raise ValueError("A GameConfig deve corresponder ao tabuleiro da partida.")
        self.stationary_analysis = calculate_stationary_board_analysis(config)
        self.portfolio_scenarios = evaluate_group_portfolios(
            config, self.stationary_analysis, number_of_opponents=1
        )
        self.portfolio_membership = np.zeros(
            (len(self.portfolio_scenarios), len(self.tile_ids)), dtype=np.float32
        )
        self.portfolio_groups = np.zeros(
            (len(self.portfolio_scenarios), len(self.group_labels)), dtype=np.float32
        )
        self.portfolio_development = np.zeros(len(self.portfolio_scenarios), dtype=np.float32)
        self.portfolio_investment = np.zeros(len(self.portfolio_scenarios), dtype=np.float32)
        self.portfolio_income = np.zeros(len(self.portfolio_scenarios), dtype=np.float32)
        self.portfolio_yield = np.zeros(len(self.portfolio_scenarios), dtype=np.float32)
        for row, scenario in enumerate(self.portfolio_scenarios):
            for tile_id in scenario.owned_tile_ids:
                self.portfolio_membership[row, self.property_index[tile_id]] = 1.0
            if scenario.group in self.group_index:
                self.portfolio_groups[row, self.group_index[scenario.group]] = 1.0
            self.portfolio_development[row] = scenario.development_level / 5.0
            self.portfolio_investment[row] = scenario.investment_cost / self.money_scale
            self.portfolio_income[row] = (
                scenario.gross_income_per_turn / self.money_scale
            )
            self.portfolio_yield[row] = scenario.gross_yield_per_turn or 0.0
        self.total_debt_slots = max_players * max_players
        self.debt_row_size = 1 + max_players + (max_players + 1) + 1
        forecast_shape = (len(FORECAST_SCENARIOS), len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES))
        property_decision_shape = (len(self.tile_ids),)
        management_shape = (
            len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids),
            2, len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES),
        )
        jail_forecast_shape = (
            len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES)
        )

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
            "board_features": spaces.Box(
                0, np.inf,
                (self.board_size, 19 + len(self.group_labels) + len(self.special_labels)),
                np.float32,
            ),
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
            "trade_history": spaces.Dict({
                "active": spaces.Box(0, 1, (max_trade_history,), np.float32),
                "participants": spaces.Box(0, 1, (max_trade_history, 2, max_players), np.float32),
                "money": spaces.Box(0, np.inf, (max_trade_history, 2), np.float32),
                "properties": spaces.Box(0, 1, (max_trade_history, 2, len(self.tile_ids)), np.float32),
                "jail_cards": spaces.Box(0, 1, (max_trade_history, 2, len(jail_card_ids)), np.float32),
                "counteroffer_number": spaces.Box(0, 1, (max_trade_history,), np.float32),
                "response": spaces.Box(0, 1, (max_trade_history, 3), np.float32),
                "responder": spaces.Box(0, 1, (max_trade_history, max_players), np.float32),
                "age": spaces.Box(0, 1, (max_trade_history,), np.float32),
                "participant_cash": spaces.Box(0, np.inf, (max_trade_history, 2), np.float32),
                "participant_position": spaces.Box(0, 1, (max_trade_history, 2, self.board_size), np.float32),
                "participant_in_jail": spaces.Box(0, 1, (max_trade_history, 2), np.float32),
                "participant_jail_cards": spaces.Box(0, 1, (max_trade_history, 2, len(jail_card_ids)), np.float32),
                "participant_net_worth": spaces.Box(0, np.inf, (max_trade_history, 2), np.float32),
                "participant_debt": spaces.Box(0, np.inf, (max_trade_history, 2), np.float32),
                "property_owner": spaces.Box(0, 1, (max_trade_history, len(self.tile_ids), max_players + 1), np.float32),
                "property_building_level": spaces.Box(0, 1, (max_trade_history, len(self.tile_ids)), np.float32),
                "property_mortgaged": spaces.Box(0, 1, (max_trade_history, len(self.tile_ids)), np.float32),
                "phase": spaces.Box(0, 1, (max_trade_history, len(PHASES)), np.float32),
                "is_debt_trade": spaces.Box(0, 1, (max_trade_history,), np.float32),
            }),
            "globals": spaces.Box(0, np.inf, (11,), np.float32),
            "dice": spaces.Dict({
                "sides": spaces.Box(2, np.inf, (2,), np.float32),
                "double_probability": spaces.Box(0, 1, (1,), np.float32),
            }),
            "jail_forecast": spaces.Dict({
                "available": spaces.Box(0, 1, (3,), np.float32),
                "expected_cash_delta": spaces.Box(-np.inf, np.inf, (3, 3), np.float32),
                "cash_delta_std": spaces.Box(0, np.inf, (3, 3), np.float32),
                "expected_debt": spaces.Box(0, np.inf, (3, 3), np.float32),
                "probability_in_jail": spaces.Box(0, 1, (3, 3), np.float32),
                "probability_of_debt": spaces.Box(0, 1, (3, 3), np.float32),
                "expected_jail_cards_remaining": spaces.Box(0, np.inf, (3, 3), np.float32),
                "expected_rent_charged": spaces.Box(0, np.inf, (3, 3), np.float32),
                "expected_tax_charged": spaces.Box(0, np.inf, (3, 3), np.float32),
                "expected_go_salary": spaces.Box(0, np.inf, (3, 3), np.float32),
                "property_opportunities": spaces.Box(
                    0, np.inf, (3, 3, len(self.tile_ids)), np.float32
                ),
                "position_probability": spaces.Box(
                    0, 1, (3, 3, self.board_size), np.float32
                ),
            }),
            "rules": spaces.Box(0, np.inf, (12,), np.float32),
            # Counts reveal public card composition, never the shuffled order.
            "deck_card_counts": spaces.Dict({
                deck_name: spaces.Dict({
                    card_id: spaces.Box(0, np.inf, (2,), np.float32)
                    for card_id in card_ids
                })
                for deck_name, card_ids in self.deck_card_ids.items()
            }),
            "rules_complete": spaces.Dict({
                "starting_money": spaces.Box(0, np.inf, (1,), np.float32),
                "go_salary": spaces.Box(0, np.inf, (1,), np.float32),
                "jail_fine": spaces.Box(0, np.inf, (1,), np.float32),
                "max_consecutive_doubles": spaces.Box(0, np.inf, (1,), np.float32),
                "max_jail_attempts": spaces.Box(0, np.inf, (1,), np.float32),
                "jail_position": spaces.Box(0, 1, (self.board_size,), np.float32),
                "go_to_jail_position": spaces.Box(0, 1, (self.board_size,), np.float32),
                "max_houses_in_bank": spaces.Box(0, np.inf, (1,), np.float32),
                "max_hotels_in_bank": spaces.Box(0, np.inf, (1,), np.float32),
                "auction_min_bid": spaces.Box(0, np.inf, (1,), np.float32),
                "income_tax_choice": spaces.Box(0, 1, (1,), np.float32),
                "max_turns_enabled": spaces.Box(0, 1, (1,), np.float32),
                "max_turns": spaces.Box(0, np.inf, (1,), np.float32),
                "trade_enabled": spaces.Box(0, 1, (1,), np.float32),
                "trade_max_counteroffers": spaces.Box(0, np.inf, (1,), np.float32),
                "debt_resolution_enabled": spaces.Box(0, 1, (1,), np.float32),
                "debt_negotiations_per_player": spaces.Box(0, np.inf, (1,), np.float32),
                "bankruptcy_to_bank": spaces.Box(0, 1, (1,), np.float32),
                "auction_after_declined_purchase": spaces.Box(0, 1, (1,), np.float32),
            }),
            "mortgage_transfer": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "recipient": spaces.Box(0, 1, (max_players,), np.float32),
                "property": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
                "interest": spaces.Box(0, np.inf, (1,), np.float32),
                "redeem_cost": spaces.Box(0, np.inf, (1,), np.float32),
            }),
            "stationary_economics": spaces.Dict({
                "property_arrivals_per_roll": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
                "utility_dice_total_per_roll": spaces.Box(0, np.inf, (len(self.tile_ids),), np.float32),
                "railroad_rent_multiplier_per_roll": spaces.Box(0, np.inf, (len(self.tile_ids),), np.float32),
                "expected_rolls_per_turn": spaces.Box(0, np.inf, (1,), np.float32),
                "portfolio_membership": spaces.Box(0, 1, self.portfolio_membership.shape, np.float32),
                "portfolio_groups": spaces.Box(0, 1, self.portfolio_groups.shape, np.float32),
                "portfolio_development": spaces.Box(0, 1, self.portfolio_development.shape, np.float32),
                "portfolio_investment": spaces.Box(0, np.inf, self.portfolio_investment.shape, np.float32),
                "portfolio_income_per_turn": spaces.Box(0, np.inf, self.portfolio_income.shape, np.float32),
                "portfolio_gross_yield_per_turn": spaces.Box(0, np.inf, self.portfolio_yield.shape, np.float32),
            }),
            "property_decision_valuation": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "candidate_property": spaces.Box(0, 1, property_decision_shape, np.float32),
                "owned_group_properties": spaces.Box(0, 1, property_decision_shape, np.float32),
                "group_complete_after_purchase": spaces.Box(0, 1, (1,), np.float32),
                "reference_price": spaces.Box(0, np.inf, (1,), np.float32),
                "next_auction_bid": spaces.Box(0, np.inf, (1,), np.float32),
                "income_before_per_turn": spaces.Box(0, np.inf, (1,), np.float32),
                "income_after_per_turn": spaces.Box(0, np.inf, (1,), np.float32),
                "marginal_income_per_turn": spaces.Box(0, np.inf, (1,), np.float32),
                "marginal_yield_at_reference_price": spaces.Box(0, np.inf, (1,), np.float32),
                "payback_turns": spaces.Box(0, np.inf, (1,), np.float32),
                "payback_available": spaces.Box(0, 1, (1,), np.float32),
            }),
            "management_decision_valuation": spaces.Dict({
                "available": spaces.Box(0, 1, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "cash_delta": spaces.Box(-np.inf, np.inf, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "income_before_per_turn": spaces.Box(0, np.inf, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "income_after_per_turn": spaces.Box(0, np.inf, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "marginal_income_per_turn": spaces.Box(-np.inf, np.inf, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "marginal_yield_per_turn": spaces.Box(-np.inf, np.inf, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
            }),
            "management_cash_forecast": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "sample_count": spaces.Box(0, np.inf, (1,), np.float32),
                "available": spaces.Box(0, 1, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids)), np.float32),
                "cash_change": spaces.Box(-np.inf, np.inf, management_shape, np.float32),
                "cash_in": spaces.Box(0, np.inf, management_shape, np.float32),
                "cash_out": spaces.Box(0, np.inf, management_shape, np.float32),
                "ending_cash": spaces.Box(0, np.inf, management_shape, np.float32),
                "minimum_cash": spaces.Box(0, np.inf, management_shape, np.float32),
                "debt_probability": spaces.Box(0, 1, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids), 2, len(FORECAST_HORIZONS)), np.float32),
                "bankruptcy_probability": spaces.Box(0, 1, (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids), 2, len(FORECAST_HORIZONS)), np.float32),
            }),
            "cash_flow_forecast": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "horizons": spaces.Box(0, np.inf, (len(FORECAST_HORIZONS),), np.float32),
                "option_available": spaces.Box(0, 1, (len(FORECAST_SCENARIOS),), np.float32),
                "sample_count": spaces.Box(0, np.inf, (1,), np.float32),
                "cash_change": spaces.Box(-np.inf, np.inf, forecast_shape, np.float32),
                "cash_in": spaces.Box(0, np.inf, forecast_shape, np.float32),
                "cash_out": spaces.Box(0, np.inf, forecast_shape, np.float32),
                "ending_cash": spaces.Box(0, np.inf, forecast_shape, np.float32),
                "minimum_cash": spaces.Box(0, np.inf, forecast_shape, np.float32),
                "debt_probability": spaces.Box(0, 1, (len(FORECAST_SCENARIOS), len(FORECAST_HORIZONS)), np.float32),
                "bankruptcy_probability": spaces.Box(0, 1, (len(FORECAST_SCENARIOS), len(FORECAST_HORIZONS)), np.float32),
            }),
            "jail_decision_forecast": spaces.Dict({
                "active": spaces.Box(0, 1, (1,), np.float32),
                "horizons": spaces.Box(0, np.inf, (len(FORECAST_HORIZONS),), np.float32),
                "option_available": spaces.Box(0, 1, (len(JAIL_OPTIONS),), np.float32),
                "sample_count": spaces.Box(0, np.inf, (1,), np.float32),
                "cash_change": spaces.Box(-np.inf, np.inf, jail_forecast_shape, np.float32),
                "cash_in": spaces.Box(0, np.inf, jail_forecast_shape, np.float32),
                "cash_out": spaces.Box(0, np.inf, jail_forecast_shape, np.float32),
                "ending_cash": spaces.Box(0, np.inf, jail_forecast_shape, np.float32),
                "minimum_cash": spaces.Box(0, np.inf, jail_forecast_shape, np.float32),
                "debt_probability": spaces.Box(0, 1, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS)), np.float32),
                "bankruptcy_probability": spaces.Box(0, 1, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS)), np.float32),
                "in_jail_probability": spaces.Box(0, 1, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS)), np.float32),
                "jail_cards_remaining": spaces.Box(0, np.inf, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS)), np.float32),
                "property_available": spaces.Box(0, 1, (len(self.tile_ids),), np.float32),
                "property_marginal_income_per_turn": spaces.Box(-np.inf, np.inf, (len(self.tile_ids),), np.float32),
                "property_marginal_yield": spaces.Box(-np.inf, np.inf, (len(self.tile_ids),), np.float32),
                "unowned_property_probability": spaces.Box(0, 1, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(self.tile_ids)), np.float32),
                "weighted_property_opportunity": spaces.Box(0, np.inf, (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(self.tile_ids)), np.float32),
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
        trade_history = self._trade_history(loop, relative_slot)
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

        dice_sides = tuple(getattr(game.dice, "sides", (6, 6)))
        if len(dice_sides) != 2 or min(dice_sides) < 2:
            raise ValueError("O mecanismo de dados precisa expor dois dados com ao menos duas faces.")
        double_probability = min(dice_sides) / (dice_sides[0] * dice_sides[1])
        dice = {
            "sides": np.asarray(dice_sides, dtype=np.float32),
            "double_probability": np.asarray([double_probability], dtype=np.float32),
        }

        jail_forecast = forecast_jail_decisions(
            loop,
            observer,
            samples=self.jail_forecast_samples,
            seed=(game.turn_counter * 1_000_003 + observer_seat * 97_409 + game.player_states[observer].jail_turns),
        ) if game.player_states[observer].is_in_jail else empty_jail_forecast(loop)
        for key in (
            "expected_cash_delta", "cash_delta_std", "expected_debt",
            "expected_rent_charged", "expected_tax_charged", "expected_go_salary",
        ):
            jail_forecast[key] /= self.money_scale

        configured_max_turns = loop.rules.max_turns
        rules_complete = {
            "starting_money": np.asarray([loop.rules.starting_money / self.money_scale], dtype=np.float32),
            "go_salary": np.asarray([loop.rules.go_salary / self.money_scale], dtype=np.float32),
            "jail_fine": np.asarray([loop.rules.jail_fine / self.money_scale], dtype=np.float32),
            "max_consecutive_doubles": np.asarray([loop.rules.max_consecutive_doubles], dtype=np.float32),
            "max_jail_attempts": np.asarray([loop.rules.max_jail_attempts], dtype=np.float32),
            "jail_position": _one_hot(loop.rules.jail_position, self.board_size),
            "go_to_jail_position": _one_hot(loop.rules.go_to_jail_position, self.board_size),
            "max_houses_in_bank": np.asarray([loop.rules.max_houses_in_bank], dtype=np.float32),
            "max_hotels_in_bank": np.asarray([loop.rules.max_hotels_in_bank], dtype=np.float32),
            "auction_min_bid": np.asarray([loop.rules.auction_min_bid / self.money_scale], dtype=np.float32),
            "income_tax_choice": np.asarray([float(loop.rules.income_tax_choice)], dtype=np.float32),
            "max_turns_enabled": np.asarray([float(configured_max_turns is not None)], dtype=np.float32),
            "max_turns": np.asarray([
                (configured_max_turns or 0) / max(1, self.max_turns)
            ], dtype=np.float32),
            "trade_enabled": np.asarray([float(loop.rules.trade_enabled)], dtype=np.float32),
            "trade_max_counteroffers": np.asarray([loop.rules.trade_max_counteroffers], dtype=np.float32),
            "debt_resolution_enabled": np.asarray([float(loop.rules.debt_resolution_enabled)], dtype=np.float32),
            "debt_negotiations_per_player": np.asarray([loop.rules.debt_negotiations_per_player], dtype=np.float32),
            "bankruptcy_to_bank": np.asarray([float(loop.rules.bankruptcy_to_bank)], dtype=np.float32),
            "auction_after_declined_purchase": np.asarray([
                float(loop.rules.auction_after_declined_purchase())
            ], dtype=np.float32),
        }

        deck_card_counts = {}
        for deck_name, deck in game.decks.items():
            draw_counts, discard_counts = deck.public_composition
            deck_card_counts[deck_name] = {
                card_id: np.asarray([draw_counts[i], discard_counts[i]], dtype=np.float32)
                for i, card_id in enumerate(self.deck_card_ids[deck_name])
            }

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
            "trade_history": trade_history,
            "globals": globals_,
            "dice": dice,
            "jail_forecast": jail_forecast,
            "rules": rules,
            "rules_complete": rules_complete,
            "deck_card_counts": deck_card_counts,
            "mortgage_transfer": mortgage_transfer,
            "stationary_economics": self._stationary_economics(loop),
            "property_decision_valuation": self._property_decision_valuation(loop, observer),
            "management_decision_valuation": self._management_decision_valuation(loop, observer),
            "management_cash_forecast": self._management_cash_forecast(loop, observer),
            "cash_flow_forecast": self._cash_flow_forecast(loop, observer),
            "jail_decision_forecast": self._jail_decision_forecast(loop, observer),
        }

    def _jail_decision_forecast(self, loop: GameLoop, observer: UUID) -> dict:
        option_count = len(JAIL_OPTIONS)
        horizon_count = len(FORECAST_HORIZONS)
        stat_count = len(FORECAST_STAT_NAMES)
        property_count = len(self.tile_ids)
        result = {
            "active": np.zeros(1, dtype=np.float32),
            "horizons": np.asarray(FORECAST_HORIZONS, dtype=np.float32),
            "option_available": np.zeros(option_count, dtype=np.float32),
            "sample_count": np.zeros(1, dtype=np.float32),
            **{
                name: np.zeros((option_count, horizon_count, stat_count), dtype=np.float32)
                for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")
            },
            **{
                name: np.zeros((option_count, horizon_count), dtype=np.float32)
                for name in ("debt_probability", "bankruptcy_probability", "in_jail_probability", "jail_cards_remaining")
            },
            "property_available": np.zeros(property_count, dtype=np.float32),
            "property_marginal_income_per_turn": np.zeros(property_count, dtype=np.float32),
            "property_marginal_yield": np.zeros(property_count, dtype=np.float32),
            "unowned_property_probability": np.zeros(
                (option_count, horizon_count, property_count), dtype=np.float32
            ),
            "weighted_property_opportunity": np.zeros(
                (option_count, horizon_count, property_count), dtype=np.float32
            ),
        }
        state = loop.game.player_states[observer]
        if (
            loop.current_decision_player_id != observer
            or loop.turn_phase != GameLoop.BEFORE_ROLL
            or not state.is_in_jail
        ):
            return result

        forecast = forecast_jail_cash_flows(loop, observer, self.config)
        result["active"][:] = forecast["active"]
        result["option_available"][:] = forecast["option_available"]
        result["sample_count"][:] = forecast["sample_count"]
        for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash"):
            result[name][:] = forecast[name] / self.money_scale
        for name in ("debt_probability", "bankruptcy_probability", "in_jail_probability", "jail_cards_remaining"):
            result[name][:] = forecast[name]

        opponents = max(0, len(loop.game.active_player_ids) - 1)
        marginal_income = np.zeros(property_count, dtype=np.float32)
        for tile_id, property_index in self.property_index.items():
            if loop.game.property_states[tile_id].owner_id is not None:
                continue
            tile = loop.game.board.get_tile_by_id(tile_id)
            group = loop.game.board.group_members(tile.group)
            owned_ids = tuple(
                item.tile_id for item in group
                if loop.game.property_states[item.tile_id].owner_id == observer
            )
            mortgaged_ids = tuple(
                owned_id for owned_id in owned_ids
                if loop.game.property_states[owned_id].is_mortgaged
            )
            valuation = evaluate_marginal_purchase(
                self.config,
                self.stationary_analysis,
                tile_id,
                owned_ids,
                mortgaged_tile_ids=mortgaged_ids,
                number_of_opponents=opponents,
            )
            result["property_available"][property_index] = 1.0
            marginal_income[property_index] = valuation.marginal_income_per_turn
            result["property_marginal_income_per_turn"][property_index] = (
                valuation.marginal_income_per_turn / self.money_scale
            )
            result["property_marginal_yield"][property_index] = (
                valuation.marginal_yield_per_turn or 0.0
            )

        for unowned_index, tile_id in enumerate(forecast["unowned_property_ids"]):
            property_index = self.property_index[int(tile_id)]
            probability = forecast["unowned_property_probability"][:, :, unowned_index]
            result["unowned_property_probability"][:, :, property_index] = probability
            result["weighted_property_opportunity"][:, :, property_index] = (
                probability * marginal_income[property_index] / self.money_scale
                if marginal_income[property_index] > 0
                else 0.0
            )
        return result

    def _property_decision_valuation(self, loop: GameLoop, observer: UUID) -> dict:
        result = {
            "active": np.zeros(1, dtype=np.float32),
            "candidate_property": np.zeros(len(self.tile_ids), dtype=np.float32),
            "owned_group_properties": np.zeros(len(self.tile_ids), dtype=np.float32),
            "group_complete_after_purchase": np.zeros(1, dtype=np.float32),
            "reference_price": np.zeros(1, dtype=np.float32),
            "next_auction_bid": np.zeros(1, dtype=np.float32),
            "income_before_per_turn": np.zeros(1, dtype=np.float32),
            "income_after_per_turn": np.zeros(1, dtype=np.float32),
            "marginal_income_per_turn": np.zeros(1, dtype=np.float32),
            "marginal_yield_at_reference_price": np.zeros(1, dtype=np.float32),
            "payback_turns": np.zeros(1, dtype=np.float32),
            "payback_available": np.zeros(1, dtype=np.float32),
        }
        candidate_id = loop.pending_tile_id
        if (
            candidate_id not in self.property_index
            or loop.current_decision_player_id != observer
            or loop.turn_phase not in (GameLoop.AWAITING_PROPERTY_DECISION, GameLoop.AUCTION)
        ):
            return result

        candidate = loop.game.board.get_tile_by_id(candidate_id)
        group_tiles = loop.game.board.group_members(candidate.group)
        owned_ids = tuple(
            tile.tile_id for tile in group_tiles
            if loop.game.property_states[tile.tile_id].owner_id == observer
        )
        mortgaged_ids = tuple(
            tile_id for tile_id in owned_ids
            if loop.game.property_states[tile_id].is_mortgaged
        )
        opponents = len(loop.game.active_player_ids) - 1
        valuation = evaluate_marginal_purchase(
            self.config,
            self.stationary_analysis,
            candidate_id,
            owned_ids,
            mortgaged_tile_ids=mortgaged_ids,
            number_of_opponents=max(0, opponents),
            price=candidate.price,
        )
        for tile_id in owned_ids:
            result["owned_group_properties"][self.property_index[tile_id]] = 1.0
        result["candidate_property"][self.property_index[candidate_id]] = 1.0
        result["active"][0] = 1.0
        result["group_complete_after_purchase"][0] = float(
            set(valuation.owned_after) == {tile.tile_id for tile in group_tiles}
        )
        result["reference_price"][0] = valuation.purchase_price / self.money_scale
        if loop.turn_phase == GameLoop.AUCTION:
            result["next_auction_bid"][0] = max(
                loop.rules.auction_min_bid,
                loop.auction_highest_bid + 1,
            ) / self.money_scale
        income_before = valuation.income_before_per_turn
        income_after = valuation.income_after_per_turn
        marginal_income = valuation.marginal_income_per_turn
        result["income_before_per_turn"][0] = income_before / self.money_scale
        result["income_after_per_turn"][0] = income_after / self.money_scale
        result["marginal_income_per_turn"][0] = marginal_income / self.money_scale
        result["marginal_yield_at_reference_price"][0] = (
            valuation.marginal_yield_per_turn or 0.0
        )
        if marginal_income > 0:
            result["payback_turns"][0] = valuation.purchase_price / marginal_income
            result["payback_available"][0] = 1.0
        return result

    def _cash_flow_forecast(self, loop: GameLoop, observer: UUID) -> dict:
        if (
            loop.turn_phase != GameLoop.AWAITING_PROPERTY_DECISION
            or loop.current_decision_player_id != observer
        ):
            return empty_cash_flow_forecast()
        forecast = forecast_property_cash_flows(
            loop,
            observer,
            self.config,
            horizons=FORECAST_HORIZONS,
            samples=DEFAULT_FORECAST_SAMPLES,
        )
        return forecast.as_dict(self.money_scale)

    def _management_decision_valuation(self, loop: GameLoop, observer: UUID) -> dict:
        shape = (len(MANAGEMENT_ACTION_TYPES), len(self.tile_ids))
        result = {
            "available": np.zeros(shape, dtype=np.float32),
            "cash_delta": np.zeros(shape, dtype=np.float32),
            "income_before_per_turn": np.zeros(shape, dtype=np.float32),
            "income_after_per_turn": np.zeros(shape, dtype=np.float32),
            "marginal_income_per_turn": np.zeros(shape, dtype=np.float32),
            "marginal_yield_per_turn": np.zeros(shape, dtype=np.float32),
        }
        if loop.current_decision_player_id != observer:
            return result
        action_index = {kind: index for index, kind in enumerate(MANAGEMENT_ACTION_TYPES)}
        candidates = [
            action for action in loop.legal_actions()
            if action.action_type in action_index and "tile_id" in action.payload
        ]
        for action in candidates:
            kind_index = action_index[action.action_type]
            tile_id = action.payload["tile_id"]
            tile_index = self.property_index[tile_id]
            tile = loop.game.board.get_tile_by_id(tile_id)
            group_tiles = loop.game.board.group_members(tile.group)
            owned_ids = tuple(
                item.tile_id for item in group_tiles
                if loop.game.property_states[item.tile_id].owner_id == observer
            )
            levels_before = {
                item.tile_id: loop.game.property_states[item.tile_id].houses
                for item in group_tiles if item.tile_id in owned_ids
            }
            mortgaged_before = {
                item.tile_id for item in group_tiles
                if item.tile_id in owned_ids and loop.game.property_states[item.tile_id].is_mortgaged
            }
            levels_after = dict(levels_before)
            mortgaged_after = set(mortgaged_before)
            state = loop.game.property_states[tile_id]
            if action.action_type == ActionType.BUILD_HOUSE:
                levels_after[tile_id] += 1
                cost = (tile.hotel_cost or tile.house_cost) if state.houses == HOTEL_LEVEL - 1 else tile.house_cost
                cash_delta = -cost
            elif action.action_type == ActionType.SELL_HOUSE:
                levels_after[tile_id] -= 1
                refund = (tile.hotel_cost or tile.house_cost) // 2 if state.houses == HOTEL_LEVEL else tile.house_cost // 2
                cash_delta = refund
            elif action.action_type == ActionType.MORTGAGE:
                mortgaged_after.add(tile_id)
                cash_delta = tile.mortgage_value
            else:
                mortgaged_after.discard(tile_id)
                cash_delta = -loop.unmortgage_cost(tile)

            opponents = max(0, len(loop.game.active_player_ids) - 1)
            before = estimate_group_state_income_per_turn(
                self.config, self.stationary_analysis, owned_ids, levels_before,
                mortgaged_before, number_of_opponents=opponents,
            )
            after = estimate_group_state_income_per_turn(
                self.config, self.stationary_analysis, owned_ids, levels_after,
                mortgaged_after, number_of_opponents=opponents,
            )
            delta_income = after - before
            result["available"][kind_index, tile_index] = 1.0
            result["cash_delta"][kind_index, tile_index] = cash_delta / self.money_scale
            result["income_before_per_turn"][kind_index, tile_index] = before / self.money_scale
            result["income_after_per_turn"][kind_index, tile_index] = after / self.money_scale
            result["marginal_income_per_turn"][kind_index, tile_index] = delta_income / self.money_scale
            if cash_delta != 0:
                result["marginal_yield_per_turn"][kind_index, tile_index] = (
                    delta_income / abs(cash_delta)
                )
        return result

    def _management_cash_forecast(self, loop: GameLoop, observer: UUID) -> dict:
        action_count = len(MANAGEMENT_ACTION_TYPES)
        property_count = len(self.tile_ids)
        horizon_count = len(FORECAST_HORIZONS)
        stat_count = len(FORECAST_STAT_NAMES)
        result = {
            "active": np.zeros(1, dtype=np.float32),
            "sample_count": np.zeros(1, dtype=np.float32),
            "available": np.zeros((action_count, property_count), dtype=np.float32),
            **{
                name: np.zeros(
                    (action_count, property_count, 2, horizon_count, stat_count),
                    dtype=np.float32,
                )
                for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")
            },
            "debt_probability": np.zeros(
                (action_count, property_count, 2, horizon_count), dtype=np.float32
            ),
            "bankruptcy_probability": np.zeros(
                (action_count, property_count, 2, horizon_count), dtype=np.float32
            ),
        }
        if loop.current_decision_player_id != observer:
            return result
        forecasts = forecast_management_cash_flows(
            loop,
            observer,
            self.config,
            samples=DEFAULT_MANAGEMENT_FORECAST_SAMPLES,
        )
        action_index = {kind: index for index, kind in enumerate(MANAGEMENT_ACTION_TYPES)}
        for (kind, tile_id), forecast in forecasts.items():
            action_slot = action_index[kind]
            property_slot = self.property_index[tile_id]
            result["available"][action_slot, property_slot] = 1.0
            data = forecast.as_dict(self.money_scale)
            for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash"):
                result[name][action_slot, property_slot] = data[name]
            result["debt_probability"][action_slot, property_slot] = data["debt_probability"]
            result["bankruptcy_probability"][action_slot, property_slot] = data["bankruptcy_probability"]
            result["active"][0] = 1.0
            result["sample_count"][0] = forecast.sample_count
        return result

    def _stationary_economics(self, loop: GameLoop) -> dict:
        active_opponents = max(0, len(loop.game.active_player_ids) - 1)
        return {
            "property_arrivals_per_roll": np.asarray([
                self.stationary_analysis.traffic_for(tile_id).arrivals_per_roll
                for tile_id in self.tile_ids
            ], dtype=np.float32),
            "utility_dice_total_per_roll": np.asarray([
                self.stationary_analysis.traffic_for(tile_id).utility_dice_total_per_roll
                for tile_id in self.tile_ids
            ], dtype=np.float32),
            "railroad_rent_multiplier_per_roll": np.asarray([
                self.stationary_analysis.traffic_for(tile_id).railroad_rent_multiplier_per_roll
                for tile_id in self.tile_ids
            ], dtype=np.float32),
            "expected_rolls_per_turn": np.asarray(
                [self.stationary_analysis.expected_rolls_per_turn], dtype=np.float32
            ),
            "portfolio_membership": self.portfolio_membership.copy(),
            "portfolio_groups": self.portfolio_groups.copy(),
            "portfolio_development": self.portfolio_development.copy(),
            "portfolio_investment": self.portfolio_investment.copy(),
            "portfolio_income_per_turn": (
                self.portfolio_income * active_opponents
            ).astype(np.float32),
            "portfolio_gross_yield_per_turn": (
                self.portfolio_yield * active_opponents
            ).astype(np.float32),
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

    def _trade_history(self, loop, relative_slot):
        values = {
            "active": np.zeros(self.max_trade_history, dtype=np.float32),
            "participants": np.zeros((self.max_trade_history, 2, self.max_players), dtype=np.float32),
            "money": np.zeros((self.max_trade_history, 2), dtype=np.float32),
            "properties": np.zeros((self.max_trade_history, 2, len(self.tile_ids)), dtype=np.float32),
            "jail_cards": np.zeros((self.max_trade_history, 2, len(self.jail_card_ids)), dtype=np.float32),
            "counteroffer_number": np.zeros(self.max_trade_history, dtype=np.float32),
            "response": np.zeros((self.max_trade_history, 3), dtype=np.float32),
            "responder": np.zeros((self.max_trade_history, self.max_players), dtype=np.float32),
            "age": np.zeros(self.max_trade_history, dtype=np.float32),
            "participant_cash": np.zeros((self.max_trade_history, 2), dtype=np.float32),
            "participant_position": np.zeros((self.max_trade_history, 2, self.board_size), dtype=np.float32),
            "participant_in_jail": np.zeros((self.max_trade_history, 2), dtype=np.float32),
            "participant_jail_cards": np.zeros((self.max_trade_history, 2, len(self.jail_card_ids)), dtype=np.float32),
            "participant_net_worth": np.zeros((self.max_trade_history, 2), dtype=np.float32),
            "participant_debt": np.zeros((self.max_trade_history, 2), dtype=np.float32),
            "property_owner": np.zeros((self.max_trade_history, len(self.tile_ids), self.max_players + 1), dtype=np.float32),
            "property_building_level": np.zeros((self.max_trade_history, len(self.tile_ids)), dtype=np.float32),
            "property_mortgaged": np.zeros((self.max_trade_history, len(self.tile_ids)), dtype=np.float32),
            "phase": np.zeros((self.max_trade_history, len(PHASES)), dtype=np.float32),
            "is_debt_trade": np.zeros(self.max_trade_history, dtype=np.float32),
        }
        offers = loop.game.trade_history[-self.max_trade_history:]
        decisions = {
            (item.negotiation_id, item.counteroffer_number): item
            for item in loop.game.trade_decision_history
        }
        contexts = {
            (item.negotiation_id, item.counteroffer_number): item
            for item in loop.game.trade_context_history
        }
        outcome_index = {"ACCEPTED": 0, "REJECTED": 1, "COUNTERED": 2}
        for row, offer in enumerate(offers):
            values["active"][row] = 1.0
            values["participants"][row, 0, relative_slot[offer.proposer_id]] = 1.0
            values["participants"][row, 1, relative_slot[offer.recipient_id]] = 1.0
            values["money"][row] = (
                offer.money_offered / self.money_scale,
                offer.money_requested / self.money_scale,
            )
            for side, property_ids in enumerate((offer.properties_offered, offer.properties_requested)):
                for tile_id in property_ids:
                    if tile_id in self.property_index:
                        values["properties"][row, side, self.property_index[tile_id]] = 1.0
            for side, card_ids in enumerate((offer.jail_cards_offered, offer.jail_cards_requested)):
                for card_id in card_ids:
                    if card_id in self.jail_card_index:
                        values["jail_cards"][row, side, self.jail_card_index[card_id]] = 1.0
            values["counteroffer_number"][row] = (
                offer.counteroffer_number / max(1, loop.rules.trade_max_counteroffers)
            )
            decision = decisions.get((offer.negotiation_id, offer.counteroffer_number))
            if decision is not None:
                values["response"][row, outcome_index[decision.response]] = 1.0
                values["responder"][row, relative_slot[decision.responder_id]] = 1.0
                values["age"][row] = min(
                    1.0,
                    max(0, loop.game.turn_counter - decision.turn) / max(1, self.max_turns),
                )
            context = contexts.get((offer.negotiation_id, offer.counteroffer_number))
            if context is not None:
                values["phase"][row, PHASES.index(context.phase)] = 1.0
                values["is_debt_trade"][row] = float(context.is_debt_trade)
                context_players = {item.player_id: item for item in context.players}
                for side, participant_id in enumerate((offer.proposer_id, offer.recipient_id)):
                    player_context = context_players[participant_id]
                    values["participant_cash"][row, side] = player_context.money / self.money_scale
                    values["participant_position"][row, side, player_context.position] = 1.0
                    values["participant_in_jail"][row, side] = float(player_context.is_in_jail)
                    for card_id in player_context.jail_cards:
                        if card_id in self.jail_card_index:
                            values["participant_jail_cards"][row, side, self.jail_card_index[card_id]] = 1.0
                    values["participant_net_worth"][row, side] = player_context.net_worth / self.money_scale
                    values["participant_debt"][row, side] = player_context.amount_owed / self.money_scale
                for property_context in context.properties:
                    if property_context.tile_id not in self.property_index:
                        continue
                    property_index = self.property_index[property_context.tile_id]
                    owner_index = (
                        0 if property_context.owner_id is None
                        else 1 + relative_slot[property_context.owner_id]
                    )
                    values["property_owner"][row, property_index, owner_index] = 1.0
                    values["property_building_level"][row, property_index] = (
                        property_context.houses / HOTEL_LEVEL
                    )
                    values["property_mortgaged"][row, property_index] = float(
                        property_context.is_mortgaged
                    )
        return values

    def _board_features(self) -> np.ndarray:
        rows = np.zeros((
            self.board_size, 19 + len(self.group_labels) + len(self.special_labels)
        ), dtype=np.float32)
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
                features[18 + len(self.group_labels) + len(self.special_labels)] = (
                    (tile.percent_option or 0) / 100.0
                )
            elif isinstance(tile, SpecialTile):
                features[4] = 1.0
                features[18 + len(self.group_labels) + self.special_index[tile.tile_type]] = 1.0
        return rows
