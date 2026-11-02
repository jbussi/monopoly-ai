import numpy as np
import pytest

from app.application.game_setup import create_game
from app.domain.analytics.economics.stationary_board import (
    calculate_stationary_board_analysis,
    evaluate_marginal_purchase,
)
from app.domain.game.actions import ActionType
from app.domain.game.game_loop import GameLoop
from app.envs.cash_flow_forecast import (
    FORECAST_HORIZONS,
    FORECAST_STAT_NAMES,
    JAIL_OPTIONS,
    MANAGEMENT_ACTION_TYPES,
    forecast_jail_decisions,
    forecast_management_cash_flows,
    forecast_property_cash_flows,
)
from app.envs.action_space_v2 import StructuredActionSpace
from app.envs.observation_v2 import ObservationEncoderV2


def test_stationary_board_analysis_has_normalized_distribution_and_property_traffic(config):
    analysis = calculate_stationary_board_analysis(config)

    assert len(analysis.position_probabilities) == len(config.board)
    assert sum(analysis.position_probabilities) == pytest.approx(1.0)
    assert sum(analysis.probabilities) == pytest.approx(1.0)
    assert analysis.expected_rolls_per_turn > 1.0
    assert len(analysis.property_traffic) == sum(
        space.type in ("property", "railroad", "utility") for space in config.board
    )
    assert all(traffic.arrivals_per_roll >= 0 for traffic in analysis.property_traffic)
    assert sum(item.arrivals_per_roll for item in analysis.property_traffic) <= 1.0 + 1e-8


def test_marginal_purchase_values_group_completion_and_auction_price(config):
    analysis = calculate_stationary_board_analysis(config)
    candidate_id = 3
    prior_owned = (1,)
    candidate = next(space for space in config.board if space.position == candidate_id)

    at_list_price = evaluate_marginal_purchase(
        config, analysis, candidate_id, prior_owned, number_of_opponents=3
    )
    at_higher_bid = evaluate_marginal_purchase(
        config,
        analysis,
        candidate_id,
        prior_owned,
        price=candidate.price * 2,
        number_of_opponents=3,
    )

    assert set(at_list_price.owned_after) == {1, 3}
    assert at_list_price.marginal_income_per_turn > 0
    assert at_list_price.income_after_per_turn > at_list_price.income_before_per_turn
    assert at_list_price.marginal_yield_per_turn == pytest.approx(
        2 * at_higher_bid.marginal_yield_per_turn
    )


def test_property_purchase_forecast_compares_buy_and_decline(config):
    loop = create_game(["A", "B"], config=config, seed=101, record_events=False)
    player_id = loop.current_decision_player_id
    candidate = next(tile for tile in loop.game.board.property_tiles if tile.tile_id == 1)
    loop._pending_tile_id = candidate.tile_id
    loop.turn_phase = GameLoop.AWAITING_PROPERTY_DECISION

    forecast = forecast_property_cash_flows(
        loop, player_id, config, horizons=(1, 3, 5), samples=3
    )

    assert forecast.option_available == (True, True)
    assert forecast.cash_change.shape == (2, 3, len(FORECAST_STAT_NAMES))
    assert forecast.cash_out[0, :, 0].min() >= candidate.price
    assert np.all((forecast.debt_probability >= 0) & (forecast.debt_probability <= 1))


def test_management_forecast_has_action_and_defer_branches(config):
    loop = create_game(["A", "B"], config=config, seed=102, record_events=False)
    player_id = loop.current_decision_player_id
    tile = next(tile for tile in loop.game.board.property_tiles if tile.tile_id == 1)
    loop.game.property_states[tile.tile_id].owner_id = player_id
    loop.turn_phase = GameLoop.BEFORE_ROLL

    forecasts = forecast_management_cash_flows(loop, player_id, config, samples=2)

    forecast = forecasts[(ActionType.MORTGAGE, tile.tile_id)]
    assert forecast.option_available == (True, True)
    assert forecast.cash_change.shape == (2, len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES))
    assert forecast.cash_change[0, 0, 0] != forecast.cash_change[1, 0, 0]


def test_jail_forecast_compares_exit_options_and_unowned_property_opportunities(config):
    loop = create_game(["A", "B"], config=config, seed=103, record_events=False)
    player_id = loop.current_decision_player_id
    player_state = loop.game.player_states[player_id]
    player_state.is_in_jail = True
    loop.game.set_position(player_id, loop.rules.jail_position)
    loop.turn_phase = GameLoop.BEFORE_ROLL
    deck = loop.game.decks["chance"]
    jail_card = next(card for card in deck._cards if card.action == "get_out_of_jail")
    deck._cards.remove(jail_card)
    player_state.jail_cards.append(("chance", jail_card))

    forecast = forecast_jail_decisions(loop, player_id, config, samples=3)

    assert forecast["option_available"].tolist() == [1.0, 1.0, 1.0]
    assert forecast["cash_change"].shape == (
        len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES)
    )
    assert forecast["unowned_property_probability"].shape == (
        len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(forecast["unowned_property_ids"])
    )
    assert np.all((forecast["unowned_property_probability"] >= 0)
                  & (forecast["unowned_property_probability"] <= 1))
    assert np.all((forecast["in_jail_probability"] >= 0)
                  & (forecast["in_jail_probability"] <= 1))
    assert np.all(forecast["unowned_property_probability"][:, 0, :] <=
                  forecast["unowned_property_probability"][:, 1, :] + 1e-7)


def test_observation_schema_contains_property_management_and_jail_decisions(config):
    loop = create_game(["A", "B"], config=config, seed=104, record_events=False)
    player_id = loop.current_decision_player_id
    action_space = StructuredActionSpace(loop, max_players=12)
    encoder = ObservationEncoderV2(
        loop,
        max_players=12,
        jail_card_ids=action_space.jail_card_ids,
        config=config,
    )
    tile = next(tile for tile in loop.game.board.property_tiles if tile.tile_id == 1)

    loop._pending_tile_id = tile.tile_id
    loop.turn_phase = GameLoop.AWAITING_PROPERTY_DECISION
    purchase_observation = encoder.encode(loop, player_id)
    assert encoder.observation_space.contains(purchase_observation)
    assert purchase_observation["property_decision_valuation"]["active"][0] == 1
    assert purchase_observation["cash_flow_forecast"]["active"][0] == 1

    loop._pending_tile_id = None
    loop.game.property_states[tile.tile_id].owner_id = player_id
    loop.turn_phase = GameLoop.BEFORE_ROLL
    management_observation = encoder.encode(loop, player_id)
    assert encoder.observation_space.contains(management_observation)
    mortgage_index = next(
        index for index, kind in enumerate(MANAGEMENT_ACTION_TYPES)
        if kind == ActionType.MORTGAGE
    )
    assert management_observation["management_decision_valuation"]["available"][
        mortgage_index, action_space.property_index[tile.tile_id]
    ] == 1

    state = loop.game.player_states[player_id]
    state.is_in_jail = True
    loop.game.set_position(player_id, loop.rules.jail_position)
    deck = loop.game.decks["chance"]
    jail_card = next(card for card in deck._cards if card.action == "get_out_of_jail")
    deck._cards.remove(jail_card)
    state.jail_cards.append(("chance", jail_card))
    jail_observation = encoder.encode(loop, player_id)
    assert encoder.observation_space.contains(jail_observation)
    assert jail_observation["jail_decision_forecast"]["active"][0] == 1
    assert jail_observation["jail_decision_forecast"]["property_available"].sum() > 0
