import json

from app.application.game_setup import create_game
from app.domain.analytics.decision_context import evaluate_candidate
from app.domain.analytics.decision_features import DecisionFeaturesEncoder
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import Debt


def make_encoder(loop):
    return DecisionFeaturesEncoder(
        loop,
        max_players=12,
        max_alternatives=16,
        horizons=(1,),
    )


def assert_specialized_context_encodes(context, loop, expected_section, expected_rows):
    assert getattr(context, expected_section) is not None
    assert json.loads(json.dumps(context.to_dict()))["schema_version"] == 1

    encoder = make_encoder(loop)
    features = encoder.encode(context)
    assert int(features["alternative_mask"].sum()) == expected_rows
    assert encoder.observation_space.contains(features)


def test_jail_analysis_encodes_all_available_exit_choices(config):
    loop = create_game(["P0", "P1"], config, seed=61)
    player_id = loop.current_decision_player_id
    loop.game.player_states[player_id].is_in_jail = True
    loop.game.player_states[player_id].jail_turns = 0
    loop.game.set_position(player_id, loop.rules.jail_position)
    loop.turn_phase = GameLoop.BEFORE_ROLL

    context = evaluate_candidate(
        loop,
        config,
        player_id,
        GameAction(ActionType.ROLL_DICE),
        horizons=(1,),
        valuation_horizons=(1,),
        simulations=1,
        seed=61,
    )

    assert_specialized_context_encodes(context, loop, "jail_analysis", 2)


def test_auction_analysis_encodes_candidate_bid_and_withdrawal(config):
    loop = create_game(["P0", "P1"], config, seed=62)
    tile = loop.game.board.property_tiles[0]
    loop._start_auction(tile.tile_id)
    player_id = loop.current_decision_player_id
    first_bid = next(
        action for action in loop.legal_actions()
        if action.action_type == ActionType.AUCTION_BID
    )

    context = evaluate_candidate(
        loop,
        config,
        player_id,
        first_bid,
        auction_bid_candidates=(first_bid.payload["amount"],),
        horizons=(1,),
        simulations=1,
        seed=62,
    )

    assert_specialized_context_encodes(context, loop, "auction_analysis", 2)


def test_trade_analysis_encodes_accept_and_reject_scenarios(config):
    loop = create_game(["P0", "P1"], config, seed=63)
    proposer = loop.current_decision_player_id
    recipient = next(player for player in loop.game.active_player_ids if player != proposer)
    proposal = GameAction(
        ActionType.PROPOSE_TRADE,
        {"target_player_id": recipient, "money_offered": 10},
    )

    context = evaluate_candidate(
        loop,
        config,
        proposer,
        proposal,
        horizons=(1,),
        simulations=1,
        seed=63,
    )

    assert_specialized_context_encodes(context, loop, "trade_analysis", 2)


def test_debt_analysis_encodes_resolution_action_and_affected_players(config):
    loop = create_game(["P0", "P1"], config, seed=64)
    debtor = loop.current_decision_player_id
    creditor = next(player for player in loop.game.active_player_ids if player != debtor)
    loop.game.player_states[debtor].money = 0
    loop.game.debts.append(Debt(debtor, creditor, 100))
    loop._settle()
    action = next(
        action for action in loop.legal_actions()
        if action.action_type == ActionType.END_DEBT_NEGOTIATIONS
    )

    context = evaluate_candidate(
        loop,
        config,
        debtor,
        action,
        horizons=(1,),
        simulations=1,
        seed=64,
    )

    assert_specialized_context_encodes(context, loop, "debt_analysis", 1)
    impacts = context.debt_analysis.scenarios[0].impacts
    assert {impact.player_id for impact in impacts} == {debtor, creditor}
