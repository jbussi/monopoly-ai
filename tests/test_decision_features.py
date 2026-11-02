import json

import numpy as np
import pytest

from app.domain.agents import FeaturePolicy
from app.domain.analytics.decision_context import DecisionContext
from app.domain.analytics.decision_features import (
    DecisionFeaturesEncoder,
    EncodedDecisionContext,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2
from app.envs.pettingzoo_env_v2 import raw_env


def enter_purchase_decision(loop):
    tile = loop.game.board.property_tiles[0]
    loop.turn_phase = GameLoop.AWAITING_PROPERTY_DECISION
    loop._pending_tile_id = tile.tile_id
    return tile


def test_gym_candidate_features_have_fixed_shape_and_serializable_context(config):
    env = MonopolyGymEnvV2(
        num_players=2,
        max_players=12,
        learner_seat=0,
        config=config,
        decision_horizons=(1,),
        max_decision_alternatives=4,
    )
    env.reset(seed=51)
    tile = enter_purchase_decision(env.loop)
    player_id = env.learner_id
    starting_cash = env.loop.game.player_states[player_id].money

    result = env.evaluate_candidate_features(
        GameAction(ActionType.BUY_PROPERTY),
        baseline_action=GameAction(ActionType.DECLINE_PROPERTY),
        horizons=(1,),
        valuation_horizons=(1,),
        simulations=1,
        seed=51,
    )

    assert isinstance(result, EncodedDecisionContext)
    assert result.context.candidate_action.action_type == ActionType.BUY_PROPERTY
    assert result.features["alternative_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert result.features["actions"].shape[0] == 4
    assert result.features["cash_forecasts"].shape == (4, 12, 1, 11)
    assert result.features["horizon_mask"][:2, 0].tolist() == [1.0, 1.0]
    assert env.decision_features_space.contains(result.features)
    assert json.loads(json.dumps(result.context.to_dict()))["schema_version"] == 1
    assert env.loop.turn_phase == GameLoop.AWAITING_PROPERTY_DECISION
    assert env.loop.game.player_states[player_id].money == starting_cash
    assert env.loop.game.property_states[tile.tile_id].owner_id is None


def test_pettingzoo_candidate_features_use_the_acting_agent(config):
    env = raw_env(
        num_players=2,
        max_players=12,
        config=config,
        decision_horizons=(1,),
        max_decision_alternatives=4,
    )
    env.reset(seed=52)
    enter_purchase_decision(env.loop)
    agent = env._agent(env.loop.current_decision_player_id)

    result = env.evaluate_candidate_features(
        agent,
        GameAction(ActionType.DECLINE_PROPERTY),
        baseline_action=GameAction(ActionType.BUY_PROPERTY),
        horizons=(1,),
        valuation_horizons=(1,),
        simulations=1,
        seed=52,
    )

    assert result.context.player_id == env._player_id(agent)
    assert result.features["alternative_mask"][:2].tolist() == [1.0, 1.0]
    assert env.decision_features_space.contains(result.features)


def test_decision_feature_encoder_rejects_silent_alternative_truncation(config):
    loop = MonopolyGymEnvV2(
        num_players=2,
        max_players=12,
        learner_seat=0,
        config=config,
        decision_horizons=(1,),
    )
    loop.reset(seed=53)
    enter_purchase_decision(loop.loop)
    context = DecisionContext(
        player_id=loop.loop.current_decision_player_id,
        phase=loop.loop.turn_phase,
        candidate_action=GameAction(ActionType.BUY_PROPERTY),
        baseline_action=GameAction(ActionType.DECLINE_PROPERTY),
        seed=53,
        simulations=1,
    )
    encoder = DecisionFeaturesEncoder(loop.loop, max_alternatives=1, horizons=(1,))

    with pytest.raises(ValueError, match="alternativas"):
        encoder.encode(context)


def test_feature_policy_returns_only_a_candidate_selected_from_legal_actions(make_game):
    loop = make_game()
    enter_purchase_decision(loop)
    actor = loop.current_decision_player_id

    def evaluator(current_loop, action):
        context = DecisionContext(
            player_id=actor,
            phase=current_loop.turn_phase,
            candidate_action=action,
            baseline_action=None,
            seed=0,
            simulations=1,
        )
        return EncodedDecisionContext(
            context=context,
            features={
                "candidate": np.asarray(
                    [float(action.action_type == ActionType.BUY_PROPERTY)],
                    dtype=np.float32,
                )
            },
        )

    legal_candidates = (
        GameAction(ActionType.BUY_PROPERTY),
        GameAction(ActionType.DECLINE_PROPERTY),
    )
    policy = FeaturePolicy(
        evaluator=evaluator,
        selector=lambda _loop, candidates: max(
            candidates,
            key=lambda candidate: candidate.decision.features["candidate"][0],
        ).action,
        candidate_provider=lambda _loop: legal_candidates,
    )

    assert policy(loop) == GameAction(ActionType.BUY_PROPERTY)


def test_feature_policy_rejects_selector_action_outside_evaluated_candidates(make_game):
    loop = make_game()
    enter_purchase_decision(loop)
    actor = loop.current_decision_player_id

    def evaluator(current_loop, action):
        context = DecisionContext(
            player_id=actor,
            phase=current_loop.turn_phase,
            candidate_action=action,
            baseline_action=None,
            seed=0,
            simulations=1,
        )
        return EncodedDecisionContext(context, {"candidate": np.zeros(1, dtype=np.float32)})

    candidates = (GameAction(ActionType.DECLINE_PROPERTY),)
    policy = FeaturePolicy(
        evaluator=evaluator,
        selector=lambda _loop, _evaluated: GameAction(ActionType.BUY_PROPERTY),
        candidate_provider=lambda _loop: candidates,
    )

    with pytest.raises(ValueError, match="não estava entre as candidatas"):
        policy(loop)
