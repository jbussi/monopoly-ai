import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test, seed_test

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import Debt
from app.envs.action_space_v2 import ACTION_TYPE_TO_INDEX
from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2
from app.envs.pettingzoo_env_v2 import raw_env
from app.envs.pettingzoo_env_v2 import env as pettingzoo_env


def make_action(action_map, kind, **fields):
    action = action_map.empty_action()
    action["action_type"] = ACTION_TYPE_TO_INDEX[kind]
    for key, value in fields.items():
        action[key] = value
    return action


def test_gym_v2_observation_and_action_contract_support_twelve_players():
    env = MonopolyGymEnvV2(num_players=12, max_players=12, learner_seat=0)
    observation, _ = env.reset(seed=17)

    assert env.action_space.contains(env.action_map.empty_action())
    assert env.observation_space.contains(observation)
    assert observation["players"]["active"].shape == (12,)
    assert observation["properties"]["owner"].shape == (28, 13)
    assert observation["action_mask"]["action_type"].shape == (21,)
    assert observation["action_mask"]["action_type"][ACTION_TYPE_TO_INDEX[ActionType.ROLL_DICE]] == 1


def test_gym_v2_official_checker():
    check_env(
        MonopolyGymEnvV2(num_players=2, max_players=12, on_illegal="terminate"),
        skip_render_check=True,
    )


def test_pettingzoo_v2_official_api_checker():
    api_test(pettingzoo_env(num_players=2, max_players=12), num_cycles=100)


def test_pettingzoo_v2_seed_checker():
    seed_test(lambda: pettingzoo_env(num_players=2, max_players=12), num_cycles=40)


def test_pettingzoo_v2_twelve_player_full_episode():
    env = pettingzoo_env(num_players=12, max_players=12, max_turns=120)
    env.reset(seed=1200)
    for agent in env.agent_iter():
        observation, _, terminated, truncated, _ = env.last()
        action = None if terminated or truncated else env.action_space(agent).sample(
            observation["action_mask"]
        )
        env.step(action)

    assert env.unwrapped.loop.game.game_over
    assert env.unwrapped.loop.game.winner_id is not None or env.unwrapped.loop.game.truncated


def test_pettingzoo_v2_trade_can_be_countered_and_accepted():
    env = raw_env(num_players=2, max_players=12)
    env.reset(seed=4)
    proposer, recipient = env.loop.game.turn_order
    tile = env.loop.game.board.property_tiles[0]
    env.loop.game.property_states[tile.tile_id].owner_id = recipient

    initial_observation = env.observe(env.agent_selection)
    property_index = env.action_map.property_index[tile.tile_id]
    assert initial_observation["action_mask"]["action_type"][ACTION_TYPE_TO_INDEX[ActionType.PROPOSE_TRADE]] == 1
    assert initial_observation["action_mask"]["target_player"][1] == 1
    assert initial_observation["action_mask"]["requested_properties"][property_index] == 1

    offer = make_action(
        env.action_map,
        ActionType.PROPOSE_TRADE,
        target_player=1,
        money_offered=np.asarray([20], dtype=np.int32),
    )
    offer["requested_properties"][property_index] = 1
    assert env.observation_space(env.agent_selection).contains(env.observe(env.agent_selection))
    env.step(offer)
    assert env.loop.current_decision_player_id == recipient

    counter = make_action(
        env.action_map,
        ActionType.COUNTER_TRADE,
        money_offered=np.asarray([50], dtype=np.int32),
    )
    counter["offered_properties"][env.action_map.property_index[tile.tile_id]] = 1
    env.step(counter)
    assert env.loop.current_decision_player_id == proposer
    env.step(make_action(env.action_map, ActionType.ACCEPT_TRADE))

    assert env.loop.game.property_states[tile.tile_id].owner_id == proposer
    assert env.loop.game.player_states[proposer].money == 1550
    assert env.loop.game.player_states[recipient].money == 1450


def test_gym_v2_debt_trade_response_can_pay_and_close_debt():
    class AcceptTrade:
        def __call__(self, loop):
            return GameAction(ActionType.ACCEPT_TRADE)

    env = MonopolyGymEnvV2(
        num_players=2,
        max_players=12,
        learner_seat=0,
        opponents=[AcceptTrade()],
    )
    env.reset(seed=9)
    debtor, creditor = env.loop.game.turn_order
    env.loop.game.player_states[debtor].money = 0
    env.loop.game.debts.append(Debt(debtor, None, 80))
    env.loop.turn_phase = GameLoop.RESOLVE_DEBT
    env.loop._settle()

    action = make_action(
        env.action_map,
        ActionType.PROPOSE_TRADE,
        target_player=1,
        money_requested=np.asarray([100], dtype=np.int32),
    )
    result = env.step(action)

    assert result[2] is False
    assert env.loop.game.debts == []
    assert env.loop.game.player_states[debtor].money == 20
    assert env.loop.game.player_states[creditor].money == 1400


def test_pettingzoo_v2_forced_auction_and_bankruptcy_flow():
    env = raw_env(num_players=2, max_players=12)
    env.reset(seed=12)
    debtor = env.loop.current_decision_player_id
    bidder = next(player_id for player_id in env.loop.game.turn_order if player_id != debtor)
    tile = env.loop.game.board.property_tiles[0]
    env.loop.game.property_states[tile.tile_id].owner_id = debtor
    env.loop.game.player_states[debtor].money = 0
    env.loop.game.debts.append(Debt(debtor, None, 1000))
    env.loop._settle()
    env.agent_selection = env._agent(debtor)

    env.step(make_action(env.action_map, ActionType.END_DEBT_NEGOTIATIONS))
    env.step(make_action(
        env.action_map,
        ActionType.AUCTION_DEBT_PROPERTY,
        tile_id=tile.tile_id,
    ))
    assert env.loop.current_decision_player_id == bidder
    bid = make_action(
        env.action_map,
        ActionType.AUCTION_BID,
        amount=np.asarray([10], dtype=np.int32),
    )
    env.step(bid)
    assert env.loop.game.property_states[tile.tile_id].owner_id == bidder
    assert env.loop.current_decision_player_id == debtor

    env.step(make_action(env.action_map, ActionType.DECLARE_BANKRUPTCY))
    assert env.loop.game.player_states[debtor].is_bankrupt
    assert env.loop.game.winner_id == bidder
