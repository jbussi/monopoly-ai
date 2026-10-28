"""Testes de propriedade: milhares de ações aleatórias não podem quebrar invariantes."""

import random

import pytest

from app.application.game_setup import create_game
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_state import HOTEL_LEVEL
from app.envs.baselines import GreedyPolicy, RandomPolicy

ALL_KINDS = list(ActionType)


def check_invariants(loop, config):
    game = loop.game
    assert all(state.money >= 0 for state in game.player_states.values())

    houses = sum(p.houses for p in game.property_states.values() if p.houses < HOTEL_LEVEL)
    hotels = sum(1 for p in game.property_states.values() if p.houses == HOTEL_LEVEL)
    assert game.houses_in_bank + houses == loop.rules.max_houses_in_bank
    assert game.hotels_in_bank + hotels == loop.rules.max_hotels_in_bank

    for prop in game.property_states.values():
        if prop.owner_id is None:
            assert not prop.is_mortgaged and prop.houses == 0
        else:
            assert not game.player_states[prop.owner_id].is_bankrupt
        assert not (prop.is_mortgaged and prop.houses)

    in_hands = sum(len(s.jail_cards) for s in game.player_states.values())
    in_decks = sum(d.remaining + d.discarded for d in game.decks.values())
    assert in_hands + in_decks == len(config.chance_cards) + len(config.community_chest_cards)

    if not game.game_over:
        assert loop.legal_actions()
        assert not game.player_states[loop.current_decision_player_id].is_bankrupt


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_random_games_keep_invariants(config, num_players):
    for seed in range(25):
        loop = create_game([f"P{i}" for i in range(num_players)], config, seed=seed,
                           max_turns=1000, record_events=False)
        rng = random.Random(seed)
        while not loop.game.game_over:
            # De vez em quando tenta uma ação qualquer: se for rejeitada, nada pode mudar
            if rng.random() < 0.1:
                junk = GameAction(rng.choice(ALL_KINDS), {"tile_id": rng.randrange(40), "amount": rng.randrange(500)})
                before = loop.legal_actions()
                if loop.process_action(junk)["status"] == "REJECTED":
                    assert loop.legal_actions() == before
                else:
                    assert junk.action_type in {a.action_type for a in before}
                    check_invariants(loop, config)
                continue
            result = loop.process_action(rng.choice(loop.legal_actions()))
            assert result["status"] == "SUCCESS", result
            check_invariants(loop, config)
        assert loop.game.winner_id is not None or loop.game.truncated


def test_greedy_games(config):
    for seed in range(10):
        loop = create_game(["A", "B"], config, seed=seed, max_turns=1000, record_events=False)
        policy = GreedyPolicy()
        while not loop.game.game_over:
            assert loop.process_action(policy(loop))["status"] == "SUCCESS"
            check_invariants(loop, config)


def test_twelve_player_random_game_keeps_invariants(config):
    loop = create_game(
        [f"P{i}" for i in range(12)], config, seed=1200,
        max_turns=120, record_events=False,
    )
    policies = {
        player_id: RandomPolicy(1200 + seat)
        for seat, player_id in enumerate(loop.game.turn_order)
    }

    while not loop.game.game_over:
        player_id = loop.current_decision_player_id
        result = loop.process_action(policies[player_id](loop))
        assert result["status"] == "SUCCESS", result
        check_invariants(loop, config)

    assert loop.game.winner_id is not None or loop.game.truncated


def test_same_seed_same_game(config):
    def play(seed):
        loop = create_game(["A", "B", "C", "D"], config, seed=seed, max_turns=500)
        policy = RandomPolicy(seed)
        while not loop.game.game_over:
            loop.process_action(policy(loop))
        return [(e["event"], e["data"]) for e in loop.game.history_logs]

    assert play(3) == play(3)
    assert play(3) != play(4)


def test_clone_is_independent_and_equivalent(config):
    loop = create_game(["A", "B", "C"], config, seed=1, max_turns=1000)
    policy = RandomPolicy(1)
    for _ in range(150):
        loop.process_action(policy(loop))

    twin = loop.clone()
    assert twin.game.board is loop.game.board
    p1, p2 = RandomPolicy(9), RandomPolicy(9)
    for _ in range(300):
        if loop.game.game_over:
            break
        loop.process_action(p1(loop))
        twin.process_action(p2(twin))

    money = lambda lp: [lp.game.player_states[p].money for p in lp.game.turn_order]  # noqa: E731
    assert money(loop) == money(twin)
    twin.game.player_states[twin.game.turn_order[0]].money += 1
    assert money(loop) != money(twin)
