import pytest

from app.envs.rewards import placement_rewards, win_loss_reward


@pytest.mark.parametrize("num_players", range(2, 13))
def test_completed_placement_rewards_are_symmetric(num_players):
    rewards = placement_rewards(num_players)

    assert len(rewards) == num_players
    assert rewards[0] == 1.0
    assert rewards[-1] == -1.0
    assert rewards == pytest.approx(tuple(-value for value in reversed(rewards)))
    assert sum(rewards) == pytest.approx(0.0)
    if num_players % 2:
        assert rewards[num_players // 2] == 0.0


@pytest.mark.parametrize("num_players", range(2, 13))
def test_completed_game_reward_matches_elimination_placement(make_game, num_players):
    loop = make_game(num_players)
    game = loop.game
    winner = game.turn_order[-1]
    elimination_order = game.turn_order[:-1]
    game.elimination_order = elimination_order
    game.winner_id = winner
    expected = placement_rewards(num_players)

    assert win_loss_reward(loop, winner) == expected[0]
    for index, player_id in enumerate(elimination_order):
        game.player_states[player_id].is_bankrupt = True
        assert win_loss_reward(loop, player_id) == expected[num_players - index - 1]


@pytest.mark.parametrize(
    ("num_players", "eliminated_count"),
    [
        (num_players, eliminated_count)
        for num_players in range(2, 13)
        for eliminated_count in range(num_players)
    ],
)
def test_truncated_game_rewards_eliminated_and_surviving_players(
    make_game, num_players, eliminated_count
):
    loop = make_game(num_players)
    game = loop.game
    eliminated = game.turn_order[:eliminated_count]
    survivors = game.turn_order[eliminated_count:]
    game.elimination_order = eliminated.copy()
    for player_id in eliminated:
        game.player_states[player_id].is_bankrupt = True
    game.game_over = True
    game.truncated = True

    normal = placement_rewards(num_players)
    for index, player_id in enumerate(eliminated):
        expected_place = num_players - index
        assert win_loss_reward(loop, player_id) == normal[expected_place - 1]

    survivor_count = len(survivors)
    if survivor_count == 1:
        expected_survival = 1.0
    elif normal[survivor_count - 1] > 0.0:
        expected_survival = 1.05 * normal[survivor_count - 1]
    else:
        expected_survival = 0.4**survivor_count
    survivor_rewards = [win_loss_reward(loop, player_id) for player_id in survivors]

    assert survivor_rewards == pytest.approx([expected_survival] * survivor_count)
    assert all(reward > 0.0 for reward in survivor_rewards)
    assert all(reward > max((win_loss_reward(loop, pid) for pid in eliminated), default=-1.0)
               for reward in survivor_rewards)


@pytest.mark.parametrize("num_players", range(2, 13))
def test_fewer_truncated_survivors_receive_larger_survival_reward(make_game, num_players):
    rewards_by_survivor_count = []
    for survivor_count in range(1, num_players + 1):
        loop = make_game(num_players)
        game = loop.game
        eliminated_count = num_players - survivor_count
        eliminated = game.turn_order[:eliminated_count]
        game.elimination_order = eliminated.copy()
        for player_id in eliminated:
            game.player_states[player_id].is_bankrupt = True
        game.truncated = True
        rewards_by_survivor_count.append(
            win_loss_reward(loop, game.turn_order[-1])
        )

    assert rewards_by_survivor_count == sorted(rewards_by_survivor_count, reverse=True)


def test_game_loop_records_eliminations_without_event_logging(make_game):
    loop = make_game(3, record_events=False)
    first_eliminated, second_eliminated, survivor = loop.game.turn_order

    loop._bankrupt(first_eliminated, None)
    loop._bankrupt(second_eliminated, None)

    assert loop.game.elimination_order == [first_eliminated, second_eliminated]
    assert loop.game.winner_id == survivor
