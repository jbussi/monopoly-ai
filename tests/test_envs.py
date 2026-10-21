import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from pettingzoo.test import api_test, seed_test  # noqa: E402

from app.application.game_setup import create_game  # noqa: E402
from app.envs import (  # noqa: E402
    DiscreteActionSpace,
    GreedyPolicy,
    MonopolyGymEnv,
    ObservationEncoder,
    render_text,
)
from app.envs import pettingzoo_env  # noqa: E402


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_pettingzoo_api(num_players):
    api_test(pettingzoo_env.env(num_players=num_players), num_cycles=1000)


def test_pettingzoo_seed():
    seed_test(lambda: pettingzoo_env.env(num_players=4), num_cycles=300)


def test_pettingzoo_full_episodes_are_zero_sum():
    env = pettingzoo_env.env(num_players=4)
    rng = np.random.default_rng(0)
    for episode in range(5):
        env.reset(seed=episode)
        totals = {a: 0.0 for a in env.possible_agents}
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            totals[agent] += reward
            env.step(None if term or trunc else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        assert sorted(totals.values()) in ([-1, -1, -1, 1], [0, 0, 0, 0])


def test_gymnasium_checker():
    check_env(MonopolyGymEnv(num_players=3, on_illegal="terminate"), skip_render_check=True)


def test_gymnasium_episode_against_greedy():
    env = MonopolyGymEnv(num_players=2, opponents=[GreedyPolicy()])
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        mask = env.action_masks()
        assert mask.any() and (mask == info["action_mask"].astype(bool)).all()
        obs, reward, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
        assert obs.shape == env.observation_space.shape
        done = term or trunc


def test_mask_matches_engine(config):
    loop = create_game(["A", "B", "C"], config, seed=2, max_turns=1000)
    space = DiscreteActionSpace(loop.game.board)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        if loop.game.game_over:
            break
        mask = space.mask(loop)
        for index in np.flatnonzero(mask):
            assert loop.validate(space.to_game_action(int(index), loop)) is None
        for index in np.flatnonzero(mask == 0)[:20]:
            assert loop.validate(space.to_game_action(int(index), loop)) is not None
        loop.process_action(space.to_game_action(int(rng.choice(np.flatnonzero(mask))), loop))


def test_observation_is_relative_to_observer(config):
    loop = create_game(["A", "B", "C"], config, seed=0)
    p0, p1, _ = loop.game.turn_order
    loop.game.player_states[p1].money = 3000
    enc = ObservationEncoder(loop)
    start = enc.layout["players"][0]
    stride = enc._player_fields
    assert enc.encode(loop, p1)[start + 1] == pytest.approx(2.0)           # slot 0 = observador
    assert enc.encode(loop, p0)[start + stride + 1] == pytest.approx(2.0)  # P1 é o próximo do P0
    assert enc.encode(loop, p0)[start + 3 * stride] == 0.0                 # 4º slot vazio


def test_render_text_lists_legal_actions(config):
    text = render_text(create_game(["A", "B"], config, seed=0))
    assert "ROLL_DICE" in text and "P1" in text
