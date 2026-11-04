import numpy as np

from app.domain.agents import (
    FeaturePolicy,
    LegalCandidateProvider,
    MaxScoreSelector,
    TradeCandidateProvider,
)
from app.domain.analytics.decision_context import DecisionContext
from app.domain.analytics.decision_features import EncodedDecisionContext
from app.domain.game.actions import ActionType, GameAction
from app.envs.rollout import run_gymnasium_episode, run_pettingzoo_episode


def test_trade_candidate_provider_generates_only_legal_seeded_offers(make_game):
    loop = make_game()
    proposer, recipient = loop.game.turn_order
    tile_a, tile_b = loop.game.board.property_tiles[:2]
    loop.game.property_states[tile_a.tile_id].owner_id = proposer
    loop.game.property_states[tile_b.tile_id].owner_id = recipient

    first = TradeCandidateProvider(max_candidates=32, seed=91)(loop)
    second = TradeCandidateProvider(max_candidates=32, seed=91)(loop)

    assert first == second
    assert first
    assert all(action.action_type == ActionType.PROPOSE_TRADE for action in first)
    assert all(loop.validate(action) is None for action in first)
    assert any(action.payload["properties_offered"] for action in first)
    assert any(action.payload["properties_requested"] for action in first)


def test_legal_candidate_provider_adds_trade_responses_and_counteroffers(make_game):
    loop = make_game()
    proposer, recipient = loop.game.turn_order
    proposal = GameAction(
        ActionType.PROPOSE_TRADE,
        {"target_player_id": recipient, "money_offered": 10},
    )
    assert loop.process_action(proposal)["status"] == "SUCCESS"
    assert loop.current_decision_player_id == recipient

    candidates = LegalCandidateProvider(max_candidates=64, seed=7)(loop)
    kinds = {action.action_type for action in candidates}

    assert ActionType.ACCEPT_TRADE in kinds
    assert ActionType.REJECT_TRADE in kinds
    assert ActionType.COUNTER_TRADE in kinds
    assert all(loop.validate(action) is None for action in candidates)


def test_feature_policy_default_provider_can_select_a_legal_trade(make_game):
    loop = make_game()
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
        return EncodedDecisionContext(context, {})

    selector = MaxScoreSelector(
        lambda _loop, candidate: float(
            candidate.action.action_type == ActionType.PROPOSE_TRADE
        )
    )
    policy = FeaturePolicy(evaluator, selector, max_candidates=64)

    selected = policy(loop)

    assert selected.action_type == ActionType.PROPOSE_TRADE
    assert loop.validate(selected) is None


def test_trade_candidate_provider_respects_counteroffer_limit(make_game):
    loop = make_game()
    proposer, recipient = loop.game.turn_order
    assert loop.process_action(GameAction(
        ActionType.PROPOSE_TRADE,
        {"target_player_id": recipient, "money_offered": 10},
    ))["status"] == "SUCCESS"
    for _ in range(loop.rules.trade_max_counteroffers):
        assert loop.process_action(
            GameAction(ActionType.COUNTER_TRADE, {"money_offered": 1})
        )["status"] == "SUCCESS"

    candidates = TradeCandidateProvider(seed=3)(loop)
    assert {item.action_type for item in candidates} == {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    }


class _GymActor:
    def __init__(self):
        self.seed = None
        self.calls = 0

    def reset(self, seed=None):
        self.seed = seed

    def act(self, observation, action_mask, *, deterministic=False):
        assert action_mask is observation["action_mask"]
        self.calls += 1
        return {"action_type": 0}


class _GymStub:
    def reset(self, *, seed):
        self.seed = seed
        self.steps = 0
        return {"action_mask": {"action_type": np.ones(1)}}, {}

    def step(self, action):
        assert action == {"action_type": 0}
        self.steps += 1
        return (
            {"action_mask": {"action_type": np.ones(1)}},
            0.5,
            self.steps == 2,
            False,
            {},
        )


def test_gymnasium_rollout_resets_actor_and_collects_episode_result():
    env = _GymStub()
    actor = _GymActor()

    result = run_gymnasium_episode(env, actor, seed=42, deterministic=True)

    assert env.seed == actor.seed == 42
    assert actor.calls == result.decisions == 2
    assert result.episode_return == 1.0
    assert result.terminated and not result.truncated
    assert not result.illegal_action


class _AECStub:
    possible_agents = ["player_0", "player_1"]

    def reset(self, *, seed):
        self.seed = seed
        self.rewards = {name: 0.0 for name in self.possible_agents}
        self.terminations = {name: False for name in self.possible_agents}
        self.truncations = {name: False for name in self.possible_agents}
        self.sequence = iter(("player_0", "player_1", "player_0", "player_1"))

    def agent_iter(self):
        for name in self.sequence:
            self.current = name
            yield name

    def last(self):
        name = self.current
        reward = self.rewards[name]
        self.rewards[name] = 0.0
        return (
            {"action_mask": {"action_type": np.ones(1)}},
            reward,
            self.terminations[name],
            self.truncations[name],
            {},
        )

    def step(self, action):
        if self.terminations[self.current] or self.truncations[self.current]:
            assert action is None
            return
        assert action == {"action_type": 0}
        if self.current == "player_0":
            self.rewards["player_1"] -= 1.0
        else:
            self.rewards["player_0"] += 1.0
            self.terminations = {name: True for name in self.possible_agents}


def test_pettingzoo_rollout_tracks_rewards_and_decisions_per_agent():
    env = _AECStub()
    actors = {name: _GymActor() for name in env.possible_agents}

    result = run_pettingzoo_episode(env, actors, seed=17, deterministic=True)

    assert actors["player_0"].seed == 17 * 1_000_003
    assert actors["player_1"].seed == 17 * 1_000_003 + 1
    assert result.episode_returns == {"player_0": 1.0, "player_1": -1.0}
    assert result.decisions_by_agent == {"player_0": 1, "player_1": 1}
    assert all(result.terminations.values())
    assert not any(result.truncations.values())


class _SpaceSamplingActor:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, action_mask, *, deterministic=False):
        assert "action_type" in action_mask
        return self.action_space.sample()


def test_rollout_helpers_run_against_real_v2_environments():
    from app.envs.baselines import GreedyPolicy
    from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2
    from app.envs.pettingzoo_env_v2 import env as pettingzoo_env

    gym_env = MonopolyGymEnvV2(
        num_players=2,
        max_players=2,
        learner_seat=0,
        max_turns=2,
        opponents=[GreedyPolicy()],
    )
    gym_result = run_gymnasium_episode(
        gym_env,
        _SpaceSamplingActor(gym_env.action_space),
        seed=75,
        deterministic=True,
    )
    assert gym_result.decisions > 0
    assert gym_result.terminated or gym_result.truncated

    pz_env = pettingzoo_env(num_players=2, max_players=2, max_turns=2)
    actors = {
        name: _SpaceSamplingActor(pz_env.action_space(name))
        for name in pz_env.possible_agents
    }
    pz_result = run_pettingzoo_episode(
        pz_env,
        actors,
        seed=76,
        deterministic=True,
    )
    assert sum(pz_result.decisions_by_agent.values()) > 0
    assert any(pz_result.terminations.values()) or any(pz_result.truncations.values())


def test_candidate_selection_env_returns_masked_discrete_choices():
    from app.domain.agents import CandidateRLAgent
    from app.envs import CandidateSelectionEnv
    from app.envs.baselines import GreedyPolicy
    from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2

    class FirstLegalCandidate:
        def reset(self, seed=None):
            self.seed = seed

        def act(self, observation, action_mask, *, deterministic=False):
            return int(np.flatnonzero(action_mask)[0])

    base = MonopolyGymEnvV2(
        num_players=2,
        max_players=2,
        learner_seat=0,
        max_turns=2,
        opponents=[GreedyPolicy()],
    )
    env = CandidateSelectionEnv(base, max_candidates=64)
    actor = FirstLegalCandidate()
    observation, _ = env.reset(seed=99)

    assert env.observation_space.contains(observation)
    assert env.action_space.n == 65
    assert observation["candidate_mask"].sum() > 0
    assert observation["candidate_analysis_mask"].sum() == 0
    assert observation["candidate_action_mask"][-1] == 1
    assert env.action_masks().tolist() == observation["candidate_action_mask"].astype(bool).tolist()
    analyzed, reward, terminated, truncated, info = env.step(env.analysis_action_index)
    assert info["analysis_requested"]
    assert reward == 0.0 and not terminated and not truncated
    assert analyzed["candidate_analysis_mask"].sum() > 0
    assert analyzed["candidate_action_mask"][-1] == 0
    result = run_gymnasium_episode(
        env,
        actor,
        seed=99,
        deterministic=True,
        action_mask_key="candidate_action_mask",
    )

    assert actor.seed == 99
    assert result.decisions > 0
