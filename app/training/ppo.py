"""Minimal masked PPO trainer for the candidate-action Gymnasium environment.

PyTorch is an optional dependency. The policy scores each legal action from its
features and the current observation, so candidate row order has no meaning.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical

from app.envs.baselines import GreedyPolicy, RandomPolicy
from app.envs.candidate_env import CandidateSelectionEnv
from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Settings for a reproducible first PPO experiment."""

    episodes: int = 100
    seed: int = 2026
    num_players: int = 2
    max_turns: int = 1000
    max_candidates: int = 64
    episodes_per_update: int = 8
    update_epochs: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    hidden_size: int = 128
    evaluation_interval: int = 20
    evaluation_games: int = 20
    trade_analysis_mode: str = "disabled"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.episodes_per_update < 1:
            raise ValueError("episodes e episodes_per_update devem ser positivos.")
        if not 2 <= self.num_players <= 12:
            raise ValueError("num_players deve estar entre 2 e 12.")
        if self.max_turns < 1 or self.max_candidates < 1:
            raise ValueError("max_turns e max_candidates devem ser positivos.")
        if self.trade_analysis_mode not in {"disabled", "automatic"}:
            raise ValueError("use trade_analysis_mode='disabled' ou 'automatic'.")


class _BaselinePool:
    """Choose a seeded baseline profile once per episode."""

    def __init__(self) -> None:
        self.policy: Any = GreedyPolicy()

    def reset(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        profiles = (
            lambda: RandomPolicy(seed),
            lambda: GreedyPolicy(reserve=0),
            lambda: GreedyPolicy(reserve=25),
            lambda: GreedyPolicy(reserve=50),
            lambda: GreedyPolicy(reserve=150),
        )
        self.policy = rng.choice(profiles)()

    def __call__(self, loop):
        return self.policy(loop)


@dataclass
class _State:
    global_features: np.ndarray
    candidate_actions: np.ndarray
    candidate_mask: np.ndarray
    analysis_indices: np.ndarray
    analysis_features: np.ndarray


class _ActorCritic(nn.Module):
    """Shared state encoder, per-candidate actor, and state-value critic."""

    def __init__(self, global_size: int, action_size: int, analysis_size: int, hidden: int):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(global_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(action_size, hidden // 2), nn.Tanh())
        self.analysis_encoder = nn.Sequential(nn.Linear(analysis_size, hidden // 2), nn.Tanh())
        self.actor = nn.Sequential(
            nn.Linear(hidden + hidden // 2 + hidden // 2 + 1, hidden),
            nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.critic = nn.Linear(hidden, 1)

    def forward_one(self, state: _State, device: torch.device):
        global_features = torch.as_tensor(
            state.global_features, dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            state.candidate_actions, dtype=torch.float32, device=device
        )
        mask = torch.as_tensor(state.candidate_mask, dtype=torch.bool, device=device)
        state_embedding = self.global_encoder(global_features)
        action_embedding = self.action_encoder(actions)
        analysis_embedding = torch.zeros(
            (len(actions), self.analysis_encoder[0].out_features),
            dtype=torch.float32,
            device=device,
        )
        has_analysis = torch.zeros((len(actions), 1), dtype=torch.float32, device=device)
        if len(state.analysis_indices):
            indices = torch.as_tensor(state.analysis_indices, dtype=torch.long, device=device)
            features = torch.as_tensor(
                state.analysis_features, dtype=torch.float32, device=device
            )
            analysis_embedding[indices] = self.analysis_encoder(features)
            has_analysis[indices] = 1.0
        repeated_state = state_embedding.expand(len(actions), -1)
        logits = self.actor(torch.cat(
            (repeated_state, action_embedding, analysis_embedding, has_analysis), dim=-1
        )).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.critic(state_embedding).squeeze(-1)


class _CandidatePPO:
    """Masked categorical PPO actor bound to one candidate environment."""

    _candidate_keys = {
        "candidate_actions", "candidate_mask", "candidate_action_mask",
        "candidate_decision_features", "candidate_analysis_mask", "action_mask",
    }

    def __init__(self, env: CandidateSelectionEnv, config: PPOTrainingConfig):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        global_spaces = {
            key: space for key, space in env.observation_space.spaces.items()
            if key not in self._candidate_keys
        }
        self.global_space = spaces.Dict(global_spaces)
        global_size = int(np.prod(spaces.flatten_space(self.global_space).shape))
        self.action_size = env.action_feature_size
        self.analysis_size = env.decision_feature_size
        self.network = _ActorCritic(
            global_size, self.action_size, self.analysis_size, config.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def encode(self, observation: dict[str, Any]) -> _State:
        global_observation = {
            key: observation[key] for key in self.global_space.spaces
        }
        global_features = spaces.flatten(self.global_space, global_observation).astype(np.float32)
        global_features = np.nan_to_num(global_features, nan=0.0, posinf=1e6, neginf=-1e6)
        mask = np.asarray(observation["candidate_action_mask"], dtype=bool)
        # The first max_candidates indices are game actions. The final query
        # index is intentionally disabled during PPO training.
        n = int(np.count_nonzero(mask[:-1]))
        candidate_mask = mask[:n]
        candidates = np.asarray(observation["candidate_actions"][:n], dtype=np.float32)
        analysis_mask = np.asarray(observation["candidate_analysis_mask"][:n], dtype=bool)
        analysis_indices = np.flatnonzero(analysis_mask).astype(np.int64)
        analysis_features = np.asarray(
            observation["candidate_decision_features"][analysis_indices], dtype=np.float32
        )
        candidates = np.nan_to_num(candidates, nan=0.0, posinf=1e6, neginf=-1e6)
        analysis_features = np.nan_to_num(
            analysis_features, nan=0.0, posinf=1e6, neginf=-1e6
        )
        return _State(global_features, candidates, candidate_mask, analysis_indices, analysis_features)

    def act(self, state: _State, deterministic: bool = False):
        logits, value = self.network.forward_one(state, self.device)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else distribution.sample()
        return int(action.item()), distribution.log_prob(action), distribution.entropy(), value

    def update(self, batch: list[dict[str, Any]]) -> dict[str, float]:
        advantages = torch.as_tensor(
            [item["advantage"] for item in batch], dtype=torch.float32, device=self.device
        )
        if len(batch) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.config.update_epochs):
            order = np.random.permutation(len(batch))
            for start in range(0, len(batch), self.config.minibatch_size):
                indices = order[start:start + self.config.minibatch_size]
                new_logs, values, entropies = [], [], []
                for index in indices:
                    item = batch[int(index)]
                    logits, value = self.network.forward_one(item["state"], self.device)
                    distribution = Categorical(logits=logits)
                    action = torch.tensor(item["action"], device=self.device)
                    new_logs.append(distribution.log_prob(action))
                    values.append(value)
                    entropies.append(distribution.entropy())
                new_logs = torch.stack(new_logs)
                values = torch.stack(values)
                entropies = torch.stack(entropies)
                old_logs = torch.as_tensor(
                    [batch[int(index)]["log_prob"] for index in indices],
                    dtype=torch.float32, device=self.device,
                )
                adv = advantages[indices]
                returns = torch.as_tensor(
                    [batch[int(index)]["return"] for index in indices],
                    dtype=torch.float32, device=self.device,
                )
                ratio = torch.exp(new_logs - old_logs)
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * adv
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy = entropies.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy.detach())
                updates += 1
        return {key: value / max(1, updates) for key, value in totals.items()}


def _make_env(config: PPOTrainingConfig) -> CandidateSelectionEnv:
    return _make_env_with_opponents(
        config, [_BaselinePool() for _ in range(config.num_players - 1)]
    )


def _make_env_with_opponents(config: PPOTrainingConfig, opponents) -> CandidateSelectionEnv:
    base = MonopolyGymEnvV2(
        num_players=config.num_players,
        max_players=config.num_players,
        max_turns=config.max_turns,
        opponents=opponents,
    )
    return CandidateSelectionEnv(
        base,
        max_candidates=config.max_candidates,
        trade_analysis_mode=config.trade_analysis_mode,
    )


def _episode(env: CandidateSelectionEnv, agent: _CandidatePPO, seed: int, training: bool):
    observation, _ = env.reset(seed=seed)
    trajectory: list[dict[str, Any]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        state = agent.encode(observation)
        with torch.no_grad():
            action, log_prob, _, value = agent.act(state, deterministic=not training)
        observation, reward, terminated, truncated, _ = env.step(action)
        trajectory.append({
            "state": state,
            "action": action,
            "log_prob": float(log_prob.detach()),
            "value": float(value.detach()),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    game = env.env.loop.game
    learner_id = env.env.learner_id
    outcome = {
        "win": game.winner_id == learner_id,
        "match_decided": game.winner_id is not None,
        "draw": game.winner_id is None and not game.truncated,
        "bankrupt": game.player_states[learner_id].is_bankrupt,
        "truncated": bool(game.truncated),
        "turns": int(game.turn_counter),
        "decisions": len(trajectory),
        "return": sum(item["reward"] for item in trajectory),
        "seat": int(game.turn_order.index(learner_id)),
        "cash": int(game.player_states[learner_id].money),
        "net_worth": int(game.net_worth(learner_id)),
        "properties": int(len(game.owned_tiles(learner_id))),
    }
    if training:
        if terminated:
            next_value = 0.0
        else:
            with torch.no_grad():
                next_value = float(
                    agent.network.forward_one(agent.encode(observation), agent.device)[1]
                )
        gae = 0.0
        for item in reversed(trajectory):
            alive = 0.0 if item["terminated"] else 1.0
            delta = item["reward"] + agent.config.gamma * next_value * alive - item["value"]
            episode_end = item["terminated"] or item["truncated"]
            gae = delta + agent.config.gamma * agent.config.gae_lambda * (0.0 if episode_end else 1.0) * gae
            item["advantage"] = gae
            item["return"] = gae + item["value"]
            next_value = item["value"]
    return trajectory, outcome


def evaluate_ppo(
    agent: _CandidatePPO,
    config: PPOTrainingConfig,
    *,
    games: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate deterministically on a reproducible sequence of episodes."""
    env = _make_env(config)
    count = config.evaluation_games if games is None else games
    start_seed = config.seed + 10_000 if seed is None else seed
    results = [
        _episode(env, agent, start_seed + index, training=False)[1]
        for index in range(count)
    ]
    env.close()
    return {
        "games": float(count),
        "win_rate": float(np.mean([item["win"] for item in results])),
        "bankruptcy_rate": float(np.mean([item["bankrupt"] for item in results])),
        "truncation_rate": float(np.mean([item["truncated"] for item in results])),
        "mean_turns": float(np.mean([item["turns"] for item in results])),
        "mean_decisions": float(np.mean([item["decisions"] for item in results])),
    }


def _make_named_baseline(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return GreedyPolicy(reserve=150)
    if name.startswith("greedy:"):
        try:
            reserve = int(name.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Baseline Greedy inválida: {name!r}.") from exc
        if reserve < 0:
            raise ValueError("A reserva da Greedy deve ser não negativa.")
        return GreedyPolicy(reserve=reserve)
    raise ValueError(f"Baseline desconhecida: {name!r}.")


def evaluate_ppo_matchups(
    agent: _CandidatePPO,
    config: PPOTrainingConfig,
    *,
    opponents: tuple[str, ...] = ("random", "greedy:0", "greedy:25", "greedy:50", "greedy:150"),
    games: int = 100,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Return one reproducible result row per game and opponent profile.

    Each profile is evaluated in a separate fixed-opponent environment. Seeds
    are shared across profiles so matchups are paired, and the Gymnasium
    environment rotates the learner's seat from its episode seed.
    """
    if games < 1:
        raise ValueError("games deve ser positivo.")
    if not opponents:
        raise ValueError("Informe ao menos uma baseline oponente.")
    rows: list[dict[str, Any]] = []
    for name in opponents:
        env = _make_env_with_opponents(
            config,
            [_make_named_baseline(name) for _ in range(config.num_players - 1)],
        )
        for game_index in range(games):
            episode_seed = seed + game_index
            # Cycle seats explicitly so every profile is evaluated from the
            # same seat for each shared seed, with counts differing by at most one.
            env.env.fixed_seat = game_index % config.num_players
            _, outcome = _episode(env, agent, episode_seed, training=False)
            rows.append({
                "opponent": name,
                "game_index": game_index,
                "seed": episode_seed,
                **outcome,
            })
        env.close()
    return rows


def load_ppo_checkpoint(
    checkpoint_path: str | Path,
    config: PPOTrainingConfig | None = None,
) -> tuple[_CandidatePPO, PPOTrainingConfig]:
    """Load a PPO checkpoint and its saved configuration for evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if config is None:
        config = PPOTrainingConfig(**checkpoint["config"])
    env = _make_env(config)
    agent = _CandidatePPO(env, config)
    agent.network.load_state_dict(checkpoint["model"])
    agent.network.eval()
    return agent, config


def train_ppo(
    config: PPOTrainingConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[_CandidatePPO, list[dict[str, float]]]:
    """Train a masked PPO policy and return it with episode/evaluation metrics."""
    config = config or PPOTrainingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = _make_env(config)
    agent = _CandidatePPO(env, config)
    history: list[dict[str, float]] = []
    pending: list[dict[str, Any]] = []
    for episode in range(config.episodes):
        if progress_callback is not None:
            progress_callback({"event": "episode_start", "episode": float(episode + 1)})
        trajectory, outcome = _episode(env, agent, config.seed + episode, training=True)
        pending.extend(trajectory)
        row = {"episode": float(episode + 1), **{key: float(value) for key, value in outcome.items()}}
        if (episode + 1) % config.episodes_per_update == 0 or episode + 1 == config.episodes:
            row.update(agent.update(pending))
            pending.clear()
        if config.evaluation_interval and (episode + 1) % config.evaluation_interval == 0:
            row.update({f"eval_{key}": value for key, value in evaluate_ppo(
                agent, config, seed=config.seed + 100_000 + episode * config.evaluation_games
            ).items() if key != "games"})
        history.append(row)
        if progress_callback is not None:
            progress_callback({**row, "event": "episode_end"})
    env.close()
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": agent.network.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "config": asdict(config),
        }, path)
    return agent, history
