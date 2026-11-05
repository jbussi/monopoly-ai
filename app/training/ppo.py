"""Minimal masked PPO trainer for the candidate-action Gymnasium environment.

PyTorch is an optional dependency. The policy scores each legal action from its
features and the current observation, so candidate row order has no meaning.
"""

from __future__ import annotations

import random
import copy
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical

from app.envs.baselines import AggressivePolicy, ConservativePolicy, RandomPolicy
from app.envs.candidate_env import CandidateSelectionEnv
from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2
from app.envs.action_space_v2 import ACTION_TYPES
from app.training.league import TrainingLeague


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Settings for population PPO training in four-player games."""

    episodes: int = 1000
    seed: int = 2026
    num_players: int = 4
    max_turns: int = 1000
    max_candidates: int = 128
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
    action_type_embedding_size: int = 16
    evaluation_interval: int = 0
    evaluation_games: int = 20
    generation_size: int = 100
    archive_opponent_rate: float = 0.5
    archive_dir: str = "runs/policy_bank"
    trade_analysis_mode: str = "automatic"
    num_workers: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.episodes_per_update < 1:
            raise ValueError("episodes e episodes_per_update devem ser positivos.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 1:
            raise ValueError("num_workers deve ser um inteiro positivo.")
        if self.device != "auto" and not self.device.startswith(("cpu", "cuda")):
            raise ValueError("device deve ser 'auto', 'cpu', 'cuda' ou 'cuda:N'.")
        if not 2 <= self.num_players <= 12:
            raise ValueError("num_players deve estar entre 2 e 12.")
        if self.max_turns < 1 or self.max_candidates < 1 or self.action_type_embedding_size < 1:
            raise ValueError("max_turns, max_candidates e action_type_embedding_size devem ser positivos.")
        if self.trade_analysis_mode not in {"disabled", "automatic"}:
            raise ValueError("use trade_analysis_mode='disabled' ou 'automatic'.")
        if self.generation_size < 1:
            raise ValueError("generation_size deve ser positivo.")
        if not 0.0 <= self.archive_opponent_rate <= 1.0:
            raise ValueError("archive_opponent_rate deve estar entre 0 e 1.")


class _ArchivedPolicy:
    """Run a frozen PPO checkpoint as an opponent in the shared live game."""

    def __init__(self, agent: _CandidatePPO, candidate_env: CandidateSelectionEnv, seed: int):
        self.agent = agent
        self.candidate_env = candidate_env
        self.base_env = candidate_env.env
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def __call__(self, loop):
        player_id = loop.current_decision_player_id
        original_id = self.base_env.learner_id
        self.base_env.learner_id = player_id
        try:
            observation = self.base_env._observation()
            observation = self.candidate_env._with_candidates(observation)
            state = self.agent.encode(observation)
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.rng.getrandbits(63))
                index = self.agent.act(state, deterministic=False)[0]
            return self.candidate_env.candidate_actions[index]
        finally:
            self.base_env.learner_id = original_id


@dataclass
class _State:
    global_features: np.ndarray
    candidate_types: np.ndarray
    candidate_actions: np.ndarray
    candidate_mask: np.ndarray
    analysis_indices: np.ndarray
    analysis_features: np.ndarray


class _ActorCritic(nn.Module):
    """Shared state encoder, per-candidate actor, and state-value critic."""

    def __init__(
        self,
        global_size: int,
        action_size: int,
        analysis_size: int,
        num_action_types: int,
        type_embedding_size: int,
        hidden: int,
    ):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(global_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.action_type_embedding = nn.Embedding(num_action_types, type_embedding_size)
        self.action_encoder = nn.Sequential(nn.Linear(action_size, hidden // 2), nn.Tanh())
        self.decision_encoder = nn.Sequential(nn.Linear(analysis_size, hidden // 2), nn.Tanh())
        self.actor = nn.Sequential(
            nn.Linear(hidden + type_embedding_size + hidden // 2 + hidden // 2 + 1, hidden),
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
        action_types = torch.as_tensor(
            state.candidate_types, dtype=torch.long, device=device
        )
        mask = torch.as_tensor(state.candidate_mask, dtype=torch.bool, device=device)
        state_embedding = self.global_encoder(global_features)
        type_embedding = self.action_type_embedding(action_types)
        action_embedding = self.action_encoder(actions)
        analysis_embedding = torch.zeros(
            (len(actions), self.decision_encoder[0].out_features),
            dtype=torch.float32,
            device=device,
        )
        has_analysis = torch.zeros((len(actions), 1), dtype=torch.float32, device=device)
        if len(state.analysis_indices):
            indices = torch.as_tensor(state.analysis_indices, dtype=torch.long, device=device)
            features = torch.as_tensor(
                state.analysis_features, dtype=torch.float32, device=device
            )
            analysis_embedding[indices] = self.decision_encoder(features)
            has_analysis[indices] = 1.0
        repeated_state = state_embedding.expand(len(actions), -1)
        logits = self.actor(torch.cat(
            (repeated_state, type_embedding, action_embedding, analysis_embedding, has_analysis), dim=-1
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
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "Foi solicitado device CUDA, mas torch.cuda.is_available() retornou False. "
                    "Use device='auto' ou device='cpu'."
                )
        global_spaces = {
            key: space for key, space in env.observation_space.spaces.items()
            if key not in self._candidate_keys
        }
        self.global_space = spaces.Dict(global_spaces)
        global_size = int(np.prod(spaces.flatten_space(self.global_space).shape))
        self.action_size = env.action_feature_size
        self.analysis_size = env.decision_feature_size
        self.num_action_types = len(ACTION_TYPES)
        self.type_embedding_size = config.action_type_embedding_size
        self.network = _ActorCritic(
            global_size,
            self.action_size - self.num_action_types,
            self.analysis_size,
            self.num_action_types,
            self.type_embedding_size,
            config.hidden_size,
        ).to(self.device)
        self.inference_device = torch.device("cpu")
        self.inference_network = (
            self.network
            if self.device.type == "cpu"
            else copy.deepcopy(self.network).to(self.inference_device)
        )
        self.inference_network.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def sync_inference_network(self) -> None:
        """Copy current learner weights to the CPU policy used during rollouts."""
        if self.inference_network is self.network:
            return
        self.inference_network.load_state_dict({
            name: tensor.detach().to("cpu")
            for name, tensor in self.network.state_dict().items()
        })
        self.inference_network.eval()

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
        type_vectors = candidates[:, :self.num_action_types]
        candidate_types = np.argmax(type_vectors, axis=1).astype(np.int64)
        candidate_payloads = candidates[:, self.num_action_types:]
        analysis_mask = np.asarray(observation["candidate_analysis_mask"][:n], dtype=bool)
        analysis_indices = np.flatnonzero(analysis_mask).astype(np.int64)
        analysis_features = np.asarray(
            observation["candidate_decision_features"][analysis_indices], dtype=np.float32
        )
        candidate_payloads = np.nan_to_num(
            candidate_payloads, nan=0.0, posinf=1e6, neginf=-1e6
        )
        analysis_features = np.nan_to_num(
            analysis_features, nan=0.0, posinf=1e6, neginf=-1e6
        )
        return _State(
            global_features,
            candidate_types,
            candidate_payloads,
            candidate_mask,
            analysis_indices,
            analysis_features,
        )

    def act(self, state: _State, deterministic: bool = False):
        logits, value = self.inference_network.forward_one(state, self.inference_device)
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
        self.sync_inference_network()
        return {key: value / max(1, updates) for key, value in totals.items()}


def _make_env(config: PPOTrainingConfig) -> CandidateSelectionEnv:
    return _make_env_with_opponents(
        config, _heuristic_lineup(config.num_players - 1)
    )


def _heuristic_lineup(count: int) -> list[Any]:
    league = TrainingLeague()
    names = tuple(league.heuristics)
    return [league.create_heuristic(names[index % len(names)]) for index in range(count)]


def _make_env_with_opponents(config: PPOTrainingConfig, opponents) -> CandidateSelectionEnv:
    base = MonopolyGymEnvV2(
        num_players=config.num_players,
        # Keep observation/action dimensions stable so archived policies can
        # face games with a different active player count.
        max_players=12,
        max_turns=config.max_turns,
        opponents=opponents,
    )
    return CandidateSelectionEnv(
        base,
        max_candidates=config.max_candidates,
        trade_analysis_mode=config.trade_analysis_mode,
    )


_ROLLOUT_WORKER_CONFIG: PPOTrainingConfig | None = None
_ROLLOUT_WORKER_ENV: CandidateSelectionEnv | None = None
_ROLLOUT_WORKER_AGENT: _CandidatePPO | None = None
_ROLLOUT_WORKER_ARCHIVE: dict[int, _CandidatePPO] = {}


def _initialize_rollout_worker(config: PPOTrainingConfig) -> None:
    """Create one persistent CPU-only game environment inside a worker process."""
    global _ROLLOUT_WORKER_CONFIG, _ROLLOUT_WORKER_ENV, _ROLLOUT_WORKER_AGENT
    global _ROLLOUT_WORKER_ARCHIVE
    worker_config = replace(config, device="cpu", num_workers=1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _ROLLOUT_WORKER_CONFIG = worker_config
    _ROLLOUT_WORKER_ENV = _make_env(worker_config)
    _ROLLOUT_WORKER_AGENT = _CandidatePPO(_ROLLOUT_WORKER_ENV, worker_config)
    _ROLLOUT_WORKER_ARCHIVE = {}


def _cpu_state_dict(network: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model weights to CPU tensors for process transfer and checkpoints."""
    return {
        name: tensor.detach().to("cpu", copy=True)
        for name, tensor in network.state_dict().items()
    }


def _serialize_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rollout states to safe tensor/primitive checkpoint values."""
    serialized = []
    for item in trajectory:
        state = item["state"]
        serialized.append({
            "state": {
                name: torch.as_tensor(getattr(state, name)).cpu().clone()
                for name in _State.__dataclass_fields__
            },
            **{key: value for key, value in item.items() if key != "state"},
        })
    return serialized


def _deserialize_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild rollout states loaded from a resumable checkpoint."""
    restored = []
    for item in trajectory:
        state_values = {
            name: item["state"][name].cpu().numpy().copy()
            for name in _State.__dataclass_fields__
        }
        restored.append({
            "state": _State(**state_values),
            **{key: value for key, value in item.items() if key != "state"},
        })
    return restored


def _rng_checkpoint() -> dict[str, Any]:
    """Capture random generators used by lineup sampling and PPO updates."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.as_tensor(numpy_state[1].copy(), dtype=torch.int64),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_checkpoint(state: dict[str, Any]) -> None:
    """Restore random generators after all checkpoint objects are initialized."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=True),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint beside its destination, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _write_episode_status(path: Path, **updates: Any) -> None:
    """Atomically record the latest durable progress for one training game."""
    status = {}
    if path.exists():
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
    if updates.get("status") == "queued" and status.get("status") == "running":
        status["previous_attempt"] = {
            key: status[key] for key in (
                "attempt", "learner_decisions", "game_turn", "turn_phase", "updated_at"
            ) if key in status
        }
    if updates.pop("attempt_started", False):
        status["attempt"] = int(status.get("attempt", 0)) + 1
    status.update(updates)
    status["updated_at"] = datetime.now().astimezone().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def _validate_resume_config(
    saved_config: dict[str, Any],
    config: PPOTrainingConfig,
    next_episode: int,
) -> None:
    """Reject resumes whose game or learning semantics differ from the checkpoint."""
    current = asdict(config)
    runtime_fields = {"episodes", "num_workers", "device"}
    differences = [
        key for key, value in current.items()
        if key not in runtime_fields and saved_config.get(key) != value
    ]
    if differences:
        raise ValueError(
            "A retomada exige a mesma configuração de jogo e PPO. Campos diferentes: "
            + ", ".join(differences)
        )
    if config.episodes < next_episode:
        raise ValueError(
            f"O checkpoint já concluiu {next_episode} partidas, acima do limite "
            f"configurado ({config.episodes})."
        )


def _run_rollout_worker(job: dict[str, Any]):
    """Run one episode on a persistent process-local CPU environment."""
    global _ROLLOUT_WORKER_AGENT, _ROLLOUT_WORKER_ENV, _ROLLOUT_WORKER_CONFIG
    global _ROLLOUT_WORKER_ARCHIVE
    if _ROLLOUT_WORKER_AGENT is None or _ROLLOUT_WORKER_ENV is None or _ROLLOUT_WORKER_CONFIG is None:
        raise RuntimeError("O worker PPO não foi inicializado.")
    agent = _ROLLOUT_WORKER_AGENT
    env = _ROLLOUT_WORKER_ENV
    config = _ROLLOUT_WORKER_CONFIG
    episode_seed = int(job["episode_seed"])
    random.seed(episode_seed)
    np.random.seed(episode_seed % (2**32))
    torch.manual_seed(episode_seed)
    agent.network.load_state_dict(job["learner_state"])
    agent.network.eval()
    agent.sync_inference_network()

    archive_states = iter(job["archive_states"])
    opponents = []
    for seat, profile_name in enumerate(job["profile_names"]):
        if profile_name != "archive":
            opponents.append(_make_named_baseline(profile_name))
            continue
        generation, state_dict = next(archive_states)
        archived = _ROLLOUT_WORKER_ARCHIVE.get(generation)
        if archived is None:
            archived = _CandidatePPO(env, config)
            _ROLLOUT_WORKER_ARCHIVE[generation] = archived
        archived.network.load_state_dict(state_dict)
        archived.network.eval()
        archived.sync_inference_network()
        opponents.append(_ArchivedPolicy(
            archived,
            env,
            seed=config.seed + int(job["episode_index"]) * 31 + seat + generation,
        ))
    env.env.opponents = opponents
    status_path = Path(job["status_path"])
    _write_episode_status(status_path, status="running", attempt_started=True)

    def report_progress(progress: dict[str, Any]) -> None:
        _write_episode_status(status_path, **progress)

    try:
        result = _episode(
            env, agent, episode_seed, training=True,
            progress_callback=report_progress,
        )
    except BaseException as exc:
        _write_episode_status(
            status_path, status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    _write_episode_status(
        status_path, status="rollout_complete", outcome=result[1]
    )
    return result


def _episode(
    env: CandidateSelectionEnv,
    agent: _CandidatePPO,
    seed: int,
    training: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
):
    from app.envs.cash_flow_forecast import forecast_stage_metrics

    forecast_stage_metrics(reset=True)
    policy_seconds = 0.0
    environment_step_seconds = 0.0
    reset_started = perf_counter()
    observation, _ = env.reset(seed=seed)
    reset_seconds = perf_counter() - reset_started
    trajectory: list[dict[str, Any]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        stage_started = perf_counter()
        state = agent.encode(observation)
        with torch.no_grad():
            action, log_prob, _, value = agent.act(state, deterministic=not training)
        policy_seconds += perf_counter() - stage_started
        stage_started = perf_counter()
        observation, reward, terminated, truncated, _ = env.step(action)
        environment_step_seconds += perf_counter() - stage_started
        trajectory.append({
            "state": state,
            "action": action,
            "log_prob": float(log_prob.detach()),
            "value": float(value.detach()),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if progress_callback is not None and len(trajectory) % 25 == 0:
            game = env.env.loop.game
            current_player = env.env.loop.current_decision_player_id
            progress_callback({
                "status": "running",
                "learner_decisions": len(trajectory),
                "game_turn": int(game.turn_counter),
                "turn_phase": str(env.env.loop.turn_phase),
                "current_actor_seat": (
                    game.turn_order.index(current_player)
                    if current_player in game.turn_order else None
                ),
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
        "winner_seat": (
            int(game.turn_order.index(game.winner_id))
            if game.winner_id in game.turn_order else -1
        ),
        "policy_seconds": policy_seconds,
        "episode_reset_seconds": reset_seconds,
        "environment_step_seconds": environment_step_seconds,
        "observation_seconds": float(env.env.observation_seconds),
        **{
            key: float(value)
            for key, value in env.stage_timings.items()
        },
        **forecast_stage_metrics(reset=True),
    }
    for seat, player_id in enumerate(game.turn_order):
        player_state = game.player_states[player_id]
        outcome.update({
            f"player_{seat}_cash": int(player_state.money),
            f"player_{seat}_net_worth": int(game.net_worth(player_id)),
            f"player_{seat}_position": int(game.position_of(player_id)),
            f"player_{seat}_properties": int(len(game.owned_tiles(player_id))),
            f"player_{seat}_bankrupt": int(player_state.is_bankrupt),
            f"player_{seat}_houses": int(sum(
                game.property_states[tile.tile_id].houses
                for tile in game.owned_tiles(player_id)
            )),
            f"player_{seat}_mortgages": int(sum(
                game.property_states[tile.tile_id].is_mortgaged
                for tile in game.owned_tiles(player_id)
            )),
        })
    if training:
        if terminated:
            next_value = 0.0
        else:
            with torch.no_grad():
                next_value = float(
                    agent.inference_network.forward_one(
                        agent.encode(observation), agent.inference_device
                    )[1]
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
    if name == "aggressive":
        return AggressivePolicy()
    if name == "conservative":
        return ConservativePolicy()
    raise ValueError(f"Baseline desconhecida: {name!r}.")


def evaluate_ppo_matchups(
    agent: _CandidatePPO,
    config: PPOTrainingConfig,
    *,
    opponents: tuple[str, ...] = ("random", "aggressive", "conservative"),
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
    if checkpoint.get("format") not in {
        "monopoly-ai-ppo-v2",
        "monopoly-ai-ppo-resume-v1",
        "monopoly-ai-policy-v2",
    }:
        raise ValueError(
            "Checkpoint incompatível com a arquitetura PPO atual; gere um novo checkpoint."
        )
    if config is None:
        config = PPOTrainingConfig(**checkpoint["config"])
    env = _make_env(config)
    agent = _CandidatePPO(env, config)
    agent.network.load_state_dict(checkpoint["model"])
    agent.network.eval()
    agent.sync_inference_network()
    return agent, config


def train_ppo(
    config: PPOTrainingConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    league: TrainingLeague | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[_CandidatePPO, list[dict[str, float]]]:
    """Train PPO with per-episode durable checkpoints and optional resumption."""
    config = config or PPOTrainingConfig()
    resume_payload = None
    if resume_from is not None:
        resume_path = Path(resume_from)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        resume_format = resume_payload.get("format")
        if resume_format == "monopoly-ai-policy-v2":
            # A generation snapshot can recover learned weights, but predates
            # per-game optimizer/history checkpoints; continue as a warm start.
            resume_payload = {
                **resume_payload,
                "format": "monopoly-ai-ppo-warm-start-v1",
                "next_episode": int(resume_payload["episode"]),
                "history": [],
                "pending": [],
                "generation": int(resume_payload["generation"]),
                "archive_generation_counts": {},
                "warm_start": True,
            }
        elif resume_format != "monopoly-ai-ppo-resume-v1":
            raise ValueError(
                "Informe um checkpoint PPO retomável ou um snapshot de geração do banco."
            )
        if checkpoint_path is None:
            checkpoint_path = resume_path
        _validate_resume_config(
            resume_payload["config"], config, int(resume_payload["next_episode"])
        )
    training_league = league or TrainingLeague(
        archive_opponent_rate=config.archive_opponent_rate
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = _make_env(config)
    agent = _CandidatePPO(env, config)
    history: list[dict[str, float]] = []
    pending: list[dict[str, Any]] = []
    archive: list[tuple[int, _CandidatePPO]] = []
    archive_root = Path(config.archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)
    profile_counts = {name: 0 for name in training_league.heuristics}
    profile_counts["archive"] = 0
    archive_generation_counts: dict[str, int] = {}
    generation = 0
    start_episode = 0
    if resume_payload is not None:
        agent.network.load_state_dict(resume_payload["model"])
        if resume_payload.get("optimizer") is not None:
            agent.optimizer.load_state_dict(resume_payload["optimizer"])
        agent.sync_inference_network()
        history = list(resume_payload["history"])
        pending = _deserialize_trajectory(resume_payload["pending"])
        profile_counts.update(resume_payload.get("profile_counts", {}))
        archive_generation_counts = dict(resume_payload["archive_generation_counts"])
        generation = int(resume_payload["generation"])
        start_episode = int(resume_payload["next_episode"])
        saved_league = resume_payload.get("league_names")
        if saved_league is not None and tuple(saved_league) != tuple(training_league.heuristics):
            raise ValueError(
                "A liga de oponentes difere do checkpoint; não é seguro retomar."
            )
    manifest_path = archive_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "monopoly-ai-policy-bank-v2":
            raise ValueError(
                "Banco de políticas incompatível com a arquitetura PPO atual. "
                "Configure um novo archive_dir para começar um banco v2."
            )
        for entry in manifest.get("policies", []):
            if (
                resume_payload is not None
                and int(entry["generation"]) > int(resume_payload["generation"])
            ):
                # Ignore a policy file created after the last committed
                # training checkpoint; the interrupted episode will be replayed.
                continue
            policy_path = archive_root / entry["path"]
            if not policy_path.exists():
                continue
            saved = torch.load(policy_path, map_location="cpu", weights_only=True)
            if saved.get("format") != "monopoly-ai-policy-v2":
                raise ValueError(
                    f"Política arquivada incompatível com a arquitetura atual: {policy_path}. "
                    "Use um banco novo ou treine snapshots v2."
                )
            snapshot = _CandidatePPO(env, config)
            snapshot.network.load_state_dict(saved["model"])
            snapshot.network.eval()
            snapshot.sync_inference_network()
            archive.append((int(saved["generation"]), snapshot))
            generation = max(generation, int(saved["generation"]))

    def save_generation(episode_number: int) -> None:
        nonlocal generation
        generation += 1
        snapshot = _CandidatePPO(env, config)
        snapshot.network.load_state_dict(copy.deepcopy(agent.network.state_dict()))
        snapshot.network.eval()
        snapshot.sync_inference_network()
        archive.append((generation, snapshot))
        payload_config = asdict(config)
        path = archive_root / f"generation_{generation:05d}.pt"
        _atomic_torch_save({
            "format": "monopoly-ai-policy-v2",
            "generation": generation,
            "episode": episode_number,
            "model": _cpu_state_dict(snapshot.network),
            "config": payload_config,
            "opponent_exposure": {
                **profile_counts,
                "archive_generations": dict(archive_generation_counts),
            },
        }, path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"format": "monopoly-ai-policy-bank-v2", "policies": []}
        manifest["policies"] = [
            entry for entry in manifest["policies"]
            if int(entry["generation"]) != generation
        ]
        manifest["policies"].append({
            "generation": generation,
            "episode": episode_number,
            "path": path.name,
            "opponent_exposure": {
                **profile_counts,
                "archive_generations": dict(archive_generation_counts),
            },
        })
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
        profile_counts.update({key: 0 for key in profile_counts})
        archive_generation_counts.clear()

    def save_training_checkpoint(next_episode: int) -> None:
        if checkpoint_path is None:
            return
        _atomic_torch_save({
            "format": "monopoly-ai-ppo-resume-v1",
            "model": _cpu_state_dict(agent.network),
            "optimizer": agent.optimizer.state_dict(),
            "config": asdict(config),
            "next_episode": next_episode,
            "generation": generation,
            "history": history,
            "pending": _serialize_trajectory(pending),
            "profile_counts": profile_counts,
            "archive_generation_counts": archive_generation_counts,
            "league_names": tuple(training_league.heuristics),
            "rng": _rng_checkpoint(),
        }, Path(checkpoint_path))

    rollout_pool = None
    rollout_futures = {}
    rollout_metadata = {}
    if config.num_workers > 1:
        worker_config = replace(config, device="cpu", num_workers=1)
        rollout_pool = ProcessPoolExecutor(
            max_workers=config.num_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_rollout_worker,
            initargs=(worker_config,),
        )

    if resume_payload is not None:
        if "rng" in resume_payload:
            _restore_rng_checkpoint(resume_payload["rng"])
        else:
            random.seed(config.seed + start_episode)
            np.random.seed((config.seed + start_episode) % (2**32))
            torch.manual_seed(config.seed + start_episode)
    status_root = archive_root / "episode_status"
    status_root.mkdir(parents=True, exist_ok=True)

    for episode in range(start_episode, config.episodes):
        status_path = status_root / f"episode_{episode + 1:06d}.json"
        if rollout_pool is None:
            episode_rng = random.Random(config.seed + episode)
            profile_names = training_league.sample_lineup(
                config.num_players - 1,
                archive_available=bool(archive),
                rng=episode_rng,
                episode_index=episode,
            )
            lineup = []
            for seat, profile_name in enumerate(profile_names):
                if profile_name == "archive":
                    old_generation, old_agent = episode_rng.choice(archive)
                    lineup.append(_ArchivedPolicy(
                        old_agent,
                        env,
                        seed=config.seed + episode * 31 + seat + old_generation,
                    ))
                    profile_counts["archive"] += 1
                    generation_label = str(old_generation)
                    archive_generation_counts[generation_label] = archive_generation_counts.get(generation_label, 0) + 1
                else:
                    lineup.append(training_league.create_heuristic(profile_name))
                    profile_counts[profile_name] += 1
            env.env.opponents = lineup
            _write_episode_status(
                status_path,
                status="queued",
                episode=episode + 1,
                generation=generation + 1,
                seed=config.seed + episode,
                opponents=profile_names,
            )
            _write_episode_status(
                status_path, status="running", attempt_started=True
            )

            def report_progress(progress: dict[str, Any]) -> None:
                _write_episode_status(status_path, **progress)

            if progress_callback is not None:
                progress_callback({
                    "event": "episode_start",
                    "episode": float(episode + 1),
                    "generation": float(generation + 1),
                    "opponents": tuple(profile_names),
                })
            trajectory, outcome = _episode(
                env, agent, config.seed + episode, training=True,
                progress_callback=report_progress,
            )
            _write_episode_status(
                status_path, status="rollout_complete", outcome=outcome
            )
        else:
            if episode not in rollout_futures:
                next_update = ((episode // config.episodes_per_update) + 1) * config.episodes_per_update
                next_generation = ((episode // config.generation_size) + 1) * config.generation_size
                batch_end = min(config.episodes, next_update, next_generation)
                learner_state = _cpu_state_dict(agent.network)
                for batch_episode in range(episode, batch_end):
                    episode_rng = random.Random(config.seed + batch_episode)
                    profile_names = training_league.sample_lineup(
                        config.num_players - 1,
                        archive_available=bool(archive),
                        rng=episode_rng,
                        episode_index=batch_episode,
                    )
                    archive_states = []
                    episode_profile_counts = {name: 0 for name in profile_counts}
                    episode_archive_counts: dict[str, int] = {}
                    for seat, profile_name in enumerate(profile_names):
                        if profile_name == "archive":
                            old_generation, old_agent = episode_rng.choice(archive)
                            archive_states.append((old_generation, _cpu_state_dict(old_agent.network)))
                            episode_profile_counts["archive"] += 1
                            generation_label = str(old_generation)
                            episode_archive_counts[generation_label] = (
                                episode_archive_counts.get(generation_label, 0) + 1
                            )
                        else:
                            episode_profile_counts[profile_name] += 1
                    if progress_callback is not None:
                        progress_callback({
                            "event": "episode_start",
                            "episode": float(batch_episode + 1),
                            "generation": float(generation + 1),
                            "opponents": tuple(profile_names),
                        })
                    episode_status_path = status_root / f"episode_{batch_episode + 1:06d}.json"
                    _write_episode_status(
                        episode_status_path,
                        status="queued",
                        episode=batch_episode + 1,
                        generation=generation + 1,
                        seed=config.seed + batch_episode,
                        opponents=profile_names,
                    )
                    rollout_futures[batch_episode] = rollout_pool.submit(
                        _run_rollout_worker,
                        {
                            "episode_index": batch_episode,
                            "episode_seed": config.seed + batch_episode,
                            "profile_names": tuple(profile_names),
                            "archive_states": archive_states,
                            "learner_state": learner_state,
                            "status_path": str(episode_status_path),
                        },
                    )
                    rollout_metadata[batch_episode] = (
                        episode_profile_counts, episode_archive_counts
                    )
            try:
                trajectory, outcome = rollout_futures.pop(episode).result()
            except BaseException as exc:
                rollout_pool.shutdown(wait=False, cancel_futures=True)
                env.close()
                raise RuntimeError(
                    f"Falha ao coletar a partida {episode + 1} da geração "
                    f"{episode // config.generation_size + 1}."
                ) from exc
            episode_profile_counts, episode_archive_counts = rollout_metadata.pop(episode)
            for name, count in episode_profile_counts.items():
                profile_counts[name] += count
            for generation_label, count in episode_archive_counts.items():
                archive_generation_counts[generation_label] = (
                    archive_generation_counts.get(generation_label, 0) + count
                )
        pending.extend(trajectory)
        row = {"episode": float(episode + 1), **{key: float(value) for key, value in outcome.items()}}
        generation_finished = (episode + 1) % config.generation_size == 0 or episode + 1 == config.episodes
        update_seconds = 0.0
        if (episode + 1) % config.episodes_per_update == 0 or generation_finished:
            update_started = perf_counter()
            row.update(agent.update(pending))
            update_seconds = perf_counter() - update_started
            pending.clear()
        row["ppo_update_seconds"] = update_seconds
        if config.evaluation_interval and (episode + 1) % config.evaluation_interval == 0:
            row.update({f"eval_{key}": value for key, value in evaluate_ppo(
                agent, config, seed=config.seed + 100_000 + episode * config.evaluation_games
            ).items() if key != "games"})
        if generation_finished:
            save_generation(episode + 1)
            row["generation"] = float(generation)
        history.append(row)
        if progress_callback is not None:
            progress_callback({**row, "event": "episode_end"})
        checkpoint_started = perf_counter()
        save_training_checkpoint(episode + 1)
        checkpoint_seconds = perf_counter() - checkpoint_started
        row["checkpoint_seconds"] = checkpoint_seconds
        history[-1] = row
        _write_episode_status(
            status_path,
            status="training_committed",
            episode=episode + 1,
            generation=int(row.get("generation", generation)),
            outcome=outcome,
            ppo_update_seconds=update_seconds,
            checkpoint_seconds=checkpoint_seconds,
        )
    env.close()
    if rollout_pool is not None:
        rollout_pool.shutdown(wait=True, cancel_futures=True)
    if checkpoint_path is not None:
        save_training_checkpoint(config.episodes)
    return agent, history
