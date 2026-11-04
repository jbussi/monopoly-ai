"""Small model-agnostic rollout helpers for Gymnasium v2 agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
from pettingzoo import AECEnv

from app.domain.agents.rl_agent import RLAgent


@dataclass(frozen=True, slots=True)
class GymEpisodeResult:
    """Outcome of one Gymnasium episode, measured in learner decisions."""

    episode_return: float
    decisions: int
    terminated: bool
    truncated: bool
    illegal_action: bool
    final_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PettingZooEpisodeResult:
    """Per-agent outcome of one PettingZoo AEC episode."""

    episode_returns: dict[str, float]
    decisions_by_agent: dict[str, int]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    final_infos: dict[str, dict[str, Any]]


def run_gymnasium_episode(
    env: gym.Env,
    agent: RLAgent,
    *,
    seed: int,
    deterministic: bool = False,
    action_mask_key: str = "action_mask",
) -> GymEpisodeResult:
    """Run one seeded episode using an observation-based structured-action actor.

    The environment controls game and opponent randomness. The same episode seed
    is also passed to an actor that implements ``reset(seed)``. One decision is
    one Gymnasium step, which may internally advance multiple engine actions.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro.")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic deve ser booleano.")
    if not isinstance(action_mask_key, str) or not action_mask_key:
        raise ValueError("action_mask_key deve ser um texto não vazio.")

    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed)

    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    decisions = 0
    terminated = truncated = illegal_action = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        action_mask = observation[action_mask_key]
        action = agent.act(
            observation,
            action_mask,
            deterministic=deterministic,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        decisions += 1
        illegal_action = illegal_action or bool(info.get("illegal_action", False))

    return GymEpisodeResult(
        episode_return=episode_return,
        decisions=decisions,
        terminated=bool(terminated),
        truncated=bool(truncated),
        illegal_action=illegal_action,
        final_info=dict(info),
    )


def run_pettingzoo_episode(
    env: AECEnv,
    agents: Mapping[str, RLAgent],
    *,
    seed: int,
    deterministic: bool = False,
) -> PettingZooEpisodeResult:
    """Run one seeded multi-agent episode using observation-based actors.

    Agents are keyed by PettingZoo agent name. Each active agent sees its own
    observation and action mask on its turn; dead agents receive ``None`` as
    required by the AEC API. Returns and decision counts are tracked per seat.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro.")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic deve ser booleano.")

    possible_agents = tuple(env.possible_agents)
    missing = set(possible_agents) - set(agents)
    extra = set(agents) - set(possible_agents)
    if missing or extra:
        raise ValueError(
            f"Mapa de agentes incompatível; faltam={sorted(missing)}, "
            f"desconhecidos={sorted(extra)}."
        )

    seen_actor_ids: set[int] = set()
    for seat, name in enumerate(possible_agents):
        agent = agents[name]
        if not callable(getattr(agent, "act", None)):
            raise TypeError(f"O agente {name!r} precisa implementar act().")
        reset = getattr(agent, "reset", None)
        if callable(reset) and id(agent) not in seen_actor_ids:
            reset(seed * 1_000_003 + seat)
            seen_actor_ids.add(id(agent))

    env.reset(seed=seed)
    episode_returns = {name: 0.0 for name in possible_agents}
    decisions = {name: 0 for name in possible_agents}
    terminations = {name: False for name in possible_agents}
    truncations = {name: False for name in possible_agents}
    final_infos = {name: {} for name in possible_agents}

    for name in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()
        episode_returns[name] += float(reward)
        terminations[name] = bool(terminated)
        truncations[name] = bool(truncated)
        final_infos[name] = dict(info)
        if terminated or truncated:
            env.step(None)
            continue

        action = agents[name].act(
            observation,
            observation["action_mask"],
            deterministic=deterministic,
        )
        decisions[name] += 1
        env.step(action)

    return PettingZooEpisodeResult(
        episode_returns=episode_returns,
        decisions_by_agent=decisions,
        terminations=terminations,
        truncations=truncations,
        final_infos=final_infos,
    )
