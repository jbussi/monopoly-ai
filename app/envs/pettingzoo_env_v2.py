"""PettingZoo AEC adapter for Monopoly v2 and the full game engine."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from app.application.game_setup import create_game, load_selected_game
from app.envs.action_space_v2 import StructuredActionSpace
from app.envs.observation_v2 import ObservationEncoderV2
from app.envs.rewards import RewardFn, win_loss_reward
from app.envs.text import render_text
from app.infraestructure.game_config.models import GameConfig


def env(**kwargs: Any) -> AECEnv:
    e = raw_env(**kwargs)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class raw_env(AECEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "monopoly_v2", "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 4,
        max_players: int = 12,
        max_turns: int = 1000,
        config: GameConfig | None = None,
        reward_fn: RewardFn = win_loss_reward,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 2 <= num_players <= max_players <= 12:
            raise ValueError("Precisa de 2 <= num_players <= max_players <= 12.")
        self.num_players = num_players
        self.max_players = max_players
        self.max_turns = max_turns
        self.config = config or load_selected_game()
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.loop = self._new_game(seed=None)
        self.action_map = StructuredActionSpace(
            self.loop, max_players, loop_provider=lambda: self.loop
        )
        self.encoder = ObservationEncoderV2(
            self.loop, max_players, max_turns, self.action_map.jail_card_ids
        )
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.observation_spaces = {
            agent: spaces.Dict({
                **self.encoder.observation_space.spaces,
                "action_mask": self.action_map.mask_space,
            })
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: self.action_map.action_space for agent in self.possible_agents}

    def _new_game(self, seed: int | None):
        return create_game(
            [f"player_{i}" for i in range(self.num_players)],
            self.config, seed=seed, max_turns=self.max_turns, record_events=False,
        )

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def _player_id(self, agent: str):
        return self.loop.game.turn_order[self.possible_agents.index(agent)]

    def _agent(self, player_id):
        return self.possible_agents[self.loop.game.turn_order.index(player_id)]

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.loop = self._new_game(seed)
        self.action_map = StructuredActionSpace(
            self.loop, self.max_players, loop_provider=lambda: self.loop
        )
        self.encoder = ObservationEncoderV2(
            self.loop, self.max_players, self.max_turns, self.action_map.jail_card_ids
        )
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self._agent(self.loop.current_decision_player_id)

    def observe(self, agent: str) -> dict[str, Any]:
        player_id = self._player_id(agent)
        state = self.encoder.encode(self.loop, player_id)
        if self.loop.game.game_over or self.loop.current_decision_player_id != player_id:
            mask = self.action_map.action_mask(self.loop)
            for values in mask.values():
                values.fill(0)
        else:
            mask = self.action_map.action_mask(self.loop)
        return {**state, "action_mask": mask}

    def action_mask(
        self,
        agent: str,
        *,
        action_type: str | int | None = None,
        target_player_index: int | None = None,
        money_offered: int | None = None,
        money_requested: int | None = None,
    ):
        """Return the acting agent's conditional masks for structured action fields."""
        player_id = self._player_id(agent)
        mask = self.action_map.action_mask(
            self.loop,
            action_type=action_type,
            target_player_index=target_player_index,
            money_offered=money_offered,
            money_requested=money_requested,
        )
        if player_id != self.loop.current_decision_player_id:
            for values in mask.values():
                values.fill(0)
        return mask

    def step(self, action: dict[str, Any] | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("Agentes ativos devem enviar uma ação estruturada.")
        self._cumulative_rewards[agent] = 0.0
        game_action = self.action_map.to_game_action(action, self.loop)
        result = self.loop.process_action(game_action)
        if result["status"] != "SUCCESS":
            raise ValueError(f"Ação ilegal ({game_action.action_type}): {result['reason']}")

        game = self.loop.game
        for name in self.agents:
            player_id = self._player_id(name)
            self.rewards[name] = float(self.reward_fn(self.loop, player_id))
            self.terminations[name] = game.player_states[player_id].is_bankrupt or game.winner_id is not None
            self.truncations[name] = game.truncated
        if not game.game_over:
            self.agent_selection = self._agent(self.loop.current_decision_player_id)
        self._accumulate_rewards()
        self._deads_step_first()
        if self.render_mode == "human":
            self.render()

    def render(self) -> str | None:
        text = render_text(self.loop)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        return None
