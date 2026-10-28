"""Ambiente PettingZoo AEC (multiagente, turnos alternados).

    from app.envs import pettingzoo_env
    env = pettingzoo_env.env(num_players=4)
    env.reset(seed=0)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        action = None if terminated or truncated else choose(obs["observation"], obs["action_mask"])
        env.step(action)

Segue a convenção dos jogos "classic" do PettingZoo: a observação é um dict com
`observation` (float32) e `action_mask` (int8). Um turno tem várias decisões do mesmo
agente (rolar, comprar, construir, encerrar), e `agent_selection` sempre aponta para quem
decide agora: num leilão ou numa dívida, pode ser alguém fora do próprio turno.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from app.application.game_setup import create_game, load_selected_game
from app.infraestructure.game_config.models import GameConfig
from app.envs.action_space import DiscreteActionSpace
from app.envs.encoding import ObservationEncoder
from app.envs.rewards import RewardFn, win_loss_reward
from app.envs.text import render_text


def env(**kwargs: Any) -> AECEnv:
    """Ambiente com os wrappers padrão do PettingZoo (validam ordem das chamadas e ações)."""
    e = raw_env(**kwargs)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class raw_env(AECEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "monopoly_v1", "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 4,
        max_players: int = 4,
        max_turns: int = 1000,
        auction_enabled: bool | None = None,
        config: GameConfig | None = None,
        reward_fn: RewardFn = win_loss_reward,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 2 <= num_players <= max_players:
            raise ValueError("Precisa de 2 <= num_players <= max_players.")

        self.num_players = num_players
        self.max_turns = max_turns
        self.auction_enabled = auction_enabled
        self.config = config or load_selected_game()
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        self.loop = self._new_game(seed=None)
        self.action_space_map = DiscreteActionSpace(self.loop.game.board)
        self.encoder = ObservationEncoder(self.loop, max_players, max_turns)

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        obs_space = spaces.Dict({
            "observation": spaces.Box(0.0, np.inf, (self.encoder.size,), np.float32),
            "action_mask": spaces.Box(0, 1, (self.action_space_map.size,), np.int8),
        })
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(self.action_space_map.size) for a in self.possible_agents}

    def _new_game(self, seed: int | None):
        return create_game(
            [f"player_{i}" for i in range(self.num_players)],
            self.config,
            seed=seed,
            max_turns=self.max_turns,
            auction_enabled=self.auction_enabled,
            record_events=False,
        )

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def _player_id(self, agent: str):
        return self.loop.game.turn_order[self.possible_agents.index(agent)]

    def _agent(self, player_id) -> str:
        return self.possible_agents[self.loop.game.turn_order.index(player_id)]

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        player_id = self._player_id(agent)
        loop = self.loop
        if loop.game.game_over or loop.current_decision_player_id != player_id:
            mask = np.zeros(self.action_space_map.size, dtype=np.int8)
        else:
            mask = self.action_space_map.mask(loop)
        return {"observation": self.encoder.encode(loop, player_id), "action_mask": mask}

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.loop = self._new_game(seed)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self._agent(self.loop.current_decision_player_id)

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        game_action = self.action_space_map.to_game_action(int(action), self.loop)
        result = self.loop.process_action(game_action)
        if result["status"] != "SUCCESS":
            raise ValueError(f"Ação ilegal {action} ({game_action.action_type}): {result['reason']}")

        game = self.loop.game
        for a in self.agents:
            player_id = self._player_id(a)
            self.rewards[a] = float(self.reward_fn(self.loop, player_id))
            self.terminations[a] = game.player_states[player_id].is_bankrupt or game.winner_id is not None
            self.truncations[a] = game.truncated

        if not game.game_over:
            self.agent_selection = self._agent(self.loop.current_decision_player_id)
        self._accumulate_rewards()
        # Agentes eliminados recebem step(None) antes do próximo agente ativo
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
        pass
