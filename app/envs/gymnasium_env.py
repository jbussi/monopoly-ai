"""Ambiente Gymnasium de agente único: o aprendiz joga contra oponentes fixos.

Compatível com algoritmos que usam máscara via `env.action_masks()` (por exemplo
`MaskablePPO` do sb3-contrib). A máscara também vem em `info["action_mask"]`.

    from app.envs import MonopolyGymEnv
    from app.envs.baselines import GreedyPolicy
    env = MonopolyGymEnv(num_players=2, opponents=[GreedyPolicy()])
"""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.application.game_setup import create_game, load_selected_game
from app.envs.action_space import DiscreteActionSpace
from app.envs.baselines import Policy, RandomPolicy
from app.envs.encoding import ObservationEncoder
from app.envs.rewards import RewardFn, win_loss_reward
from app.envs.text import render_text
from app.infraestructure.game_config.models import GameConfig


class MonopolyGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_players: int = 2,
        opponents: Sequence[Policy] | None = None,
        learner_seat: int | None = None,
        max_players: int = 4,
        max_turns: int = 1000,
        auction_enabled: bool | None = None,
        config: GameConfig | None = None,
        reward_fn: RewardFn = win_loss_reward,
        render_mode: str | None = None,
        on_illegal: str = "raise",
        illegal_reward: float = -1.0,
    ) -> None:
        """
        opponents: uma política por oponente (num_players - 1). Padrão: aleatórias.
        learner_seat: assento fixo do aprendiz; None sorteia a cada reset.
        on_illegal: "raise" lança ValueError; "terminate" encerra o episódio com
            `illegal_reward` (para algoritmos que não usam máscara).
        """
        if not 2 <= num_players <= max_players:
            raise ValueError("Precisa de 2 <= num_players <= max_players.")
        if opponents is None:
            opponents = [RandomPolicy() for _ in range(num_players - 1)]
        if len(opponents) != num_players - 1:
            raise ValueError("Passe exatamente num_players - 1 oponentes.")
        if on_illegal not in ("raise", "terminate"):
            raise ValueError("on_illegal deve ser 'raise' ou 'terminate'.")

        self.num_players = num_players
        self.opponents = list(opponents)
        self.fixed_seat = learner_seat
        self.max_turns = max_turns
        self.auction_enabled = auction_enabled
        self.config = config or load_selected_game()
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.on_illegal = on_illegal
        self.illegal_reward = illegal_reward

        self.loop = self._new_game(seed=None)
        self.action_space_map = DiscreteActionSpace(self.loop.game.board)
        self.encoder = ObservationEncoder(self.loop, max_players, max_turns)
        self.observation_space = spaces.Box(0.0, np.inf, (self.encoder.size,), np.float32)
        self.action_space = spaces.Discrete(self.action_space_map.size)
        self.learner_id = self.loop.game.turn_order[0]
        self._opponent_of: dict = {}

    def _new_game(self, seed: int | None):
        return create_game(
            [f"player_{i}" for i in range(self.num_players)],
            self.config,
            seed=seed,
            max_turns=self.max_turns,
            auction_enabled=self.auction_enabled,
            record_events=False,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.loop = self._new_game(int(self.np_random.integers(2**31)))
        order = self.loop.game.turn_order
        seat = self.fixed_seat if self.fixed_seat is not None else int(self.np_random.integers(self.num_players))
        self.learner_id = order[seat]
        others = [player_id for player_id in order if player_id != self.learner_id]
        self._opponent_of = dict(zip(others, self.opponents))
        for opponent in self.opponents:
            if hasattr(opponent, "reset"):
                opponent.reset(int(self.np_random.integers(2**31)))

        self._play_opponents()
        return self._obs(), self._info()

    def step(self, action: int):
        mask = self.action_masks()
        if not mask[int(action)]:
            if self.on_illegal == "terminate":
                return self._obs(), self.illegal_reward, True, False, {**self._info(), "illegal_action": True}
            raise ValueError(f"Ação ilegal {action} na fase {self.loop.turn_phase}.")

        self.loop.process_action(self.action_space_map.to_game_action(int(action), self.loop))
        self._play_opponents()

        game = self.loop.game
        reward = float(self.reward_fn(self.loop, self.learner_id))
        terminated = game.player_states[self.learner_id].is_bankrupt or game.winner_id is not None
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, game.truncated, self._info()

    def action_masks(self) -> np.ndarray:
        loop = self.loop
        if loop.game.game_over or loop.current_decision_player_id != self.learner_id:
            return np.zeros(self.action_space_map.size, dtype=bool)
        return self.action_space_map.mask(loop).astype(bool)

    def render(self) -> str | None:
        text = render_text(self.loop)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def _play_opponents(self) -> None:
        loop = self.loop
        game = loop.game
        while (
            not game.game_over
            and not game.player_states[self.learner_id].is_bankrupt
            and loop.current_decision_player_id != self.learner_id
        ):
            loop.process_action(self._opponent_of[loop.current_decision_player_id](loop))

    def _obs(self) -> np.ndarray:
        return self.encoder.encode(self.loop, self.learner_id)

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks().astype(np.int8),
            "seat": self.loop.game.turn_order.index(self.learner_id),
        }
