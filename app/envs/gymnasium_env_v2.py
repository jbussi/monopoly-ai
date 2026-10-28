"""Gymnasium single-agent adapter for the full structured-action environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.application.game_setup import create_game, load_selected_game
from app.envs.action_space_v2 import StructuredActionSpace
from app.envs.baselines import Policy, RandomPolicy
from app.envs.observation_v2 import ObservationEncoderV2
from app.envs.rewards import RewardFn, win_loss_reward
from app.envs.text import render_text
from app.infraestructure.game_config.models import GameConfig


class MonopolyGymEnvV2(gym.Env):
    """A learner plays against fixed policies with full trades and debt resolution."""

    metadata = {"render_modes": ["human", "ansi"], "name": "monopoly_v2"}

    def __init__(
        self,
        num_players: int = 2,
        opponents: Sequence[Policy] | None = None,
        learner_seat: int | None = None,
        max_players: int = 12,
        max_turns: int = 1000,
        config: GameConfig | None = None,
        reward_fn: RewardFn = win_loss_reward,
        render_mode: str | None = None,
        on_illegal: str = "raise",
        illegal_reward: float = -1.0,
    ) -> None:
        if not 2 <= num_players <= max_players <= 12:
            raise ValueError("Precisa de 2 <= num_players <= max_players <= 12.")
        if opponents is None:
            opponents = [RandomPolicy() for _ in range(num_players - 1)]
        if len(opponents) != num_players - 1:
            raise ValueError("Passe exatamente num_players - 1 oponentes.")
        if on_illegal not in ("raise", "terminate"):
            raise ValueError("on_illegal deve ser 'raise' ou 'terminate'.")
        if learner_seat is not None and not 0 <= learner_seat < num_players:
            raise ValueError("learner_seat deve identificar um assento da partida.")

        self.num_players = num_players
        self.opponents = list(opponents)
        self.fixed_seat = learner_seat
        self.max_players = max_players
        self.max_turns = max_turns
        self.config = config or load_selected_game()
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.on_illegal = on_illegal
        self.illegal_reward = illegal_reward
        self._setup_episode(seed=None)
        self.action_space = self.action_map.action_space
        self.observation_space = spaces.Dict({
            **self.encoder.observation_space.spaces,
            "action_mask": self.action_map.mask_space,
        })

    def _new_game(self, seed: int | None):
        return create_game(
            [f"player_{i}" for i in range(self.num_players)],
            self.config,
            seed=seed,
            max_turns=self.max_turns,
            record_events=False,
        )

    def _setup_episode(self, seed: int | None) -> None:
        self.loop = self._new_game(seed)
        self.action_map = StructuredActionSpace(
            self.loop, self.max_players, loop_provider=lambda: self.loop
        )
        self.encoder = ObservationEncoderV2(
            self.loop, self.max_players, self.max_turns, self.action_map.jail_card_ids
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(2**31))
        self._setup_episode(seed=episode_seed)
        order = self.loop.game.turn_order
        seat = self.fixed_seat if self.fixed_seat is not None else int(self.np_random.integers(self.num_players))
        self.learner_id = order[seat]
        others = [player_id for player_id in order if player_id != self.learner_id]
        self._opponent_of = dict(zip(others, self.opponents))
        for opponent in self.opponents:
            if hasattr(opponent, "reset"):
                opponent.reset(int(self.np_random.integers(2**31)))
        self._play_opponents()
        return self._observation(), self._info()

    def step(self, action: dict[str, Any]):
        try:
            game_action = self.action_map.to_game_action(action, self.loop)
            result = self.loop.process_action(game_action)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return self._illegal_step(str(exc))
        if result["status"] != "SUCCESS":
            return self._illegal_step(result["reason"])

        self._play_opponents()
        game = self.loop.game
        reward = float(self.reward_fn(self.loop, self.learner_id))
        terminated = game.player_states[self.learner_id].is_bankrupt or game.winner_id is not None
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, game.truncated, self._info()

    def _illegal_step(self, reason: str):
        if self.on_illegal == "raise":
            raise ValueError(f"Ação ilegal na fase {self.loop.turn_phase}: {reason}")
        return (
            self._observation(), self.illegal_reward, True, False,
            {**self._info(), "illegal_action": True, "illegal_reason": reason},
        )

    def action_masks(
        self, *, action_type: str | int | None = None, target_player_index: int | None = None,
        money_offered: int | None = None, money_requested: int | None = None,
    ):
        """Conditional masks for autoregressive policies; see observation['action_mask']."""
        if self.loop.game.game_over or self.loop.current_decision_player_id != self.learner_id:
            mask = self.action_map.action_mask(self.loop)
            for values in mask.values():
                values.fill(0)
            return mask
        return self.action_map.action_mask(
            self.loop,
            action_type=action_type,
            target_player_index=target_player_index,
            money_offered=money_offered,
            money_requested=money_requested,
        )

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
            player_id = loop.current_decision_player_id
            result = loop.process_action(self._opponent_of[player_id](loop))
            if result["status"] != "SUCCESS":
                raise RuntimeError(f"A política oponente produziu uma ação inválida: {result}")

    def _observation(self) -> dict[str, Any]:
        state = self.encoder.encode(self.loop, self.learner_id)
        if self.loop.game.game_over or self.loop.current_decision_player_id != self.learner_id:
            mask = self.action_map.action_mask(self.loop)
            for values in mask.values():
                values.fill(0)
        else:
            mask = self.action_map.action_mask(self.loop)
        return {**state, "action_mask": mask}

    def _info(self) -> dict[str, Any]:
        return {
            "seat": self.loop.game.turn_order.index(self.learner_id),
            "current_actor_seat": self.loop.game.turn_order.index(self.loop.current_decision_player_id),
        }
