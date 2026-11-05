"""Discrete candidate-action wrapper for structured Monopoly Gymnasium v2."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatten, flatten_space

from app.domain.analytics.decision_context import DecisionContext
from app.domain.analytics.decision_features import DecisionFeaturesEncoder
from app.domain.agents.trade_candidates import LegalCandidateProvider
from app.domain.game.actions import ActionType, GameAction
from app.envs.action_space_v2 import ACTION_TYPES, ACTION_TYPE_TO_INDEX
from app.envs.gymnasium_env_v2 import MonopolyGymEnvV2


class CandidateSelectionEnv(gym.Wrapper):
    """Expose one categorical choice over a bounded set of legal actions.

    Each observation contains padded action features and ``candidate_mask``.
    Actions are indices into that set; the selected ``GameAction`` is converted
    back through the underlying structured action map before the game advances.
    This format is suitable for masked categorical policies such as PPO.
    """

    def __init__(
        self,
        env: MonopolyGymEnvV2,
        *,
        max_candidates: int = 128,
        max_trade_candidates: int = 4,
        candidate_analysis_simulations: int = 1,
        candidate_analysis_horizons: tuple[int, ...] = (1, 2),
        trade_analysis_mode: str = "on_demand",
    ) -> None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates deve ser um inteiro positivo.")
        if (
            isinstance(max_trade_candidates, bool)
            or not isinstance(max_trade_candidates, int)
            or not 0 <= max_trade_candidates <= max_candidates
        ):
            raise ValueError("max_trade_candidates deve estar entre 0 e max_candidates.")
        if (
            isinstance(candidate_analysis_simulations, bool)
            or not isinstance(candidate_analysis_simulations, int)
            or candidate_analysis_simulations < 1
        ):
            raise ValueError("candidate_analysis_simulations deve ser um inteiro positivo.")
        if not candidate_analysis_horizons or any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
            for horizon in candidate_analysis_horizons
        ):
            raise ValueError("candidate_analysis_horizons deve conter inteiros positivos.")
        if trade_analysis_mode not in {"disabled", "on_demand", "automatic"}:
            raise ValueError(
                "trade_analysis_mode deve ser 'disabled', 'on_demand' ou 'automatic'."
            )
        super().__init__(env)
        self.max_candidates = max_candidates
        self.max_trade_candidates = max_trade_candidates
        self.candidate_analysis_simulations = candidate_analysis_simulations
        self.candidate_analysis_horizons = tuple(sorted(set(candidate_analysis_horizons)))
        self.trade_analysis_mode = trade_analysis_mode
        self.candidate_provider = LegalCandidateProvider(
            max_candidates=max_candidates,
            max_trade_candidates=max_trade_candidates,
        )
        self.candidate_actions: tuple[GameAction, ...] = ()
        self.candidate_mask = np.zeros(max_candidates, dtype=np.int8)
        self.candidate_action_mask = np.zeros(max_candidates + 1, dtype=np.int8)
        self.money_scale = float(max(1, env.loop.rules.starting_money, env.loop.rules.go_salary))
        self._episode_seed = 0
        self.stage_timings = {"candidate_encoding_seconds": 0.0, "trade_analysis_seconds": 0.0}
        self._analysis_done = False
        self._base_observation: dict[str, Any] = {}
        self.decision_encoder = self._make_decision_encoder(
            env, self.candidate_analysis_horizons
        )
        self.decision_feature_space = flatten_space(self.decision_encoder.observation_space)
        self.decision_feature_size = int(np.prod(self.decision_feature_space.shape))
        self.candidate_analysis_mask = np.zeros(max_candidates, dtype=np.int8)
        self.action_feature_size = (
            len(ACTION_TYPES)
            + (env.max_players + 1)
            + (env.action_map.board_size + 1)
            + 3
            + 2 * len(env.action_map.tile_ids)
            + 2 * len(env.action_map.jail_card_ids)
        )
        self.analysis_action_index = max_candidates
        self.action_space = spaces.Discrete(max_candidates + 1)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "candidate_actions": spaces.Box(
                -np.inf,
                np.inf,
                shape=(max_candidates, self.action_feature_size),
                dtype=np.float32,
            ),
            "candidate_mask": spaces.MultiBinary(max_candidates),
            "candidate_action_mask": spaces.MultiBinary(max_candidates + 1),
            "candidate_decision_features": spaces.Box(
                -np.inf,
                np.inf,
                shape=(max_candidates, self.decision_feature_size),
                dtype=np.float32,
            ),
            "candidate_analysis_mask": spaces.MultiBinary(max_candidates),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the wrapped episode and generate its first legal candidates."""
        observation, info = self.env.reset(seed=seed, options=options)
        self.stage_timings = {"candidate_encoding_seconds": 0.0, "trade_analysis_seconds": 0.0}
        self._episode_seed = 0 if seed is None else int(seed)
        self.candidate_provider.reset(seed)
        self.decision_encoder = self._make_decision_encoder(
            self.env, self.candidate_analysis_horizons
        )
        self._analysis_done = False
        return self._with_candidates(observation), info

    def step(self, action: int):
        """Select a candidate or request trade features without a game move."""
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("A ação deve ser um índice inteiro de candidata ou consulta.")
        index = int(action)
        if index == self.analysis_action_index:
            if not self.candidate_action_mask[index]:
                raise ValueError("A análise de trocas não está disponível nesta decisão.")
            self._analyze_trade_candidates()
            return (
                self._with_candidates(self._base_observation, refresh_candidates=False),
                0.0,
                False,
                False,
                {"analysis_requested": True},
            )
        if not 0 <= index < self.max_candidates or not self.candidate_mask[index]:
            raise ValueError(f"Índice de candidata ilegal: {index}.")
        selected = self.candidate_actions[index]
        structured_action = self.env.action_map.from_game_action(selected, self.env.loop)
        observation, reward, terminated, truncated, info = self.env.step(structured_action)
        self._analysis_done = False
        return (
            self._with_candidates(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        """Return a bool mask compatible with masked categorical algorithms."""
        return self.candidate_action_mask.astype(bool, copy=True)

    def _with_candidates(
        self,
        observation: dict[str, Any],
        *,
        refresh_candidates: bool = True,
    ) -> dict[str, Any]:
        started = perf_counter()
        loop = self.env.loop
        self._ensure_decision_encoder_matches_game(loop)
        if refresh_candidates:
            is_learner_decision = (
                not loop.game.game_over
                and loop.current_decision_player_id == self.env.learner_id
            )
            candidates = self.candidate_provider(loop) if is_learner_decision else ()
            if len(candidates) > self.max_candidates:
                raise ValueError(
                    f"Foram geradas {len(candidates)} candidatas; o limite é {self.max_candidates}."
                )
            self.candidate_actions = tuple(candidates)
            self.candidate_mask.fill(0)
            self.candidate_mask[:len(candidates)] = 1
            self.candidate_analysis_mask.fill(0)
            self._analysis_done = False
            self._base_observation = observation

        features = np.zeros((self.max_candidates, self.action_feature_size), dtype=np.float32)
        for row, candidate in enumerate(self.candidate_actions):
            features[row] = self._encode_action(candidate)

        decision_features = np.zeros(
            (self.max_candidates, self.decision_feature_size), dtype=np.float32
        )
        if self.trade_analysis_mode == "automatic" and not self._analysis_done:
            decision_features = self._analyze_trade_candidates()
        elif self._analysis_done:
            decision_features = self._candidate_decision_features

        self.candidate_action_mask.fill(0)
        self.candidate_action_mask[:len(self.candidate_actions)] = 1
        if (
            self.trade_analysis_mode == "on_demand"
            and not self._analysis_done
            and any(self._needs_trade_analysis(loop, candidate) for candidate in self.candidate_actions)
        ):
            self.candidate_action_mask[self.analysis_action_index] = 1
        result = {
            **observation,
            "candidate_actions": features,
            "candidate_mask": self.candidate_mask.copy(),
            "candidate_action_mask": self.candidate_action_mask.copy(),
            "candidate_decision_features": decision_features,
            "candidate_analysis_mask": self.candidate_analysis_mask.copy(),
        }
        self.stage_timings["candidate_encoding_seconds"] += perf_counter() - started
        return result

    def _ensure_decision_encoder_matches_game(self, loop) -> None:
        """Refresh the context encoder if the wrapped environment changed games."""
        turn_order = tuple(loop.game.turn_order)
        if self.decision_encoder.turn_order == turn_order:
            return
        self.decision_encoder = self._make_decision_encoder(
            self.env, self.candidate_analysis_horizons
        )
        self.decision_feature_space = flatten_space(
            self.decision_encoder.observation_space
        )
        self.decision_feature_size = int(np.prod(self.decision_feature_space.shape))
        self.candidate_analysis_mask.fill(0)
        self._candidate_decision_features = np.zeros(
            (self.max_candidates, self.decision_feature_size), dtype=np.float32
        )
        self._analysis_done = False

    def _analyze_trade_candidates(self) -> np.ndarray:
        started = perf_counter()
        loop = self.env.loop
        result = np.zeros(
            (self.max_candidates, self.decision_feature_size), dtype=np.float32
        )
        trade_analysis_cache: DecisionContext | None = None
        for row, candidate in enumerate(self.candidate_actions):
            if not self._needs_trade_analysis(loop, candidate):
                continue
            if (
                candidate.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
                and trade_analysis_cache is not None
            ):
                context = replace(trade_analysis_cache, candidate_action=candidate)
            else:
                context = self.env.evaluate_candidate(
                    candidate,
                    horizons=self.candidate_analysis_horizons,
                    valuation_horizons=self.candidate_analysis_horizons,
                    simulations=self.candidate_analysis_simulations,
                    seed=self._analysis_seed(loop, row),
                )
                if candidate.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
                    trade_analysis_cache = context
            if context.player_id not in loop.game.player_states:
                raise ValueError(
                    "A análise de candidata retornou um jogador de outra partida "
                    f"(jogador={context.player_id}, jogadores={tuple(loop.game.turn_order)})."
                )
            encoded = self.decision_encoder.encode(context)
            result[row] = flatten(
                self.decision_encoder.observation_space,
                encoded,
            ).astype(np.float32, copy=False)
            self.candidate_analysis_mask[row] = 1
        self._analysis_done = True
        self._candidate_decision_features = result
        self.stage_timings["trade_analysis_seconds"] += perf_counter() - started
        return result

    @staticmethod
    def _make_decision_encoder(
        env: MonopolyGymEnvV2,
        horizons: tuple[int, ...],
    ) -> DecisionFeaturesEncoder:
        return DecisionFeaturesEncoder(
            env.loop,
            max_players=env.max_players,
            max_alternatives=4,
            horizons=horizons,
        )

    def _needs_trade_analysis(self, loop, candidate: GameAction) -> bool:
        return (
            self.trade_analysis_mode != "disabled"
            and not loop.game.debts
            and candidate.action_type in {
                ActionType.PROPOSE_TRADE,
                ActionType.ACCEPT_TRADE,
                ActionType.REJECT_TRADE,
                ActionType.COUNTER_TRADE,
            }
        )

    def _analysis_seed(self, loop, candidate_index: int) -> int:
        return (
            self._episode_seed * 1_000_003
            + loop.game.turn_counter * 97_409
            + candidate_index * 65_537
        ) % (2**63)

    def _encode_action(self, candidate: GameAction) -> np.ndarray:
        loop = self.env.loop
        structured = self.env.action_map.from_game_action(candidate, loop)
        action_type = np.zeros(len(ACTION_TYPES), dtype=np.float32)
        action_type[ACTION_TYPE_TO_INDEX[candidate.action_type]] = 1.0
        target = np.zeros(self.env.max_players + 1, dtype=np.float32)
        target[int(structured["target_player"])] = 1.0
        tile = np.zeros(loop.game.board.size + 1, dtype=np.float32)
        tile[int(structured["tile_id"])] = 1.0
        amounts = np.asarray([
            structured["amount"][0],
            structured["money_offered"][0],
            structured["money_requested"][0],
        ], dtype=np.float32) / self.money_scale
        return np.concatenate((
            action_type,
            target,
            tile,
            amounts,
            np.asarray(structured["offered_properties"], dtype=np.float32),
            np.asarray(structured["requested_properties"], dtype=np.float32),
            np.asarray(structured["offered_jail_cards"], dtype=np.float32),
            np.asarray(structured["requested_jail_cards"], dtype=np.float32),
        ))
