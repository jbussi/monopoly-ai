"""Conditional property opportunities when choosing how to leave jail."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

import numpy as np

from app.domain.analytics.economics.game_valuation import (
    PropertyAcquisitionValuation,
    value_property_acquisition,
)
from app.domain.analytics.markov.stationary_board import (
    calculate_stationary_board_analysis,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.baselines import GreedyPolicy
from app.infraestructure.game_config.models import GameConfig


DEFAULT_JAIL_HORIZONS = (1, 3, 5)
DEFAULT_JAIL_SIMULATIONS = 64
_JAIL_CHOICES = (
    ActionType.PAY_JAIL_FINE,
    ActionType.USE_JAIL_CARD,
    ActionType.ROLL_DICE,
)


@dataclass(frozen=True, slots=True)
class ConditionalPropertyOpportunity:
    tile_id: int
    name: str
    probability_offered_to_player: float
    probability_bought_by_greedy_policy: float
    probability_bought_if_offered: float
    acquisition_valuation: PropertyAcquisitionValuation
    greedy_weighted_positive_net_value: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class JailChoiceHorizonForecast:
    action_type: str
    horizon_rounds: int
    requested_turns: int
    mean_cash_delta: float
    std_cash_delta: float
    p10_cash_delta: float
    median_cash_delta: float
    p90_cash_delta: float
    bankruptcy_probability: float
    in_jail_probability: float
    probability_any_property_offered: float
    mean_distinct_properties_offered: float
    property_opportunities: tuple[ConditionalPropertyOpportunity, ...]


@dataclass(frozen=True, slots=True)
class JailChoiceForecast:
    action_type: str
    horizons: tuple[JailChoiceHorizonForecast, ...]

    def for_rounds(self, rounds: int) -> JailChoiceHorizonForecast:
        for forecast in self.horizons:
            if forecast.horizon_rounds == rounds:
                return forecast
        raise KeyError(f"Horizonte não calculado: {rounds} rodada(s).")


@dataclass(frozen=True, slots=True)
class JailOpportunityAnalysis:
    player_id: UUID
    seed: int
    simulations: int
    starting_cash: int
    valuation_horizons: tuple[int, ...]
    choices: tuple[JailChoiceForecast, ...]

    def for_choice(self, action_type: str | ActionType) -> JailChoiceForecast:
        action_name = str(action_type)
        for choice in self.choices:
            if choice.action_type == action_name:
                return choice
        raise KeyError(f"Opção de prisão não analisada: {action_name}.")


def analyze_jail_exit_opportunities(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    *,
    horizons: Iterable[int] = DEFAULT_JAIL_HORIZONS,
    simulations: int = DEFAULT_JAIL_SIMULATIONS,
    seed: int | None = None,
    greedy_reserve: int = 150,
    valuation_horizons: Iterable[int] = (1, 3, 5),
    discount_rate_per_round: float | None = None,
) -> JailOpportunityAnalysis:
    """Compare jail choices by simulating their conditional property access.

    The requested action is forced once for the jailed player. Every action
    after that, for every player, follows ``GreedyPolicy``. For each currently
    unowned property the result estimates (a) the chance the player receives
    the purchase offer and (b) the chance the greedy policy can and does buy it.
    Property values are measured from the current game state and kept separate
    from access probabilities so an agent can combine them without a fixed
    weighting rule.
    """
    horizon_values = _validate_positive_ints("horizons", horizons)
    valuation_horizon_values = _validate_positive_ints(
        "valuation_horizons", valuation_horizons
    )
    _validate_positive_int("simulations", simulations)
    _validate_positive_int("greedy_reserve", greedy_reserve, allow_zero=True)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed deve ser um inteiro ou None.")
    if discount_rate_per_round is not None:
        if isinstance(discount_rate_per_round, bool) or not isinstance(
            discount_rate_per_round, (int, float)
        ):
            raise TypeError("discount_rate_per_round deve ser numérico.")
        if not np.isfinite(discount_rate_per_round) or discount_rate_per_round <= -1:
            raise ValueError("discount_rate_per_round deve ser finito e maior que -1.")

    if loop.turn_phase != GameLoop.BEFORE_ROLL:
        raise ValueError("A análise de saída da prisão exige a fase BEFORE_ROLL.")
    if loop.current_decision_player_id != player_id:
        raise ValueError("O jogador analisado precisa ser o decisor atual.")
    state = loop.game.player_states.get(player_id)
    if state is None:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    if not state.is_in_jail or state.is_bankrupt:
        raise ValueError("O jogador analisado precisa estar preso e ativo.")

    legal_choices = {
        action.action_type
        for action in loop.legal_actions()
        if action.action_type in _JAIL_CHOICES
    }
    if ActionType.ROLL_DICE not in legal_choices:
        raise ValueError("A rolagem de dados não está disponível para o jogador preso.")

    effective_seed = random.SystemRandom().getrandbits(64) if seed is None else seed
    base_rng = random.Random(effective_seed)
    scenario_seeds = [base_rng.getrandbits(64) for _ in range(simulations)]
    starting_cash = state.money
    active_players_at_start = max(1, len(loop.game.active_player_ids))
    start_turn = loop.game.turn_counter
    target_turns = {
        rounds: start_turn + rounds * active_players_at_start
        for rounds in horizon_values
    }
    action_limit = max(
        256,
        (max(target_turns.values()) - start_turn) * 128,
    )

    stationary = calculate_stationary_board_analysis(config, loop.rules)
    unowned_tiles = tuple(
        tile
        for tile in loop.game.board.property_tiles
        if loop.game.property_states[tile.tile_id].owner_id is None
    )
    valuations = {
        tile.tile_id: value_property_acquisition(
            loop,
            config,
            player_id,
            tile.tile_id,
            horizons=valuation_horizon_values,
            discount_rate_per_round=discount_rate_per_round,
            stationary=stationary,
        )
        for tile in unowned_tiles
    }

    choice_results: list[JailChoiceForecast] = []
    for choice in _JAIL_CHOICES:
        if choice not in legal_choices:
            continue
        snapshots: dict[int, list[dict[str, object]]] = {
            rounds: [] for rounds in horizon_values
        }

        for scenario_seed in scenario_seeds:
            scenario = loop.clone()
            scenario.game.record_events = False
            scenario.game.history_logs.clear()
            _reseed_scenario(scenario, random.Random(scenario_seed))
            policies = {
                candidate: GreedyPolicy(reserve=greedy_reserve)
                for candidate in scenario.game.turn_order
            }
            offered: set[int] = set()
            bought: set[int] = set()
            recorded: set[int] = set()
            actions = 0

            initial_action = GameAction(choice)
            _process_and_record(
                scenario,
                player_id,
                initial_action,
                offered,
                bought,
                valuations,
            )
            actions += 1

            while len(recorded) < len(horizon_values):
                for rounds in horizon_values:
                    if rounds in recorded:
                        continue
                    if (
                        scenario.game.game_over
                        or scenario.game.turn_counter >= target_turns[rounds]
                    ):
                        player_state = scenario.game.player_states[player_id]
                        snapshots[rounds].append({
                            "cash": player_state.money,
                            "bankrupt": player_state.is_bankrupt,
                            "in_jail": player_state.is_in_jail,
                            "offered": frozenset(offered),
                            "bought": frozenset(bought),
                        })
                        recorded.add(rounds)

                if len(recorded) == len(horizon_values):
                    break
                if actions >= action_limit:
                    raise RuntimeError(
                        "A simulação da prisão excedeu o limite de ações "
                        "antes de alcançar o horizonte."
                    )

                actor = scenario.current_decision_player_id
                action = policies[actor](scenario)
                _process_and_record(
                    scenario,
                    player_id,
                    action,
                    offered,
                    bought,
                    valuations,
                )
                actions += 1

        choice_horizons = tuple(
            _summarize_choice_horizon(
                choice=choice,
                rounds=rounds,
                requested_turns=rounds * active_players_at_start,
                starting_cash=starting_cash,
                snapshots=snapshots[rounds],
                valuations=valuations,
            )
            for rounds in horizon_values
        )
        choice_results.append(
            JailChoiceForecast(str(choice), choice_horizons)
        )

    return JailOpportunityAnalysis(
        player_id=player_id,
        seed=effective_seed,
        simulations=simulations,
        starting_cash=starting_cash,
        valuation_horizons=valuation_horizon_values,
        choices=tuple(choice_results),
    )


def _process_and_record(
    loop: GameLoop,
    player_id: UUID,
    action: GameAction,
    offered: set[int],
    bought: set[int],
    valuations: dict[int, PropertyAcquisitionValuation],
) -> None:
    actor = loop.current_decision_player_id
    pending_tile = loop.pending_tile_id
    result = loop.process_action(action)
    if result["status"] != "SUCCESS":
        raise RuntimeError(
            "A simulação condicional produziu uma ação rejeitada: "
            f"{action.action_type}: {result['reason']}"
        )
    if actor != player_id:
        return
    if result.get("event") == "AWAITING_PROPERTY_DECISION":
        tile_id = result.get("tile_id")
        if (
            tile_id in valuations
            and loop.game.property_states[tile_id].owner_id is None
        ):
            offered.add(tile_id)
    if action.action_type == ActionType.BUY_PROPERTY and pending_tile in valuations:
        bought.add(pending_tile)


def _summarize_choice_horizon(
    *,
    choice: ActionType,
    rounds: int,
    requested_turns: int,
    starting_cash: int,
    snapshots: list[dict[str, object]],
    valuations: dict[int, PropertyAcquisitionValuation],
) -> JailChoiceHorizonForecast:
    if not snapshots:
        raise RuntimeError(f"Nenhum cenário registrado para {rounds} rodada(s).")
    cash_deltas = np.asarray(
        [int(snapshot["cash"]) - starting_cash for snapshot in snapshots],
        dtype=np.float64,
    )
    offered = [snapshot["offered"] for snapshot in snapshots]
    bought = [snapshot["bought"] for snapshot in snapshots]
    all_tile_ids = sorted(valuations)
    opportunities: list[ConditionalPropertyOpportunity] = []
    for tile_id in all_tile_ids:
        offered_count = sum(tile_id in tile_ids for tile_ids in offered)
        bought_count = sum(tile_id in tile_ids for tile_ids in bought)
        probability_offered = offered_count / len(snapshots)
        probability_bought = bought_count / len(snapshots)
        probability_bought_if_offered = (
            bought_count / offered_count if offered_count else 0.0
        )
        valuation = valuations[tile_id]
        weighted_value = tuple(
            (
                horizon.horizon_rounds,
                probability_bought * max(0.0, horizon.expected_net_cash_flow),
            )
            for horizon in valuation.portfolio_impact.horizons
        )
        opportunities.append(
            ConditionalPropertyOpportunity(
                tile_id=tile_id,
                name=valuation.name,
                probability_offered_to_player=probability_offered,
                probability_bought_by_greedy_policy=probability_bought,
                probability_bought_if_offered=probability_bought_if_offered,
                acquisition_valuation=valuation,
                greedy_weighted_positive_net_value=weighted_value,
            )
        )

    p10, median, p90 = np.quantile(cash_deltas, (0.1, 0.5, 0.9))
    distinct_counts = np.asarray([len(tile_ids) for tile_ids in offered], dtype=np.float64)
    return JailChoiceHorizonForecast(
        action_type=str(choice),
        horizon_rounds=rounds,
        requested_turns=requested_turns,
        mean_cash_delta=float(np.mean(cash_deltas)),
        std_cash_delta=float(np.std(cash_deltas)),
        p10_cash_delta=float(p10),
        median_cash_delta=float(median),
        p90_cash_delta=float(p90),
        bankruptcy_probability=float(np.mean([snapshot["bankrupt"] for snapshot in snapshots])),
        in_jail_probability=float(np.mean([snapshot["in_jail"] for snapshot in snapshots])),
        probability_any_property_offered=float(np.mean(distinct_counts > 0)),
        mean_distinct_properties_offered=float(np.mean(distinct_counts)),
        property_opportunities=tuple(opportunities),
    )


def _reseed_scenario(loop: GameLoop, seed_rng: random.Random) -> None:
    for source in (loop.game.dice, *loop.game.decks.values()):
        rng = getattr(source, "_rng", None)
        reseed = getattr(rng, "seed", None)
        if callable(reseed):
            reseed(seed_rng.getrandbits(64))


def _validate_positive_ints(name: str, values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} deve ser uma coleção de inteiros positivos.") from exc
    if not result:
        raise ValueError(f"Informe ao menos um valor para {name}.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"Cada valor de {name} deve ser um inteiro.")
    if any(value <= 0 for value in result):
        raise ValueError(f"Todos os valores de {name} devem ser positivos.")
    if len(set(result)) != len(result):
        raise ValueError(f"Os valores de {name} não podem se repetir.")
    return tuple(sorted(result))


def _validate_positive_int(
    name: str,
    value: int,
    *,
    allow_zero: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} deve ser um inteiro.")
    if value < (0 if allow_zero else 1):
        qualifier = "não negativo" if allow_zero else "positivo"
        raise ValueError(f"{name} deve ser {qualifier}.")
