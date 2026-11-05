"""State-conditioned analytical cash-flow forecasts using board movement probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from math import exp, pi, sqrt
from statistics import NormalDist
from threading import RLock
from time import perf_counter
from typing import Any, Iterable
from uuid import UUID

import numpy as np

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.board import PropertyTile, TaxTile
from app.infraestructure.game_config.models import GameConfig


FORECAST_HORIZONS = (1, 2)
FORECAST_SCENARIOS = ("buy", "decline")
FORECAST_STAT_NAMES = ("mean", "p05", "p25", "p50", "p75", "p95")
# Legacy argument retained for source compatibility; these forecasts are analytic.
# Compatibility names for callers; analytical forecasts use no samples.
DEFAULT_FORECAST_SAMPLES = 0
DEFAULT_MANAGEMENT_FORECAST_SAMPLES = 0
MANAGEMENT_ACTION_TYPES = (
    ActionType.BUILD_HOUSE,
    ActionType.SELL_HOUSE,
    ActionType.MORTGAGE,
    ActionType.UNMORTGAGE,
)
JAIL_OPTIONS = (
    ActionType.ROLL_DICE,
    ActionType.PAY_JAIL_FINE,
    ActionType.USE_JAIL_CARD,
)

# Models cache exact dice/card outcomes by movement state. Retaining a small
# process-local pool lets all forecasts in an episode (including cloned trade
# scenarios) reuse that work without retaining game state or cash values.
_MOVEMENT_MODELS: OrderedDict[tuple[int, int, int], tuple[object, object, object]] = OrderedDict()
_MOVEMENT_MODELS_LOCK = RLock()
_MOVEMENT_MODEL_CACHE_LIMIT = 8
_FORECAST_STAGE_SECONDS: dict[str, float] = {}
_FORECAST_STAGE_CALLS: dict[str, int] = {}
_MOVEMENT_MODEL_CACHE_HITS = 0
_MOVEMENT_MODEL_CACHE_MISSES = 0


def forecast_stage_metrics(*, reset: bool = False) -> dict[str, float | int]:
    """Return process-local forecast timing totals, optionally clearing them."""
    global _MOVEMENT_MODEL_CACHE_HITS, _MOVEMENT_MODEL_CACHE_MISSES
    result: dict[str, float | int] = {}
    for stage, seconds in _FORECAST_STAGE_SECONDS.items():
        result[f"forecast_{stage}_seconds"] = seconds
        result[f"forecast_{stage}_calls"] = _FORECAST_STAGE_CALLS.get(stage, 0)
    result["movement_model_cache_hits"] = _MOVEMENT_MODEL_CACHE_HITS
    result["movement_model_cache_misses"] = _MOVEMENT_MODEL_CACHE_MISSES
    if reset:
        _FORECAST_STAGE_SECONDS.clear()
        _FORECAST_STAGE_CALLS.clear()
        _MOVEMENT_MODEL_CACHE_HITS = 0
        _MOVEMENT_MODEL_CACHE_MISSES = 0
    return result


def _record_forecast_stage(stage: str, started: float) -> None:
    _FORECAST_STAGE_SECONDS[stage] = (
        _FORECAST_STAGE_SECONDS.get(stage, 0.0) + perf_counter() - started
    )
    _FORECAST_STAGE_CALLS[stage] = _FORECAST_STAGE_CALLS.get(stage, 0) + 1


@dataclass(frozen=True, slots=True)
class PropertyDecisionCashFlowForecast:
    """Analytical cash moments and normal-approximation quantiles by scenario."""
    horizons: tuple[int, ...]
    scenario_names: tuple[str, ...]
    statistic_names: tuple[str, ...]
    sample_count: int
    option_available: tuple[bool, bool]
    cash_change: np.ndarray
    cash_in: np.ndarray
    cash_out: np.ndarray
    ending_cash: np.ndarray
    minimum_cash: np.ndarray
    debt_probability: np.ndarray
    bankruptcy_probability: np.ndarray

    def as_dict(self, money_scale: float) -> dict[str, Any]:
        """Return model-ready arrays; sample_count is zero for analytic outputs."""
        return {
            "active": np.asarray([1.0], dtype=np.float32),
            "horizons": np.asarray(self.horizons, dtype=np.float32),
            "option_available": np.asarray(self.option_available, dtype=np.float32),
            "sample_count": np.asarray([self.sample_count], dtype=np.float32),
            "cash_change": (self.cash_change / money_scale).astype(np.float32),
            "cash_in": (self.cash_in / money_scale).astype(np.float32),
            "cash_out": (self.cash_out / money_scale).astype(np.float32),
            "ending_cash": (self.ending_cash / money_scale).astype(np.float32),
            "minimum_cash": (self.minimum_cash / money_scale).astype(np.float32),
            "debt_probability": self.debt_probability.astype(np.float32),
            "bankruptcy_probability": self.bankruptcy_probability.astype(np.float32),
        }


def forecast_property_cash_flows(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    *,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    samples: int = DEFAULT_FORECAST_SAMPLES,
    legal_actions: list[GameAction] | None = None,
) -> PropertyDecisionCashFlowForecast:
    """Compare buying and declining using exact movement probabilities.

    Ownership, development, mortgages, and player positions remain fixed
    during the forecast. No future purchase, trade, construction, liquidation,
    or policy decisions are modeled. Cash-flow quantiles use a normal
    approximation from exact Markov reward moments.
    """
    started = perf_counter()
    if loop.turn_phase != GameLoop.AWAITING_PROPERTY_DECISION or loop.pending_tile_id is None:
        raise ValueError("A previsão exige uma propriedade aguardando decisão de compra.")
    if loop.current_decision_player_id != focal_player_id:
        raise ValueError("O jogador previsto deve ser o decisor atual.")
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(
        not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
        for horizon in horizons
    ):
        raise ValueError("horizons deve conter inteiros positivos em ordem crescente, sem repetição.")
    tile_id = loop.pending_tile_id
    available_actions = loop.legal_actions() if legal_actions is None else legal_actions
    legal_types = {action.action_type for action in available_actions}
    option_available = (
        ActionType.BUY_PROPERTY in legal_types,
        ActionType.DECLINE_PROPERTY in legal_types,
    )
    branches: list[dict[str, np.ndarray]] = []
    movement_model = _movement_model(loop, config)
    for index, action_type in enumerate((ActionType.BUY_PROPERTY, ActionType.DECLINE_PROPERTY)):
        if not option_available[index]:
            branches.append(_empty_analytic_path(len(horizons)))
            continue
        branch = loop.clone()
        if action_type == ActionType.BUY_PROPERTY:
            applied = branch.process_action(GameAction(action_type))
            if applied["status"] != "SUCCESS":
                raise RuntimeError(f"Não foi possível aplicar a compra ao forecast: {applied}.")
        branches.append(_analytic_cash_path(
            branch, focal_player_id, config, horizons,
            starting_cash=loop.game.player_states[focal_player_id].money,
            movement_model=movement_model,
        ))
    forecast = _combine_analytic_paths(horizons, option_available, branches)
    _record_forecast_stage("property_cash", started)
    return forecast


def empty_cash_flow_forecast() -> dict[str, np.ndarray]:
    """Fixed-shape inactive value used outside a property decision."""
    horizon_count = len(FORECAST_HORIZONS)
    shape = (2, horizon_count, len(FORECAST_STAT_NAMES))
    return {
        "active": np.zeros(1, dtype=np.float32),
        "horizons": np.asarray(FORECAST_HORIZONS, dtype=np.float32),
        "option_available": np.zeros(2, dtype=np.float32),
        "sample_count": np.zeros(1, dtype=np.float32),
        "cash_change": np.zeros(shape, dtype=np.float32),
        "cash_in": np.zeros(shape, dtype=np.float32),
        "cash_out": np.zeros(shape, dtype=np.float32),
        "ending_cash": np.zeros(shape, dtype=np.float32),
        "minimum_cash": np.zeros(shape, dtype=np.float32),
        "debt_probability": np.zeros((2, horizon_count), dtype=np.float32),
        "bankruptcy_probability": np.zeros((2, horizon_count), dtype=np.float32),
    }


def forecast_management_cash_flows(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    *,
    samples: int = DEFAULT_MANAGEMENT_FORECAST_SAMPLES,
    legal_actions: list[GameAction] | None = None,
) -> dict[tuple[str, int], PropertyDecisionCashFlowForecast]:
    """Compare each legal management action with leaving the current state intact."""
    if loop.current_decision_player_id != focal_player_id:
        return {}
    legal_actions = loop.legal_actions() if legal_actions is None else legal_actions
    candidates = [
        action for action in legal_actions
        if action.action_type in MANAGEMENT_ACTION_TYPES
        and isinstance(action.payload.get("tile_id"), int)
    ]
    started = perf_counter()
    result: dict[tuple[str, int], PropertyDecisionCashFlowForecast] = {}
    movement_model = _movement_model(loop, config)
    # The reference state is invariant across candidates; calculating it once
    # avoids repeating the largest shared branch in every action comparison.
    baseline = _analytic_cash_path(
        loop, focal_player_id, config, FORECAST_HORIZONS,
        movement_model=movement_model,
    )
    for candidate in candidates:
        changed = loop.clone()
        applied = changed.process_action(candidate)
        if applied["status"] != "SUCCESS":
            continue
        paths = [
            _analytic_cash_path(
                changed, focal_player_id, config, FORECAST_HORIZONS,
                starting_cash=loop.game.player_states[focal_player_id].money,
                movement_model=movement_model,
            ),
            baseline,
        ]
        result[(candidate.action_type, candidate.payload["tile_id"])] = _combine_analytic_paths(
            FORECAST_HORIZONS, (True, True), paths
        )
    _record_forecast_stage("management_cash", started)
    return result


def _analytic_cash_path(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    horizons: tuple[int, ...],
    *,
    starting_cash: int | None = None,
    movement_model: MonopolyBoardModel | None = None,
) -> dict[str, np.ndarray]:
    """Propagate exact movement probabilities and summarize cash-flow moments.

    Movement and configured card draws are represented by the finite Markov
    model. Current ownership, buildings, and mortgages are held fixed. The
    reported cash quantiles use a normal approximation to the exact first two
    reward moments; future debt-resolution choices are intentionally absent.
    """
    started = perf_counter()
    from app.domain.analytics.markov.stationary_board import MonopolyMovementState

    game = loop.game
    starting_cash = (
        game.player_states[focal_player_id].money
        if starting_cash is None else int(starting_cash)
    )
    initial_delta = game.player_states[focal_player_id].money - starting_cash
    active_ids = tuple(game.active_player_ids)
    model = movement_model or _movement_model(loop, config)
    kernel_cache: dict[tuple[UUID, MonopolyMovementState], dict] = {}

    def cash_reward(mover_id: UUID, outcome: MonopolyMovementOutcome) -> float:
        reward = 0.0
        player_is_focal = mover_id == focal_player_id
        if player_is_focal:
            reward += outcome.go_salary_count * loop.rules.go_salary
            reward += outcome.player_cash_delta
            if outcome.repair_house_cost or outcome.repair_hotel_cost:
                for property_tile in game.owned_tiles(focal_player_id):
                    level = game.property_states[property_tile.tile_id].houses
                    if level == 5:
                        reward -= outcome.repair_hotel_cost
                    else:
                        reward -= level * outcome.repair_house_cost
        else:
            reward += outcome.other_player_cash_delta

        if outcome.landing_position is None:
            return reward
        tile = game.board.get_tile(outcome.landing_position)
        if isinstance(tile, PropertyTile):
            property_state = game.property_states[tile.tile_id]
            owner_id = property_state.owner_id
            if owner_id is not None and not property_state.is_mortgaged:
                rent = game.rent_for(tile, int(outcome.rent_dice_total))
                rent = int(round(rent * outcome.rent_multiplier))
                if player_is_focal and owner_id != mover_id:
                    reward -= rent
                elif not player_is_focal and owner_id == focal_player_id:
                    reward += rent
        elif player_is_focal and isinstance(tile, TaxTile):
            amount = tile.tax_amount
            if tile.percent_option is not None and loop.rules.income_tax_choice:
                amount = min(
                    amount,
                    game.net_worth(focal_player_id) * tile.percent_option // 100,
                )
            reward -= amount
        return reward

    def turn_kernel(mover_id: UUID, state: MonopolyMovementState) -> dict:
        """Map a start-of-turn movement state to end states and reward moments."""
        key = (mover_id, state)
        cached = kernel_cache.get(key)
        if cached is not None:
            return cached
        weighted: dict[MonopolyMovementState, list[float]] = {}
        for outcome in model.outcomes_from(state):
            reward = cash_reward(mover_id, outcome)
            if outcome.turn_ended:
                branches = ((outcome.state, 1.0, reward, reward * reward),)
            else:
                continuation = turn_kernel(mover_id, outcome.state)
                branches = tuple(
                    (end_state, probability,
                     reward + mean,
                     reward * reward + 2.0 * reward * mean + second)
                    for end_state, (probability, mean, second) in continuation.items()
                )
            for end_state, probability, mean, second in branches:
                mass = outcome.probability * probability
                aggregate = weighted.setdefault(end_state, [0.0, 0.0, 0.0])
                aggregate[0] += mass
                aggregate[1] += mass * mean
                aggregate[2] += mass * second
        result = {
            end_state: (mass, first / mass, second / mass)
            for end_state, (mass, first, second) in weighted.items()
            if mass > 0
        }
        kernel_cache[key] = result
        return result

    total_means = np.zeros(len(horizons), dtype=np.float64)
    total_variances = np.zeros(len(horizons), dtype=np.float64)
    focal_jail_probability = np.zeros(len(horizons), dtype=np.float64)
    for player_id in active_ids:
        player = game.player_states[player_id]
        if player.is_bankrupt:
            continue
        start = MonopolyMovementState(
            position=game.position_of(player_id),
            in_jail=player.is_in_jail,
            jail_attempts=min(player.jail_turns, model.max_jail_attempts - 1) if player.is_in_jail else 0,
            doubles_streak=0,
        )
        # State values are probability mass, weighted first moment, and
        # weighted second moment of this player's cumulative contribution.
        distribution = {start: (1.0, 0.0, 0.0)}
        for turn_index in range(max(horizons)):
            next_distribution: dict[MonopolyMovementState, list[float]] = {}
            for source_state, (mass, first, second) in distribution.items():
                for end_state, (probability, reward_mean, reward_second) in turn_kernel(
                    player_id, source_state
                ).items():
                    target = next_distribution.setdefault(end_state, [0.0, 0.0, 0.0])
                    target[0] += mass * probability
                    target[1] += probability * (first + mass * reward_mean)
                    target[2] += probability * (
                        second + 2.0 * first * reward_mean + mass * reward_second
                    )
            distribution = {
                state: tuple(values) for state, values in next_distribution.items()
            }
            if turn_index + 1 in horizons:
                horizon_index = horizons.index(turn_index + 1)
                total_means[horizon_index] += sum(values[1] for values in distribution.values())
                raw_second = sum(values[2] for values in distribution.values())
                player_mean = sum(values[1] for values in distribution.values())
                total_variances[horizon_index] += max(0.0, raw_second - player_mean**2)
                if player_id == focal_player_id:
                    focal_jail_probability[horizon_index] = sum(
                        values[0] for state, values in distribution.items() if state.in_jail
                    )

    result = {name: np.zeros((len(horizons), len(FORECAST_STAT_NAMES)), dtype=np.float64)
              for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")}
    result["cash_mean"] = np.zeros(len(horizons), dtype=np.float64)
    result["cash_variance"] = np.zeros(len(horizons), dtype=np.float64)
    debt_probability = np.zeros(len(horizons), dtype=np.float64)
    existing_debt = any(debt.debtor_id == focal_player_id for debt in game.debts)
    base_change = initial_delta
    for index, horizon in enumerate(horizons):
        mean = base_change + total_means[index]
        variance = total_variances[index]
        sigma = sqrt(variance)
        result["cash_mean"][index] = mean
        result["cash_variance"][index] = variance
        change_stats = _normal_summary(mean, sigma)
        ending_mean = starting_cash + mean
        ending_stats = _normal_summary(ending_mean, sigma)
        result["cash_change"][index] = change_stats
        result["ending_cash"][index] = ending_stats
        result["cash_in"][index] = _positive_part_summary(mean, sigma)
        # Preserve immediate spending (for example, a property's purchase
        # price) as a cash outlay even when later income offsets it.
        immediate_outlay = max(0.0, -float(initial_delta))
        result["cash_out"][index] = immediate_outlay + _positive_part_summary(
            -total_means[index], sigma
        )
        result["minimum_cash"][index] = np.minimum(ending_stats, starting_cash)
        debt_probability[index] = max(
            float(existing_debt),
            _normal_cdf((0.0 - ending_mean) / sigma) if sigma else float(ending_mean < 0),
        )
    result["debt"] = debt_probability
    result["bankruptcy"] = np.zeros(len(horizons), dtype=np.float64)
    result["in_jail"] = focal_jail_probability
    _record_forecast_stage("cash_path", started)
    return result


def _movement_model(loop: GameLoop, config: GameConfig) -> MonopolyBoardModel:
    """Get the cached exact movement model for this config/rule combination."""
    global _MOVEMENT_MODEL_CACHE_HITS, _MOVEMENT_MODEL_CACHE_MISSES
    from app.domain.analytics.markov.stationary_board import MonopolyBoardModel
    player_count = max(2, len(loop.game.active_player_ids))
    key = (id(config), id(loop.rules), player_count)
    with _MOVEMENT_MODELS_LOCK:
        cached = _MOVEMENT_MODELS.get(key)
        if cached is not None and cached[0] is config and cached[1] is loop.rules:
            _MOVEMENT_MODEL_CACHE_HITS += 1
            _MOVEMENT_MODELS.move_to_end(key)
            return cached[2]
        _MOVEMENT_MODEL_CACHE_MISSES += 1
        model = MonopolyBoardModel(
            config, loop.rules, number_of_players=player_count
        )
        _MOVEMENT_MODELS[key] = (config, loop.rules, model)
        _MOVEMENT_MODELS.move_to_end(key)
        while len(_MOVEMENT_MODELS) > _MOVEMENT_MODEL_CACHE_LIMIT:
            _MOVEMENT_MODELS.popitem(last=False)
        return model


def _normal_summary(mean: float, sigma: float) -> np.ndarray:
    if sigma <= 1e-12:
        return np.full(len(FORECAST_STAT_NAMES), mean, dtype=np.float64)
    normal = NormalDist()
    return np.asarray(
        [mean, *(mean + sigma * normal.inv_cdf(q) for q in (0.05, 0.25, 0.50, 0.75, 0.95))],
        dtype=np.float64,
    )


def _normal_cdf(value: float) -> float:
    return NormalDist().cdf(value)


def _positive_part_summary(mean: float, sigma: float) -> np.ndarray:
    """Distribution summary for max(X, 0), where X is approximately normal."""
    if sigma <= 1e-12:
        return np.full(len(FORECAST_STAT_NAMES), max(0.0, mean), dtype=np.float64)
    normal = NormalDist()
    positive_quantiles = np.maximum(
        0.0,
        np.asarray([mean + sigma * normal.inv_cdf(q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]),
    )
    z = mean / sigma
    density = exp(-0.5 * z * z) / sqrt(2.0 * pi)
    expected = sigma * density + mean * normal.cdf(z)
    return np.asarray([expected, *positive_quantiles], dtype=np.float64)


def _combine_analytic_paths(horizons, option_available, paths):
    fields = ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")
    summarized = {name: np.stack([path[name] for path in paths], axis=0) for name in fields}
    return PropertyDecisionCashFlowForecast(
        horizons=tuple(horizons),
        scenario_names=FORECAST_SCENARIOS,
        statistic_names=FORECAST_STAT_NAMES,
        sample_count=0,
        option_available=tuple(option_available),
        cash_change=summarized["cash_change"],
        cash_in=summarized["cash_in"],
        cash_out=summarized["cash_out"],
        ending_cash=summarized["ending_cash"],
        minimum_cash=summarized["minimum_cash"],
        debt_probability=np.stack([path["debt"] for path in paths], axis=0),
        bankruptcy_probability=np.stack([path["bankruptcy"] for path in paths], axis=0),
    )


def forecast_jail_decisions(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    *,
    samples: int = DEFAULT_FORECAST_SAMPLES,
    legal_actions: list[GameAction] | None = None,
) -> dict[str, np.ndarray]:
    """Compare jail choices with fixed-state Markov cash-flow projections."""
    started = perf_counter()
    from app.domain.analytics.markov.stationary_board import MonopolyMovementState

    if (
        loop.turn_phase != GameLoop.BEFORE_ROLL
        or loop.current_decision_player_id != focal_player_id
        or not loop.game.player_states[focal_player_id].is_in_jail
    ):
        raise ValueError("A previsão de prisão exige que o jogador atual esteja preso antes da rolagem.")
    available_actions = loop.legal_actions() if legal_actions is None else legal_actions
    legal_kinds = {action.action_type for action in available_actions}
    available = np.asarray([kind in legal_kinds for kind in JAIL_OPTIONS], dtype=bool)
    property_ids = tuple(
        tile.tile_id for tile in loop.game.board.property_tiles
        if loop.game.property_states[tile.tile_id].owner_id is None
    )
    model = _movement_model(loop, config)
    start_cash = loop.game.player_states[focal_player_id].money
    paths: list[dict[str, np.ndarray]] = []
    branches: list[GameLoop] = []
    for option_index, kind in enumerate(JAIL_OPTIONS):
        if not available[option_index]:
            branches.append(loop)
            paths.append(_empty_analytic_path(len(FORECAST_HORIZONS)))
            paths[-1].update({
                "in_jail": np.zeros(len(FORECAST_HORIZONS), dtype=np.float64),
                "jail_cards": np.zeros(len(FORECAST_HORIZONS), dtype=np.float64),
            })
            continue
        branch = loop.clone()
        if kind != ActionType.ROLL_DICE:
            applied = branch.process_action(GameAction(kind))
            if applied["status"] != "SUCCESS":
                raise RuntimeError(f"Ação de prisão rejeitada no forecast: {applied}.")
        branches.append(branch)
        path = _analytic_cash_path(
            branch, focal_player_id, config, FORECAST_HORIZONS,
            starting_cash=start_cash,
            movement_model=model,
        )
        state = branch.game.player_states[focal_player_id]
        path["jail_cards"] = np.full(
            len(FORECAST_HORIZONS), len(state.jail_cards), dtype=np.float64
        )
        paths.append(path)

    shape = (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES))
    result: dict[str, np.ndarray] = {
        "active": np.ones(1, dtype=np.float32),
        "horizons": np.asarray(FORECAST_HORIZONS, dtype=np.float32),
        "option_available": available.astype(np.float32),
        "sample_count": np.zeros(1, dtype=np.float32),
    }
    for field in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash"):
        result[field] = np.stack([path[field] for path in paths], axis=0).astype(np.float32)

    for field, path_field in (
        ("debt_probability", "debt"),
        ("bankruptcy_probability", "bankruptcy"),
        ("in_jail_probability", "in_jail"),
        ("jail_cards_remaining", "jail_cards"),
    ):
        values = np.stack([path[path_field] for path in paths], axis=0)
        result[field] = values.astype(np.float32)

    result["unowned_property_probability"] = np.zeros(
        (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(property_ids)), dtype=np.float32
    )
    for option_index, branch in enumerate(branches):
        if not available[option_index] or not property_ids:
            continue
        player = branch.game.player_states[focal_player_id]
        start = MonopolyMovementState(
            position=branch.game.position_of(focal_player_id),
            in_jail=player.is_in_jail,
            jail_attempts=min(player.jail_turns, model.max_jail_attempts - 1) if player.is_in_jail else 0,
            doubles_streak=0,
        )
        property_positions = {
            tile_id: branch.game.board._position_by_id[tile_id]
            for tile_id in property_ids
        }
        result["unowned_property_probability"][option_index] = np.asarray([
            [
                np.clip(
                    1.0 - _probability_no_landing(
                        model, start, property_positions[tile_id], horizon
                    ),
                    0.0,
                    1.0,
                )
                for tile_id in property_ids
            ]
            for horizon in FORECAST_HORIZONS
        ], dtype=np.float32)
    result["unowned_property_ids"] = np.asarray(property_ids, dtype=np.int32)
    _record_forecast_stage("jail_cash", started)
    return result


def _probability_no_landing(
    model: MonopolyBoardModel,
    start: MonopolyMovementState,
    tile_id: int,
    turns: int,
) -> float:
    """Return the exact chance of avoiding one unowned tile through N turns."""
    cache: dict[MonopolyMovementState, dict[MonopolyMovementState, float]] = {}

    def no_hit_turn(state: MonopolyMovementState) -> dict[MonopolyMovementState, float]:
        if state in cache:
            return cache[state]
        output: dict[MonopolyMovementState, float] = {}
        for outcome in model.outcomes_from(state):
            if outcome.landing_position == tile_id:
                continue
            if outcome.turn_ended:
                continuations = {outcome.state: 1.0}
            else:
                continuations = no_hit_turn(outcome.state)
            for target, probability in continuations.items():
                output[target] = output.get(target, 0.0) + outcome.probability * probability
        cache[state] = output
        return output

    distribution = {start: 1.0}
    for _ in range(turns):
        next_distribution: dict[MonopolyMovementState, float] = {}
        for state, mass in distribution.items():
            for target, probability in no_hit_turn(state).items():
                next_distribution[target] = next_distribution.get(target, 0.0) + mass * probability
        distribution = next_distribution
    return sum(distribution.values())


def _empty_analytic_path(horizon_count: int) -> dict[str, np.ndarray]:
    shape = (horizon_count, len(FORECAST_STAT_NAMES))
    return {
        name: np.zeros(shape, dtype=np.float64)
        for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")
    } | {
        "debt": np.zeros(horizon_count, dtype=np.float64),
        "bankruptcy": np.zeros(horizon_count, dtype=np.float64),
    }


DEFAULT_HORIZONS = (1, 2)
DEFAULT_SIMULATIONS = 0


@dataclass(frozen=True, slots=True)
class CashFlowHorizonForecast:
    """Normal-approximation cash-flow distribution for one forecast horizon."""

    horizon_rounds: int
    requested_turns: int
    mean_net_cash_flow: float
    std_net_cash_flow: float
    p10_net_cash_flow: float
    median_net_cash_flow: float
    p90_net_cash_flow: float
    mean_ending_cash: float
    p10_ending_cash: float
    median_ending_cash: float
    p90_ending_cash: float
    bankruptcy_probability: float
    game_over_probability: float
    # Empty for analytic forecasts; retained in the public record for compatibility.
    net_cash_flow_samples: tuple[int, ...]
    ending_cash_samples: tuple[int, ...]
    bankrupt_samples: tuple[bool, ...]
    game_over_samples: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class CashFlowForecast:
    """Fixed-holdings Markov forecast for one player."""

    player_id: UUID
    seed: int
    simulations: int
    starting_cash: int
    policy_name: str
    conditioned_on: str | None
    horizons: tuple[CashFlowHorizonForecast, ...]

    def for_rounds(self, rounds: int) -> CashFlowHorizonForecast:
        """Return the forecast for the requested number of rounds."""
        for forecast in self.horizons:
            if forecast.horizon_rounds == rounds:
                return forecast
        raise KeyError(f"Horizonte não calculado: {rounds} rodada(s).")


def forecast_cash_flow(
    loop: GameLoop,
    player_id: UUID,
    *,
    config: GameConfig,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int | None = None,
    greedy_reserve: int = 150,
    initial_action: GameAction | None = None,
) -> CashFlowForecast:
    """Forecast cash flow from Markov movement probabilities and fixed holdings.

    The ``simulations``, ``seed``, and ``greedy_reserve`` arguments remain for
    source compatibility but do not affect this analytic forecast. Ownership,
    development, and mortgages are fixed after the optional immediate action.
    The movement transition is exact for the configured dice/card model; cash
    quantiles are a normal approximation from its exact first two moments.
    ``simulations`` in the returned object is zero and empirical sample tuples
    are empty. Bankruptcy is not projected because no future debt decisions are
    modeled; negative-cash exposure is reported separately as debt probability.
    """
    started = perf_counter()
    horizon_values = _validate_horizons(horizons)
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise TypeError("seed deve ser um inteiro ou None.")
    if player_id not in loop.game.player_states:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    if initial_action is not None and not isinstance(initial_action, GameAction):
        raise TypeError("initial_action deve ser GameAction ou None.")

    starting_cash = loop.game.player_states[player_id].money
    scenario = loop.clone()
    if initial_action is not None:
        applied = scenario.process_action(initial_action)
        if applied["status"] != "SUCCESS":
            raise ValueError(
                "A ação inicial do forecast condicionado foi rejeitada: "
                f"{initial_action.action_type}: {applied['reason']}"
            )
    path = _analytic_cash_path(
        scenario, player_id, config, horizon_values,
        starting_cash=starting_cash,
    )
    forecasts = []
    active_turns_per_round = max(1, len(scenario.game.active_player_ids))
    already_bankrupt = scenario.game.player_states[player_id].is_bankrupt
    already_over = scenario.game.game_over
    for index, rounds in enumerate(horizon_values):
        mean = float(path["cash_mean"][index])
        std = sqrt(float(path["cash_variance"][index]))
        normal = NormalDist()
        net_quantiles = [mean + std * normal.inv_cdf(q) for q in (0.1, 0.5, 0.9)] if std else [mean] * 3
        ending_mean = starting_cash + mean
        cash_quantiles = [value + starting_cash for value in net_quantiles]
        forecasts.append(CashFlowHorizonForecast(
            horizon_rounds=rounds,
            requested_turns=rounds * active_turns_per_round,
            mean_net_cash_flow=mean,
            std_net_cash_flow=std,
            p10_net_cash_flow=float(net_quantiles[0]),
            median_net_cash_flow=float(net_quantiles[1]),
            p90_net_cash_flow=float(net_quantiles[2]),
            mean_ending_cash=ending_mean,
            p10_ending_cash=float(cash_quantiles[0]),
            median_ending_cash=float(cash_quantiles[1]),
            p90_ending_cash=float(cash_quantiles[2]),
            bankruptcy_probability=float(already_bankrupt),
            game_over_probability=float(already_over),
            net_cash_flow_samples=(),
            ending_cash_samples=(),
            bankrupt_samples=(),
            game_over_samples=(),
        ))
    forecast = CashFlowForecast(
        player_id=player_id,
        seed=0 if seed is None else seed,
        simulations=0,
        starting_cash=starting_cash,
        policy_name="FixedStateMarkov",
        conditioned_on=(initial_action.action_type if initial_action else None),
        horizons=tuple(forecasts),
    )
    _record_forecast_stage("conditioned_cash", started)
    return forecast


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(horizons)
    except TypeError as exc:
        raise TypeError("horizons deve ser uma coleção de inteiros positivos.") from exc
    if not values:
        raise ValueError("Informe ao menos um horizonte.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("Cada horizonte deve ser um inteiro.")
    if any(value <= 0 for value in values):
        raise ValueError("Os horizontes devem ser positivos.")
    if len(set(values)) != len(values):
        raise ValueError("Os horizontes não podem se repetir.")
    return tuple(sorted(values))
