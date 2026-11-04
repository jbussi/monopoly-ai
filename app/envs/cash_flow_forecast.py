"""State-conditioned Monte Carlo cash-flow forecasts for property decisions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import UUID

import numpy as np

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.baselines import GreedyPolicy
from app.infraestructure.game_config.models import GameConfig


FORECAST_HORIZONS = (1, 3, 5)
FORECAST_SCENARIOS = ("buy", "decline")
FORECAST_STAT_NAMES = ("mean", "p05", "p25", "p50", "p75", "p95")
DEFAULT_FORECAST_SAMPLES = 32
DEFAULT_MANAGEMENT_FORECAST_SAMPLES = 8
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


@dataclass(frozen=True, slots=True)
class PropertyDecisionCashFlowForecast:
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
        """Return model-ready arrays with monetary fields normalized."""
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


class _ForecastPolicy:
    """Use the project GreedyPolicy, while keeping the declined asset unowned."""

    def __init__(self, focal_player_id: UUID, declined_tile_id: int):
        self.focal_player_id = focal_player_id
        self.declined_tile_id = declined_tile_id
        self.greedy = GreedyPolicy()

    def __call__(self, loop: GameLoop) -> GameAction:
        if loop.current_decision_player_id == self.focal_player_id:
            if (
                loop.turn_phase == GameLoop.AWAITING_PROPERTY_DECISION
                and loop.pending_tile_id == self.declined_tile_id
            ):
                return GameAction(ActionType.DECLINE_PROPERTY)
            if (
                loop.turn_phase == GameLoop.AUCTION
                and loop.pending_tile_id == self.declined_tile_id
            ):
                if loop.current_decision_player_id in loop._auction_active:
                    return GameAction(ActionType.AUCTION_WITHDRAW)
        return self.greedy(loop)


def forecast_property_cash_flows(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    *,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    samples: int = DEFAULT_FORECAST_SAMPLES,
    legal_actions: list[GameAction] | None = None,
) -> PropertyDecisionCashFlowForecast:
    """Compare buy-now with decline, simulating engine outcomes from this state.

    Horizons count completed turns of the focal player after the current
    property-resolution step. Every future player uses GreedyPolicy. The
    declined branch prevents the focal player from acquiring this same
    property later. Both branches share sample seeds for a paired comparison.
    """
    if loop.turn_phase != GameLoop.AWAITING_PROPERTY_DECISION or loop.pending_tile_id is None:
        raise ValueError("A previsão exige uma propriedade aguardando decisão de compra.")
    if loop.current_decision_player_id != focal_player_id:
        raise ValueError("O jogador previsto deve ser o decisor atual.")
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(
        not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
        for horizon in horizons
    ):
        raise ValueError("horizons deve conter inteiros positivos em ordem crescente, sem repetição.")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples deve ser um inteiro maior ou igual a 2.")

    tile_id = loop.pending_tile_id
    available_actions = loop.legal_actions() if legal_actions is None else legal_actions
    legal_types = {action.action_type for action in available_actions}
    option_available = (
        ActionType.BUY_PROPERTY in legal_types,
        ActionType.DECLINE_PROPERTY in legal_types,
    )
    base_seed = _state_seed(loop, focal_player_id, tile_id)
    scenario_samples: list[list[dict[str, np.ndarray]]] = [[], []]
    paired_seeds = [
        (base_seed + sample_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        for sample_index in range(samples)
    ]

    for seed in paired_seeds:
        for scenario_index, action in enumerate((
            GameAction(ActionType.BUY_PROPERTY),
            GameAction(ActionType.DECLINE_PROPERTY),
        )):
            if not option_available[scenario_index]:
                scenario_samples[scenario_index].append(_empty_path(len(horizons)))
                continue
            path = _simulate_branch(
                loop,
                focal_player_id,
                tile_id,
                action,
                config,
                horizons,
                seed,
            )
            scenario_samples[scenario_index].append(path)

    return _summarize(
        horizons,
        samples,
        option_available,
        scenario_samples,
    )


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
    """Forecast each legal property-management action against deferring it once."""
    if loop.current_decision_player_id != focal_player_id:
        return {}
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples deve ser um inteiro maior ou igual a 2.")
    legal_actions = loop.legal_actions() if legal_actions is None else legal_actions
    candidates = [
        action for action in legal_actions
        if action.action_type in MANAGEMENT_ACTION_TYPES
        and isinstance(action.payload.get("tile_id"), int)
    ]
    result: dict[tuple[str, int], PropertyDecisionCashFlowForecast] = {}
    for candidate in candidates:
        deferred = _defer_management_action(loop, candidate, legal_actions)
        base_seed = _state_seed(loop, focal_player_id, candidate.payload["tile_id"])
        base_seed ^= int.from_bytes(candidate.action_type.encode("utf-8"), "little")
        paths_by_scenario: list[list[dict[str, np.ndarray]]] = [[], []]
        for sample_index in range(samples):
            seed = (base_seed + sample_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            for branch_index, first_action in enumerate((candidate, deferred)):
                paths_by_scenario[branch_index].append(_simulate_branch(
                    loop,
                    focal_player_id,
                    -1,
                    first_action,
                    config,
                    FORECAST_HORIZONS,
                    seed,
                ))
        result[(candidate.action_type, candidate.payload["tile_id"])] = _summarize(
            FORECAST_HORIZONS,
            samples,
            (True, True),
            paths_by_scenario,
        )
    return result


def forecast_jail_decisions(
    loop: GameLoop,
    focal_player_id: UUID,
    config: GameConfig,
    *,
    samples: int = DEFAULT_FORECAST_SAMPLES,
    legal_actions: list[GameAction] | None = None,
) -> dict[str, np.ndarray]:
    """Compare jail choices, cash trajectories, and future unowned-property chances."""
    if (
        loop.turn_phase != GameLoop.BEFORE_ROLL
        or loop.current_decision_player_id != focal_player_id
        or not loop.game.player_states[focal_player_id].is_in_jail
    ):
        raise ValueError("A previsão de prisão exige que o jogador atual esteja preso antes da rolagem.")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples deve ser um inteiro maior ou igual a 2.")

    available_actions = loop.legal_actions() if legal_actions is None else legal_actions
    legal_kinds = {action.action_type for action in available_actions}
    available = np.asarray([kind in legal_kinds for kind in JAIL_OPTIONS], dtype=bool)
    property_ids = tuple(
        tile.tile_id for tile in loop.game.board.property_tiles
        if loop.game.property_states[tile.tile_id].owner_id is None
    )
    base_seed = _state_seed(loop, focal_player_id, -2)
    scenario_paths: list[list[dict[str, np.ndarray]]] = [[] for _ in JAIL_OPTIONS]
    seeds = [
        (base_seed + index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        for index in range(samples)
    ]
    for seed in seeds:
        for option_index, kind in enumerate(JAIL_OPTIONS):
            if not available[option_index]:
                scenario_paths[option_index].append(_empty_jail_path(len(property_ids)))
                continue
            scenario_paths[option_index].append(_simulate_branch(
                loop,
                focal_player_id,
                -2,
                GameAction(kind),
                config,
                FORECAST_HORIZONS,
                seed,
                tracked_property_ids=property_ids,
            ))

    shape = (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(FORECAST_STAT_NAMES))
    result: dict[str, np.ndarray] = {
        "active": np.ones(1, dtype=np.float32),
        "horizons": np.asarray(FORECAST_HORIZONS, dtype=np.float32),
        "option_available": available.astype(np.float32),
        "sample_count": np.asarray([samples], dtype=np.float32),
    }
    for field in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash"):
        distributions = np.zeros(shape, dtype=np.float64)
        for option_index, paths in enumerate(scenario_paths):
            values = np.stack([path[field] for path in paths], axis=0)
            distributions[option_index, :, 0] = values.mean(axis=0)
            distributions[option_index, :, 1:] = np.quantile(
                values, (0.05, 0.25, 0.50, 0.75, 0.95), axis=0
            ).T
        result[field] = distributions.astype(np.float32)

    for field, path_field in (
        ("debt_probability", "debt"),
        ("bankruptcy_probability", "bankruptcy"),
        ("in_jail_probability", "in_jail"),
        ("jail_cards_remaining", "jail_cards"),
    ):
        values = np.zeros((len(JAIL_OPTIONS), len(FORECAST_HORIZONS)), dtype=np.float64)
        for option_index, paths in enumerate(scenario_paths):
            values[option_index] = np.stack(
                [path[path_field] for path in paths], axis=0
            ).mean(axis=0)
        result[field] = values.astype(np.float32)

    result["unowned_property_probability"] = np.zeros(
        (len(JAIL_OPTIONS), len(FORECAST_HORIZONS), len(property_ids)), dtype=np.float32
    )
    for option_index, paths in enumerate(scenario_paths):
        if property_ids:
            result["unowned_property_probability"][option_index] = np.stack(
                [path["opportunity_hits"] for path in paths], axis=0
            ).mean(axis=0).astype(np.float32)
    result["unowned_property_ids"] = np.asarray(property_ids, dtype=np.int32)
    return result


def _defer_management_action(
    loop: GameLoop,
    candidate: GameAction,
    legal: list[GameAction],
) -> GameAction:
    other_management = [
        action for action in legal
        if action.action_type in MANAGEMENT_ACTION_TYPES
        and action != candidate
    ]
    preferred = {
        GameLoop.BEFORE_ROLL: (ActionType.ROLL_DICE,),
        GameLoop.AFTER_ROLL: (ActionType.END_TURN,),
        GameLoop.RESOLVE_DEBT: (
            ActionType.END_DEBT_NEGOTIATIONS,
            ActionType.AUCTION_DEBT_PROPERTY,
            ActionType.DECLARE_BANKRUPTCY,
        ),
    }.get(loop.turn_phase, ())
    for kind in preferred:
        for action in legal:
            if action.action_type == kind and action != candidate:
                return action
    if other_management:
        return other_management[0]
    alternatives = [action for action in legal if action != candidate]
    if alternatives:
        return alternatives[0]
    raise ValueError(f"Não há ação legal para adiar {candidate.action_type}.")


def _simulate_branch(
    source: GameLoop,
    focal_player_id: UUID,
    tile_id: int,
    first_action: GameAction,
    config: GameConfig,
    horizons: tuple[int, ...],
    seed: int,
    tracked_property_ids: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    loop = source.clone()
    _prepare_determinization(loop, config, seed)
    loop.game.record_events = False
    loop.game.history_logs.clear()
    initial_cash = loop.game.player_states[focal_player_id].money
    cash_in = 0.0
    cash_out = 0.0
    minimum_cash = float(initial_cash)
    debt_seen = any(debt.debtor_id == focal_player_id for debt in loop.game.debts)
    bankruptcy_seen = loop.game.player_states[focal_player_id].is_bankrupt
    opportunity_seen: set[int] = set()
    completed_focal_turns = 0
    snapshots: dict[int, dict[str, float]] = {}
    max_actions = max(horizons) * max(100, len(loop.game.turn_order) * 50)

    def record_property_opportunity() -> None:
        """Record an unowned tracked tile offered to the forecast player."""
        if (
            tracked_property_ids
            and loop.current_decision_player_id == focal_player_id
            and loop.turn_phase == GameLoop.AWAITING_PROPERTY_DECISION
            and loop.pending_tile_id in tracked_property_ids
            and loop.game.property_states[loop.pending_tile_id].owner_id is None
        ):
            opportunity_seen.add(loop.pending_tile_id)

    def snapshot() -> dict[str, Any]:
        """Capture forecast metrics for the current simulated horizon."""
        state = loop.game.player_states[focal_player_id]
        return {
            "cash_change": float(state.money - initial_cash),
            "cash_in": cash_in,
            "cash_out": cash_out,
            "ending_cash": float(state.money),
            "minimum_cash": minimum_cash,
            "debt": float(debt_seen),
            "bankruptcy": float(bankruptcy_seen or state.is_bankrupt),
            "in_jail": float(state.is_in_jail),
            "jail_cards": float(len(state.jail_cards)),
            "opportunity_hits": np.asarray(
                [float(tile_id in opportunity_seen) for tile_id in tracked_property_ids],
                dtype=np.float64,
            ),
        }

    action_result = loop.process_action(first_action)
    if action_result["status"] != "SUCCESS":
        raise RuntimeError(f"Ação inicial de previsão rejeitada: {action_result.get('reason')}.")
    cash = loop.game.player_states[focal_player_id].money
    delta = cash - initial_cash
    cash_in += max(0, delta)
    cash_out += max(0, -delta)
    minimum_cash = min(minimum_cash, float(cash))
    debt_seen |= any(debt.debtor_id == focal_player_id for debt in loop.game.debts)
    bankruptcy_seen |= loop.game.player_states[focal_player_id].is_bankrupt

    policy = _ForecastPolicy(focal_player_id, tile_id)
    next_horizon_index = 0
    for _ in range(max_actions):
        if loop.game.game_over or loop.game.truncated:
            break
        record_property_opportunity()
        turn_counter_before = loop.game.turn_counter
        turn_owner_before = loop.game.current_player_id
        cash_before = loop.game.player_states[focal_player_id].money
        action = policy(loop)
        result = loop.process_action(action)
        if result["status"] != "SUCCESS":
            # A baseline can have stale assumptions only if the engine phase
            # changed unexpectedly; stop this trajectory at its last valid state.
            break

        cash_after = loop.game.player_states[focal_player_id].money
        delta = cash_after - cash_before
        cash_in += max(0, delta)
        cash_out += max(0, -delta)
        minimum_cash = min(minimum_cash, float(cash_after))
        debt_seen |= any(debt.debtor_id == focal_player_id for debt in loop.game.debts)
        bankruptcy_seen |= loop.game.player_states[focal_player_id].is_bankrupt
        if (
            turn_owner_before == focal_player_id
            and
            loop.game.turn_counter > turn_counter_before
        ):
            completed_focal_turns += 1

        while (
            next_horizon_index < len(horizons)
            and completed_focal_turns >= horizons[next_horizon_index]
        ):
            snapshots[horizons[next_horizon_index]] = snapshot()
            next_horizon_index += 1

        # Every requested result is fixed once the last horizon is recorded.
        # Continuing until the episode ends wastes hundreds of engine actions
        # per Monte Carlo path and cannot change any forecast value.
        if next_horizon_index == len(horizons):
            break

    final = snapshot()
    for horizon in horizons:
        snapshots.setdefault(horizon, final)
    return {
        name: np.asarray([snapshots[horizon][name] for horizon in horizons], dtype=np.float64)
        for name in ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash", "debt", "bankruptcy")
    } | {
        name: np.asarray([snapshots[horizon][name] for horizon in horizons], dtype=np.float64)
        for name in ("in_jail", "jail_cards")
    } | {
        "opportunity_hits": np.stack(
            [snapshots[horizon]["opportunity_hits"] for horizon in horizons], axis=0
        ),
    }


def _prepare_determinization(loop: GameLoop, config: GameConfig, seed: int) -> None:
    dice_rng = getattr(loop.game.dice, "_rng", None)
    if dice_rng is not None:
        dice_rng.seed(seed)
    held: dict[str, set[str]] = {name: set() for name in ("chance", "community_chest")}
    for player_state in loop.game.player_states.values():
        for deck_name, card in player_state.jail_cards:
            held.setdefault(deck_name, set()).add(card.id)

    source_cards = {
        "chance": config.chance_cards,
        "community_chest": config.community_chest_cards,
    }
    for index, deck_name in enumerate(("chance", "community_chest")):
        deck = loop.game.decks[deck_name]
        # Reconstruct the known deck composition without consulting its hidden order.
        deck._cards = [card for card in source_cards[deck_name] if card.id not in held[deck_name]]
        deck._discard_pile = []
        deck._rng.seed(seed + index + 1)
        deck.shuffle()


def _summarize(horizons, sample_count, option_available, samples_by_scenario):
    statistic_fields = ("cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash")
    summarized: dict[str, np.ndarray] = {}
    for field in statistic_fields:
        values = np.zeros((2, len(horizons), len(FORECAST_STAT_NAMES)), dtype=np.float64)
        for scenario_index, paths in enumerate(samples_by_scenario):
            sample_values = np.stack([path[field] for path in paths], axis=0)
            values[scenario_index, :, 0] = sample_values.mean(axis=0)
            values[scenario_index, :, 1:] = np.quantile(
                sample_values, (0.05, 0.25, 0.50, 0.75, 0.95), axis=0
            ).T
        summarized[field] = values

    debt_probability = np.zeros((2, len(horizons)), dtype=np.float64)
    bankruptcy_probability = np.zeros_like(debt_probability)
    for scenario_index, paths in enumerate(samples_by_scenario):
        debt_probability[scenario_index] = np.stack(
            [path["debt"] for path in paths], axis=0
        ).mean(axis=0)
        bankruptcy_probability[scenario_index] = np.stack(
            [path["bankruptcy"] for path in paths], axis=0
        ).mean(axis=0)

    return PropertyDecisionCashFlowForecast(
        horizons=horizons,
        scenario_names=FORECAST_SCENARIOS,
        statistic_names=FORECAST_STAT_NAMES,
        sample_count=sample_count,
        option_available=option_available,
        cash_change=summarized["cash_change"],
        cash_in=summarized["cash_in"],
        cash_out=summarized["cash_out"],
        ending_cash=summarized["ending_cash"],
        minimum_cash=summarized["minimum_cash"],
        debt_probability=debt_probability,
        bankruptcy_probability=bankruptcy_probability,
    )


def _empty_path(horizon_count: int) -> dict[str, np.ndarray]:
    zeros = np.zeros(horizon_count, dtype=np.float64)
    return {name: zeros.copy() for name in (
        "cash_change", "cash_in", "cash_out", "ending_cash", "minimum_cash", "debt", "bankruptcy"
    )}


def _empty_jail_path(property_count: int) -> dict[str, np.ndarray]:
    path = _empty_path(len(FORECAST_HORIZONS))
    path.update({
        "in_jail": np.zeros(len(FORECAST_HORIZONS), dtype=np.float64),
        "jail_cards": np.zeros(len(FORECAST_HORIZONS), dtype=np.float64),
        "opportunity_hits": np.zeros((len(FORECAST_HORIZONS), property_count), dtype=np.float64),
    })
    return path


def _state_seed(loop: GameLoop, focal_player_id: UUID, tile_id: int) -> int:
    game = loop.game
    public_state = (
        focal_player_id.hex,
        tile_id,
        game.turn_counter,
        loop.turn_phase,
        tuple((pid.hex, game.player_states[pid].money, game.player_states[pid].is_in_jail,
               game.player_states[pid].jail_turns, game.player_states[pid].is_bankrupt,
               game.position_of(pid)) for pid in game.turn_order),
        tuple((tile_id, prop.owner_id.hex if prop.owner_id else None, prop.houses, prop.is_mortgaged)
              for tile_id, prop in sorted(game.property_states.items())),
        tuple((deck_name, tuple(sorted(card.id for card in cards)))
              for deck_name, cards in sorted(
                  (name, [card for card in (*deck._cards, *deck._discard_pile)])
                  for name, deck in game.decks.items()
              )),
    )
    digest = hashlib.blake2b(repr(public_state).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


DEFAULT_HORIZONS = (1, 3, 5)
DEFAULT_SIMULATIONS = 64
MAX_ACTIONS_PER_TURN = 128


@dataclass(frozen=True, slots=True)
class CashFlowHorizonForecast:
    """Empirical cash-flow distribution for one forecast horizon."""

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
    net_cash_flow_samples: tuple[int, ...]
    ending_cash_samples: tuple[int, ...]
    bankrupt_samples: tuple[bool, ...]
    game_over_samples: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class CashFlowForecast:
    """Forecast for one player, with all players following GreedyPolicy."""

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
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int | None = None,
    greedy_reserve: int = 150,
    initial_action: GameAction | None = None,
) -> CashFlowForecast:
    """Estimate the player's liquid-cash outcomes under greedy self-play.

    The current game is cloned for each simulation, so the caller's game state
    is not changed. Built-in dice and deck RNGs are independently reseeded for
    every scenario; reusing ``seed`` with the same state reproduces the result.

    A round is one completed turn per player active at the start of this
    forecast. If a scenario ends before its horizon, the terminal state is used
    for that horizon and later ones. A bankrupt player has zero cash thereafter.
    """
    horizon_values = _validate_horizons(horizons)
    _validate_positive_integer("simulations", simulations)
    _validate_positive_integer("greedy_reserve", greedy_reserve, allow_zero=True)
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise TypeError("seed deve ser um inteiro ou None.")
    if player_id not in loop.game.player_states:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    if initial_action is not None and not isinstance(initial_action, GameAction):
        raise TypeError("initial_action deve ser GameAction ou None.")

    effective_seed = (
        random.SystemRandom().getrandbits(64) if seed is None else seed
    )
    seed_rng = random.Random(effective_seed)
    starting_cash = loop.game.player_states[player_id].money
    turns_per_round = max(1, len(loop.game.active_player_ids))
    start_turn = loop.game.turn_counter
    targets = {
        rounds: start_turn + rounds * turns_per_round
        for rounds in horizon_values
    }
    max_requested_turns = max(targets.values()) - start_turn
    action_limit = max(256, max_requested_turns * MAX_ACTIONS_PER_TURN)

    observations: dict[int, list[tuple[int, bool, bool]]] = {
        rounds: [] for rounds in horizon_values
    }

    for _ in range(simulations):
        scenario = loop.clone()
        scenario.game.record_events = False
        scenario.game.history_logs.clear()
        _reseed_random_sources(scenario, seed_rng)
        policies = {
            candidate: GreedyPolicy(reserve=greedy_reserve)
            for candidate in scenario.game.turn_order
        }
        recorded: set[int] = set()
        actions = 0

        if initial_action is not None:
            result = scenario.process_action(initial_action)
            if result["status"] != "SUCCESS":
                raise ValueError(
                    "A ação inicial do forecast condicionado foi rejeitada: "
                    f"{initial_action.action_type}: {result['reason']}"
                )
            actions += 1

        while len(recorded) < len(horizon_values):
            for rounds in horizon_values:
                if rounds in recorded:
                    continue
                if scenario.game.game_over or scenario.game.turn_counter >= targets[rounds]:
                    state = scenario.game.player_states[player_id]
                    observations[rounds].append(
                        (state.money, state.is_bankrupt, scenario.game.game_over)
                    )
                    recorded.add(rounds)

            if len(recorded) == len(horizon_values):
                break
            if actions >= action_limit:
                raise RuntimeError(
                    "A simulação greedy excedeu o limite de ações antes de "
                    "alcançar o horizonte."
                )

            actor = scenario.current_decision_player_id
            action = policies[actor](scenario)
            result = scenario.process_action(action)
            if result["status"] != "SUCCESS":
                raise RuntimeError(
                    "GreedyPolicy gerou uma ação rejeitada no forecast: "
                    f"{action.action_type}: {result['reason']}"
                )
            actions += 1

    forecasts = tuple(
        _summarize_horizon(
            rounds=rounds,
            requested_turns=rounds * turns_per_round,
            starting_cash=starting_cash,
            observations=observations[rounds],
        )
        for rounds in horizon_values
    )
    return CashFlowForecast(
        player_id=player_id,
        seed=effective_seed,
        simulations=simulations,
        starting_cash=starting_cash,
        policy_name="GreedyPolicy",
        conditioned_on=(initial_action.action_type if initial_action else None),
        horizons=forecasts,
    )


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


def _validate_positive_integer(
    name: str,
    value: int,
    *,
    allow_zero: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} deve ser um inteiro.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "não negativo" if allow_zero else "positivo"
        raise ValueError(f"{name} deve ser {qualifier}.")


def _reseed_random_sources(loop: GameLoop, seed_rng: random.Random) -> None:
    sources = (loop.game.dice, *loop.game.decks.values())
    for source in sources:
        rng = getattr(source, "_rng", None)
        reseed = getattr(rng, "seed", None)
        if callable(reseed):
            reseed(seed_rng.getrandbits(64))


def _summarize_horizon(
    *,
    rounds: int,
    requested_turns: int,
    starting_cash: int,
    observations: list[tuple[int, bool, bool]],
) -> CashFlowHorizonForecast:
    if not observations:
        raise RuntimeError(f"Nenhum cenário foi registrado para {rounds} rodada(s).")

    ending_cash = np.asarray([item[0] for item in observations], dtype=np.float64)
    net_cash_flow = ending_cash - starting_cash
    bankrupt = np.asarray([item[1] for item in observations], dtype=np.float64)
    game_over = np.asarray([item[2] for item in observations], dtype=np.float64)
    flow_p10, flow_p50, flow_p90 = np.quantile(net_cash_flow, (0.1, 0.5, 0.9))
    cash_p10, cash_p50, cash_p90 = np.quantile(ending_cash, (0.1, 0.5, 0.9))

    return CashFlowHorizonForecast(
        horizon_rounds=rounds,
        requested_turns=requested_turns,
        mean_net_cash_flow=float(np.mean(net_cash_flow)),
        std_net_cash_flow=float(np.std(net_cash_flow)),
        p10_net_cash_flow=float(flow_p10),
        median_net_cash_flow=float(flow_p50),
        p90_net_cash_flow=float(flow_p90),
        mean_ending_cash=float(np.mean(ending_cash)),
        p10_ending_cash=float(cash_p10),
        median_ending_cash=float(cash_p50),
        p90_ending_cash=float(cash_p90),
        bankruptcy_probability=float(np.mean(bankrupt)),
        game_over_probability=float(np.mean(game_over)),
        net_cash_flow_samples=tuple(int(value) for value in net_cash_flow),
        ending_cash_samples=tuple(int(value) for value in ending_cash),
        bankrupt_samples=tuple(bool(value) for value in bankrupt),
        game_over_samples=tuple(bool(value) for value in game_over),
    )
