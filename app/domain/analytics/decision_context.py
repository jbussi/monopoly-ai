"""Action-conditioned decision information assembled for an agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping
from uuid import UUID

import numpy as np

from app.domain.analytics.economics.game_valuation import (
    PortfolioScenarioValuation,
    PropertyScenarioState,
    analyze_portfolio_scenario,
)
from app.domain.analytics.economics.auction_analysis import (
    AuctionDecisionAnalysis,
    analyze_auction_decision,
)
from app.domain.analytics.economics.debt_analysis import (
    DebtResolutionAnalysis,
    analyze_debt_resolution,
)
from app.domain.analytics.economics.jail_opportunity import (
    JailOpportunityAnalysis,
    analyze_jail_exit_opportunities,
)
from app.domain.analytics.economics.trade_analysis import (
    TradeConditionalAnalysis,
    analyze_pending_trade,
    analyze_trade_proposal,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.cash_flow_forecast import CashFlowForecast, forecast_cash_flow
from app.envs.baselines import Policy
from app.infraestructure.game_config.models import GameConfig


@dataclass(frozen=True, slots=True)
class ActionScenarioInformation:
    """Observed metrics for one explicit first action and its rollout."""

    action: GameAction
    cash_before: int
    cash_after_immediate_action: int
    cash_delta_immediate: int
    book_net_worth_before: int
    book_net_worth_after_immediate_action: int
    book_net_worth_delta: int
    portfolio: PortfolioScenarioValuation | None
    cash_forecast: CashFlowForecast


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """State facts plus conditional outcome data; contains no action ranking."""

    player_id: UUID
    phase: str
    candidate_action: GameAction
    baseline_action: GameAction | None
    seed: int
    simulations: int
    candidate: ActionScenarioInformation | None = None
    baseline: ActionScenarioInformation | None = None
    trade_analysis: TradeConditionalAnalysis | None = None
    jail_analysis: JailOpportunityAnalysis | None = None
    auction_analysis: AuctionDecisionAnalysis | None = None
    debt_analysis: DebtResolutionAnalysis | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the stable, JSON-compatible decision-information contract."""
        return decision_context_to_dict(self)


DECISION_CONTEXT_SCHEMA_VERSION = 1


def decision_context_to_dict(context: DecisionContext) -> dict[str, object]:
    """Serialize a decision context without discarding scenario detail.

    The schema keeps action-specific sections and optional values explicit.
    Variable-length histories, candidates, and horizon lists remain arrays;
    consumers should use the accompanying action/round identifiers instead of
    assuming positions imply meaning.
    """
    if not isinstance(context, DecisionContext):
        raise TypeError("context deve ser DecisionContext.")
    result = _to_primitive(context)
    assert isinstance(result, dict)
    return {"schema_version": DECISION_CONTEXT_SCHEMA_VERSION, **result}


def _to_primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return _to_primitive(value.tolist())
    if isinstance(value, np.generic):
        return _to_primitive(value.item())
    if isinstance(value, Mapping):
        return {str(_to_primitive(key)): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_primitive(item) for item in sorted(value, key=repr)]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("DecisionContext não pode conter números não finitos.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Tipo não serializável em DecisionContext: {type(value).__name__}.")


def evaluate_candidate(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    candidate_action: GameAction,
    *,
    baseline_action: GameAction | None = None,
    horizons: Iterable[int] = (1, 2),
    simulations: int = 64,
    seed: int | None = None,
    greedy_reserve: int = 150,
    valuation_horizons: Iterable[int] = (1, 3, 5),
    discount_rate_per_round: float | None = None,
    opponent_policies: Mapping[UUID, Policy] | None = None,
    auction_bid_candidates: Iterable[int] | None = None,
    debt_action_candidates: Iterable[GameAction] | None = None,
) -> DecisionContext:
    """Return conditional information for a legal candidate, without ranking it.

    Ordinary decisions require an explicit legal baseline action. Trade
    proposals, trade responses/counteroffers, and jail exits use their paired
    domain analyses because their natural outcomes contain additional actors
    or follow-up decisions.
    """
    if loop.current_decision_player_id != player_id:
        raise ValueError("player_id precisa ser o decisor atual.")
    if not isinstance(candidate_action, GameAction):
        raise TypeError("candidate_action deve ser GameAction.")
    _require_legal(loop, candidate_action, "candidate_action")
    effective_seed = _resolve_seed(seed)
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("simulations deve ser um inteiro positivo.")
    horizon_values = _validate_horizons(horizons)
    valuation_values = _validate_horizons(valuation_horizons)
    phase = loop.turn_phase

    if phase == GameLoop.RESOLVE_DEBT:
        debt_actions = list(loop.legal_actions())
        debt_actions.extend(debt_action_candidates or ())
        if loop.game.pending_trade is not None:
            debt_actions.extend(
                action for action in loop.trade_response_actions()
                if loop.validate(action) is None
            )
        debt_actions.append(candidate_action)
        debt_actions = [
            action for index, action in enumerate(debt_actions)
            if not any(action == earlier for earlier in debt_actions[:index])
        ]
        analysis = analyze_debt_resolution(
            loop,
            config,
            candidate_actions=debt_actions,
            horizons=horizon_values,
            simulations=simulations,
            seed=effective_seed,
            greedy_reserve=greedy_reserve,
        )
        return DecisionContext(
            player_id, phase, candidate_action, None,
            effective_seed, simulations, debt_analysis=analysis,
        )

    if candidate_action.action_type in {
        ActionType.AUCTION_BID,
        ActionType.AUCTION_WITHDRAW,
    }:
        bid_candidates = auction_bid_candidates
        if bid_candidates is None:
            bid_candidates = (
                (candidate_action.payload["amount"],)
                if candidate_action.action_type == ActionType.AUCTION_BID
                else None
            )
        else:
            bid_candidates = tuple(bid_candidates)
            if (
                candidate_action.action_type == ActionType.AUCTION_BID
                and candidate_action.payload["amount"] not in bid_candidates
            ):
                raise ValueError("auction_bid_candidates deve incluir a ação candidata.")
        analysis = analyze_auction_decision(
            loop,
            config,
            candidate_bids=bid_candidates,
            horizons=horizon_values,
            simulations=simulations,
            seed=effective_seed,
            greedy_reserve=greedy_reserve,
            opponent_policies=opponent_policies,
            include_withdraw=True,
            discount_rate_per_round=discount_rate_per_round,
        )
        return DecisionContext(
            player_id, phase, candidate_action, None,
            effective_seed, simulations, auction_analysis=analysis,
        )

    if candidate_action.action_type == ActionType.PROPOSE_TRADE:
        analysis = analyze_trade_proposal(
            loop,
            config,
            candidate_action,
            horizons=horizon_values,
            simulations=simulations,
            seed=effective_seed,
            greedy_reserve=greedy_reserve,
            discount_rate_per_round=discount_rate_per_round,
        )
        return DecisionContext(
            player_id, phase, candidate_action, None,
            effective_seed, simulations, trade_analysis=analysis,
        )

    if candidate_action.action_type in {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.COUNTER_TRADE,
    }:
        trade_loop = loop
        if candidate_action.action_type == ActionType.COUNTER_TRADE:
            trade_loop = loop.clone()
            trade_loop.game.record_events = False
            result = trade_loop.process_action(candidate_action)
            if result["status"] != "SUCCESS":
                raise ValueError(f"A contraproposta é ilegal: {result['reason']}")
        analysis = analyze_pending_trade(
            trade_loop,
            config,
            horizons=horizon_values,
            simulations=simulations,
            seed=effective_seed,
            greedy_reserve=greedy_reserve,
            discount_rate_per_round=discount_rate_per_round,
        )
        return DecisionContext(
            player_id, phase, candidate_action, None,
            effective_seed, simulations, trade_analysis=analysis,
        )

    if candidate_action.action_type in {
        ActionType.PAY_JAIL_FINE,
        ActionType.USE_JAIL_CARD,
        ActionType.ROLL_DICE,
    } and loop.game.player_states[player_id].is_in_jail:
        analysis = analyze_jail_exit_opportunities(
            loop,
            config,
            player_id,
            horizons=horizon_values,
            simulations=simulations,
            seed=effective_seed,
            greedy_reserve=greedy_reserve,
            valuation_horizons=valuation_values,
            discount_rate_per_round=discount_rate_per_round,
        )
        return DecisionContext(
            player_id, phase, candidate_action, None,
            effective_seed, simulations, jail_analysis=analysis,
        )

    if baseline_action is None:
        raise ValueError(
            "Informe baseline_action para comparar a alternativa sem impor "
            "uma preferência estratégica."
        )
    if not isinstance(baseline_action, GameAction):
        raise TypeError("baseline_action deve ser GameAction.")
    if candidate_action == baseline_action:
        raise ValueError("candidate_action e baseline_action precisam diferir.")
    _require_legal(loop, baseline_action, "baseline_action")

    candidate = _evaluate_scenario(
        loop, config, player_id, candidate_action,
        horizons=horizon_values,
        simulations=simulations,
        seed=effective_seed,
        greedy_reserve=greedy_reserve,
        valuation_horizons=valuation_values,
        discount_rate_per_round=discount_rate_per_round,
    )
    baseline = _evaluate_scenario(
        loop, config, player_id, baseline_action,
        horizons=horizon_values,
        simulations=simulations,
        seed=effective_seed,
        greedy_reserve=greedy_reserve,
        valuation_horizons=valuation_values,
        discount_rate_per_round=discount_rate_per_round,
    )
    return DecisionContext(
        player_id, phase, candidate_action, baseline_action,
        effective_seed, simulations, candidate=candidate, baseline=baseline,
    )


def _evaluate_scenario(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    action: GameAction,
    *,
    horizons: tuple[int, ...],
    simulations: int,
    seed: int,
    greedy_reserve: int,
    valuation_horizons: tuple[int, ...],
    discount_rate_per_round: float | None,
) -> ActionScenarioInformation:
    scenario = loop.clone()
    scenario.game.record_events = False
    scenario.game.history_logs.clear()
    result = scenario.process_action(action)
    if result["status"] != "SUCCESS":
        raise ValueError(f"A ação candidata foi rejeitada: {result['reason']}")

    initial_state = loop.game.player_states[player_id]
    ending_state = scenario.game.player_states[player_id]
    portfolio = None
    if not ending_state.is_bankrupt:
        property_changes = {}
        for tile in loop.game.board.property_tiles:
            before = loop.game.property_states[tile.tile_id]
            after = scenario.game.property_states[tile.tile_id]
            if (
                before.owner_id != after.owner_id
                or before.houses != after.houses
                or before.is_mortgaged != after.is_mortgaged
            ):
                property_changes[tile.tile_id] = PropertyScenarioState(
                    owner_id=after.owner_id,
                    houses=after.houses,
                    is_mortgaged=after.is_mortgaged,
                )
        portfolio = analyze_portfolio_scenario(
            loop,
            config,
            player_id,
            property_changes,
            immediate_cash_flow=ending_state.money - initial_state.money,
            horizons=valuation_horizons,
            discount_rate_per_round=discount_rate_per_round,
        )

    forecast = forecast_cash_flow(
        loop,
        player_id,
        config=config,
        horizons=horizons,
        simulations=simulations,
        seed=seed,
        greedy_reserve=greedy_reserve,
        initial_action=action,
    )
    return ActionScenarioInformation(
        action=action,
        cash_before=initial_state.money,
        cash_after_immediate_action=ending_state.money,
        cash_delta_immediate=ending_state.money - initial_state.money,
        book_net_worth_before=loop.game.net_worth(player_id),
        book_net_worth_after_immediate_action=scenario.game.net_worth(player_id),
        book_net_worth_delta=(
            scenario.game.net_worth(player_id) - loop.game.net_worth(player_id)
        ),
        portfolio=portfolio,
        cash_forecast=forecast,
    )


def _require_legal(loop: GameLoop, action: GameAction, name: str) -> None:
    reason = loop.validate(action)
    if reason is not None:
        raise ValueError(f"{name} é ilegal: {reason}")


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().getrandbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro ou None.")
    return seed


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(horizons)
    except TypeError as exc:
        raise TypeError("horizons deve ser uma coleção de inteiros positivos.") from exc
    if (
        not values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values)
    ):
        raise ValueError("horizons deve conter inteiros positivos.")
    if len(set(values)) != len(values):
        raise ValueError("horizons não pode conter valores duplicados.")
    return tuple(sorted(values))
