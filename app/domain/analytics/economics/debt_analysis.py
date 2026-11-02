"""Action-conditioned information for debt resolution and insolvency."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.cash_flow_forecast import CashFlowForecast, forecast_cash_flow


@dataclass(frozen=True, slots=True)
class DebtPlayerImpact:
    """Immediate and forecast outcomes for a player affected by an option."""

    player_id: UUID
    cash_before: int
    cash_after: int
    cash_delta: int
    net_worth_before: int
    net_worth_after: int
    net_worth_delta: int
    is_bankrupt_after_action: bool
    cash_forecast: CashFlowForecast


@dataclass(frozen=True, slots=True)
class DebtActionScenario:
    """Observed consequences of one explicit debt-resolution action."""

    action: GameAction
    impacts: tuple[DebtPlayerImpact, ...]


@dataclass(frozen=True, slots=True)
class DebtResolutionAnalysis:
    """Candidate action outcomes, without strategic ranking or recommendation."""

    debtor_id: UUID
    creditor_ids: tuple[UUID, ...]
    seed: int
    simulations: int
    horizons: tuple[int, ...]
    scenarios: tuple[DebtActionScenario, ...]


_DEBT_ACTIONS = {
    ActionType.SELL_HOUSE,
    ActionType.MORTGAGE,
    ActionType.PROPOSE_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.COUNTER_TRADE,
    ActionType.END_DEBT_NEGOTIATIONS,
    ActionType.AUCTION_DEBT_PROPERTY,
    ActionType.DECLARE_BANKRUPTCY,
}


def analyze_debt_resolution(
    loop: GameLoop,
    *,
    candidate_actions: Iterable[GameAction] | None = None,
    horizons: Iterable[int] = (1, 3, 5),
    simulations: int = 64,
    seed: int | None = None,
    greedy_reserve: int = 150,
) -> DebtResolutionAnalysis:
    """Forecast explicit legal options in ``RESOLVE_DEBT`` for debtor/creditors.

    By default, concrete actions from ``GameLoop.legal_actions`` are included.
    Structured trade offers/counters can be supplied in ``candidate_actions``.
    Every scenario starts from the same state and uses the same seed, making
    comparisons reproducible. The rollout uses GreedyPolicy for every player.
    """
    if loop.turn_phase != GameLoop.RESOLVE_DEBT or loop.game.debt_resolution is None:
        raise ValueError("A análise de dívida exige uma decisão na fase RESOLVE_DEBT.")
    debtor_id = loop.game.debt_resolution.debtor_id
    if loop.current_decision_player_id != debtor_id:
        raise ValueError("O devedor precisa ser o decisor atual.")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("simulations deve ser um inteiro positivo.")
    try:
        raw_horizons = tuple(horizons)
    except TypeError as exc:
        raise TypeError("horizons deve conter inteiros positivos.") from exc
    if not raw_horizons or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in raw_horizons
    ):
        raise ValueError("horizons deve conter inteiros positivos.")
    if len(set(raw_horizons)) != len(raw_horizons):
        raise ValueError("horizons não pode conter valores duplicados.")
    horizon_values = tuple(sorted(raw_horizons))
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed deve ser um inteiro ou None.")
    effective_seed = random.SystemRandom().getrandbits(64) if seed is None else seed

    if candidate_actions is None:
        available = [
            action for action in loop.legal_actions()
            if action.action_type in _DEBT_ACTIONS
        ]
        available.extend(
            action for action in loop.trade_response_actions()
            if loop.validate(action) is None
        )
        actions = tuple(available)
    else:
        actions = tuple(candidate_actions)
    if not actions:
        raise ValueError("Nenhuma ação concreta de resolução de dívida foi informada.")
    if any(action == previous for index, action in enumerate(actions) for previous in actions[:index]):
        raise ValueError("candidate_actions não pode conter ações duplicadas.")
    for action in actions:
        if not isinstance(action, GameAction):
            raise TypeError("candidate_actions deve conter apenas GameAction.")
        if action.action_type not in _DEBT_ACTIONS:
            raise ValueError(f"Ação fora do escopo de dívida: {action.action_type}.")
        reason = loop.validate(action)
        if reason is not None:
            raise ValueError(f"Ação {action.action_type} ilegal: {reason}")

    creditor_ids = tuple(dict.fromkeys(
        debt.creditor_id for debt in loop.game.debts
        if debt.debtor_id == debtor_id and debt.creditor_id is not None
    ))
    affected_ids = (debtor_id, *creditor_ids)
    scenarios: list[DebtActionScenario] = []
    for action_index, action in enumerate(actions):
        branch = loop.clone()
        branch.game.record_events = False
        branch.game.history_logs.clear()
        result = branch.process_action(action)
        if result["status"] != "SUCCESS":
            raise ValueError(f"A ação {action.action_type} foi rejeitada: {result['reason']}")
        impacts: list[DebtPlayerImpact] = []
        for player_id in affected_ids:
            before = loop.game
            after = branch.game
            forecast = forecast_cash_flow(
                loop,
                player_id,
                horizons=horizon_values,
                simulations=simulations,
                seed=effective_seed + action_index,
                greedy_reserve=greedy_reserve,
                initial_action=action,
            )
            nw_before = before.net_worth(player_id)
            nw_after = after.net_worth(player_id)
            impacts.append(DebtPlayerImpact(
                player_id=player_id,
                cash_before=before.player_states[player_id].money,
                cash_after=after.player_states[player_id].money,
                cash_delta=after.player_states[player_id].money - before.player_states[player_id].money,
                net_worth_before=nw_before,
                net_worth_after=nw_after,
                net_worth_delta=nw_after - nw_before,
                is_bankrupt_after_action=after.player_states[player_id].is_bankrupt,
                cash_forecast=forecast,
            ))
        scenarios.append(DebtActionScenario(action, tuple(impacts)))

    return DebtResolutionAnalysis(
        debtor_id=debtor_id,
        creditor_ids=creditor_ids,
        seed=effective_seed,
        simulations=simulations,
        horizons=horizon_values,
        scenarios=tuple(scenarios),
    )
