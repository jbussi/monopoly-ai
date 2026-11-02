"""Scenario information and observed behavior for bilateral trade decisions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from app.domain.analytics.economics.game_valuation import (
    PortfolioScenarioValuation,
    PropertyScenarioState,
    analyze_portfolio_scenario,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop, MORTGAGE_TRANSFER_INTEREST_PERCENT
from app.domain.game.trade import TradeContextRecord, TradeOffer
from app.envs.cash_flow_forecast import CashFlowForecast, forecast_cash_flow
from app.infraestructure.game_config.models import GameConfig


@dataclass(frozen=True, slots=True)
class TradeOfferHistoryEntry:
    offer: TradeOffer
    response: str | None
    responder_id: UUID | None
    response_turn: int | None
    context: TradeContextRecord | None


@dataclass(frozen=True, slots=True)
class TradeBehaviorProfile:
    player_id: UUID
    opponent_id: UUID | None
    offers_received: int
    accepted: int
    rejected: int
    countered: int
    offers_made: int
    mean_counteroffer_number_when_responding: float | None
    history: tuple[TradeOfferHistoryEntry, ...]


@dataclass(frozen=True, slots=True)
class TradeParticipantScenario:
    player_id: UUID
    cash_before: int
    cash_after_immediate_exchange: int
    book_net_worth_before: int
    book_net_worth_after_immediate_exchange: int
    book_net_worth_delta: int
    portfolio: PortfolioScenarioValuation
    cash_forecast: CashFlowForecast


@dataclass(frozen=True, slots=True)
class TradeOutcomeScenario:
    outcome: str
    participants: tuple[TradeParticipantScenario, ...]


@dataclass(frozen=True, slots=True)
class TradeMortgageTransferOption:
    tile_id: int
    recipient_id: UUID
    interest_if_kept: int
    redemption_cost: int


@dataclass(frozen=True, slots=True)
class TradeConditionalAnalysis:
    negotiation_id: int
    counteroffer_number: int
    proposer_id: UUID
    recipient_id: UUID
    money_offered: int
    money_requested: int
    properties_offered: tuple[int, ...]
    properties_requested: tuple[int, ...]
    jail_cards_offered: tuple[str, ...]
    jail_cards_requested: tuple[str, ...]
    mortgage_transfer_options: tuple[TradeMortgageTransferOption, ...]
    seed: int
    simulations: int
    scenarios: tuple[TradeOutcomeScenario, ...]
    proposer_behavior: TradeBehaviorProfile
    recipient_behavior: TradeBehaviorProfile


def analyze_pending_trade(
    loop: GameLoop,
    config: GameConfig,
    *,
    horizons: Iterable[int] = (1, 3, 5),
    simulations: int = 64,
    seed: int | None = None,
    greedy_reserve: int = 150,
    discount_rate_per_round: float | None = None,
) -> TradeConditionalAnalysis:
    """Expose accept/reject forecasts and portfolio facts for both players.

    Forecasts force the response once, then use GreedyPolicy for every later
    action by every player. No common utility or recommendation is calculated.
    Portfolio values report whole-portfolio rent estimates and their change.
    """
    negotiation = loop.game.pending_trade
    if negotiation is None:
        raise ValueError("Não há oferta pendente para analisar.")
    offer = negotiation.current_offer
    if loop.current_decision_player_id != offer.recipient_id:
        raise ValueError("A oferta deve estar aguardando resposta do destinatário.")
    if any(loop.game.player_states[player_id].is_bankrupt for player_id in (offer.proposer_id, offer.recipient_id)):
        raise ValueError("Os participantes da oferta precisam estar ativos.")
    horizon_values = _validate_horizons(horizons)
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("simulations deve ser um inteiro positivo.")
    if seed is None:
        effective_seed = random.SystemRandom().getrandbits(64)
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro ou None.")
    else:
        effective_seed = seed

    property_changes: dict[int, PropertyScenarioState] = {}
    for tile_id in offer.properties_offered:
        state = loop.game.property_states[tile_id]
        property_changes[tile_id] = PropertyScenarioState(
            owner_id=offer.recipient_id,
            houses=state.houses,
            is_mortgaged=state.is_mortgaged,
        )
    for tile_id in offer.properties_requested:
        state = loop.game.property_states[tile_id]
        property_changes[tile_id] = PropertyScenarioState(
            owner_id=offer.proposer_id,
            houses=state.houses,
            is_mortgaged=state.is_mortgaged,
        )

    cash_by_player = {
        offer.proposer_id: offer.money_requested - offer.money_offered,
        offer.recipient_id: offer.money_offered - offer.money_requested,
    }
    book_value_by_tile = {}
    for tile_id in (*offer.properties_offered, *offer.properties_requested):
        tile = loop.game.board.get_tile_by_id(tile_id)
        state = loop.game.property_states[tile_id]
        book_value_by_tile[tile_id] = (
            tile.price
            - (tile.mortgage_value if state.is_mortgaged else 0)
            + state.houses * tile.house_cost
        )
    book_delta_by_player = {
        offer.proposer_id: -sum(book_value_by_tile[tile_id] for tile_id in offer.properties_offered)
        + sum(book_value_by_tile[tile_id] for tile_id in offer.properties_requested),
        offer.recipient_id: sum(book_value_by_tile[tile_id] for tile_id in offer.properties_offered)
        - sum(book_value_by_tile[tile_id] for tile_id in offer.properties_requested),
    }
    outcomes: list[TradeOutcomeScenario] = []
    for outcome, action_type in (
        ("ACCEPT", ActionType.ACCEPT_TRADE),
        ("REJECT", ActionType.REJECT_TRADE),
    ):
        changes = property_changes if outcome == "ACCEPT" else {}
        participants = []
        for player_id in (offer.proposer_id, offer.recipient_id):
            portfolio = analyze_portfolio_scenario(
                loop,
                config,
                player_id,
                changes,
                immediate_cash_flow=(cash_by_player[player_id] if outcome == "ACCEPT" else 0),
                horizons=horizon_values,
                discount_rate_per_round=discount_rate_per_round,
            )
            cash_forecast = forecast_cash_flow(
                loop,
                player_id,
                horizons=horizon_values,
                simulations=simulations,
                seed=effective_seed,
                greedy_reserve=greedy_reserve,
                initial_action=GameAction(action_type),
            )
            participants.append(
                TradeParticipantScenario(
                    player_id=player_id,
                    cash_before=loop.game.player_states[player_id].money,
                    cash_after_immediate_exchange=(
                        loop.game.player_states[player_id].money
                        + (cash_by_player[player_id] if outcome == "ACCEPT" else 0)
                    ),
                    book_net_worth_before=loop.game.net_worth(player_id),
                    book_net_worth_after_immediate_exchange=(
                        loop.game.net_worth(player_id)
                        + (cash_by_player[player_id] + book_delta_by_player[player_id]
                           if outcome == "ACCEPT" else 0)
                    ),
                    book_net_worth_delta=(
                        cash_by_player[player_id] + book_delta_by_player[player_id]
                        if outcome == "ACCEPT" else 0
                    ),
                    portfolio=portfolio,
                    cash_forecast=cash_forecast,
                )
            )
        outcomes.append(TradeOutcomeScenario(outcome, tuple(participants)))

    return TradeConditionalAnalysis(
        negotiation_id=offer.negotiation_id,
        counteroffer_number=offer.counteroffer_number,
        proposer_id=offer.proposer_id,
        recipient_id=offer.recipient_id,
        money_offered=offer.money_offered,
        money_requested=offer.money_requested,
        properties_offered=offer.properties_offered,
        properties_requested=offer.properties_requested,
        jail_cards_offered=offer.jail_cards_offered,
        jail_cards_requested=offer.jail_cards_requested,
        mortgage_transfer_options=_mortgage_transfer_options(loop, offer),
        seed=effective_seed,
        simulations=simulations,
        scenarios=tuple(outcomes),
        proposer_behavior=trade_behavior_profile(
            loop, offer.proposer_id, offer.recipient_id
        ),
        recipient_behavior=trade_behavior_profile(
            loop, offer.recipient_id, offer.proposer_id
        ),
    )


def analyze_trade_proposal(
    loop: GameLoop,
    config: GameConfig,
    proposal: GameAction,
    *,
    horizons: Iterable[int] = (1, 3, 5),
    simulations: int = 64,
    seed: int | None = None,
    greedy_reserve: int = 150,
    discount_rate_per_round: float | None = None,
) -> TradeConditionalAnalysis:
    """Evaluate a candidate proposal without changing the caller's game."""
    if proposal.action_type != ActionType.PROPOSE_TRADE:
        raise ValueError("proposal deve ser uma ação PROPOSE_TRADE.")
    if loop.game.pending_trade is not None:
        raise ValueError("Já existe uma oferta aguardando resposta.")
    scenario = loop.clone()
    scenario.game.record_events = False
    result = scenario.process_action(proposal)
    if result["status"] != "SUCCESS":
        raise ValueError(f"A proposta candidata é ilegal: {result['reason']}")
    return analyze_pending_trade(
        scenario,
        config,
        horizons=horizons,
        simulations=simulations,
        seed=seed,
        greedy_reserve=greedy_reserve,
        discount_rate_per_round=discount_rate_per_round,
    )


def trade_behavior_profile(
    loop: GameLoop,
    player_id: UUID,
    opponent_id: UUID | None = None,
) -> TradeBehaviorProfile:
    """Summarize recorded responses while retaining the linked offer history."""
    if player_id not in loop.game.player_states:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    if opponent_id is not None and opponent_id not in loop.game.player_states:
        raise KeyError(f"Oponente desconhecido: {opponent_id!r}")
    decisions = {
        (item.negotiation_id, item.counteroffer_number): item
        for item in loop.game.trade_decision_history
    }
    contexts = {
        (item.negotiation_id, item.counteroffer_number): item
        for item in loop.game.trade_context_history
    }
    entries = []
    received = made = accepted = rejected = countered = 0
    response_numbers = []
    for offer in loop.game.trade_history:
        if player_id not in (offer.proposer_id, offer.recipient_id):
            continue
        other_id = offer.recipient_id if offer.proposer_id == player_id else offer.proposer_id
        if opponent_id is not None and other_id != opponent_id:
            continue
        if offer.proposer_id == player_id:
            made += 1
        else:
            received += 1
        decision = decisions.get((offer.negotiation_id, offer.counteroffer_number))
        entries.append(
            TradeOfferHistoryEntry(
                offer=offer,
                response=decision.response if decision else None,
                responder_id=decision.responder_id if decision else None,
                response_turn=decision.turn if decision else None,
                context=contexts.get((offer.negotiation_id, offer.counteroffer_number)),
            )
        )
        if decision is not None and offer.recipient_id == player_id:
            response_numbers.append(offer.counteroffer_number)
            if decision.response == "ACCEPTED":
                accepted += 1
            elif decision.response == "REJECTED":
                rejected += 1
            elif decision.response == "COUNTERED":
                countered += 1
    mean_counter = (
        sum(response_numbers) / len(response_numbers) if response_numbers else None
    )
    return TradeBehaviorProfile(
        player_id=player_id,
        opponent_id=opponent_id,
        offers_received=received,
        accepted=accepted,
        rejected=rejected,
        countered=countered,
        offers_made=made,
        mean_counteroffer_number_when_responding=mean_counter,
        history=tuple(entries),
    )


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(horizons)
    except TypeError as exc:
        raise TypeError("horizons deve ser uma coleção de inteiros positivos.") from exc
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("horizons deve conter inteiros positivos.")
    if len(set(values)) != len(values):
        raise ValueError("horizons não pode conter valores duplicados.")
    return tuple(sorted(values))


def _mortgage_transfer_options(
    loop: GameLoop,
    offer: TradeOffer,
) -> tuple[TradeMortgageTransferOption, ...]:
    transferred = (
        (offer.properties_offered, offer.recipient_id),
        (offer.properties_requested, offer.proposer_id),
    )
    options = []
    for tile_ids, recipient_id in transferred:
        for tile_id in tile_ids:
            property_state = loop.game.property_states[tile_id]
            if not property_state.is_mortgaged:
                continue
            tile = loop.game.board.get_tile_by_id(tile_id)
            interest = (
                tile.mortgage_value * MORTGAGE_TRANSFER_INTEREST_PERCENT + 99
            ) // 100
            options.append(
                TradeMortgageTransferOption(
                    tile_id=tile_id,
                    recipient_id=recipient_id,
                    interest_if_kept=interest,
                    redemption_cost=loop.unmortgage_cost(tile),
                )
            )
    return tuple(options)
