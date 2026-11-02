"""Bid-conditioned auction outcomes, portfolio information, and bidder history."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Iterable, Mapping
from uuid import UUID

import numpy as np

from app.domain.analytics.economics.game_valuation import (
    PortfolioScenarioValuation,
    PropertyScenarioState,
    analyze_portfolio_scenario,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.auction import (
    AuctionDecisionRecord,
    AuctionResultRecord,
    AuctionSessionRecord,
)
from app.domain.game.game_loop import (
    AUCTION_BID_INCREMENTS,
    MORTGAGE_TRANSFER_INTEREST_PERCENT,
    GameLoop,
)
from app.envs.baselines import GreedyPolicy, Policy
from app.infraestructure.game_config.models import GameConfig


DEFAULT_AUCTION_HORIZONS = (1, 3, 5)
DEFAULT_AUCTION_SIMULATIONS = 64


@dataclass(frozen=True, slots=True)
class AuctionHistoryEntry:
    session: AuctionSessionRecord
    decision: AuctionDecisionRecord
    result: AuctionResultRecord | None


@dataclass(frozen=True, slots=True)
class AuctionBehaviorProfile:
    player_id: UUID
    auctions_participated: int
    bids: int
    withdrawals: int
    wins: int
    bid_rate: float
    win_rate_per_auction: float
    mean_bid_to_property_price_ratio: float | None
    mean_max_bid_to_property_price_ratio: float | None
    mean_withdraw_price_to_property_price_ratio: float | None
    history: tuple[AuctionHistoryEntry, ...]


@dataclass(frozen=True, slots=True)
class AuctionPropertyValuation:
    tile_id: int
    name: str
    group: str
    market_price: int
    candidate_bid: int
    owned_group_properties_before: int
    owned_group_properties_after: int
    group_size: int
    completes_group: bool
    book_asset_value: int
    portfolio: PortfolioScenarioValuation


@dataclass(frozen=True, slots=True)
class AuctionHorizonEffect:
    horizon_rounds: int
    requested_turns: int
    mean_cash_delta: float
    p10_cash_delta: float
    median_cash_delta: float
    p90_cash_delta: float
    bankruptcy_probability: float
    mean_book_net_worth_delta: float


@dataclass(frozen=True, slots=True)
class AuctionPlayerEffect:
    player_id: UUID
    probability_wins: float
    probability_owns_auctioned_property_after_result: float
    mean_cash_delta_at_result: float
    mean_book_net_worth_delta_at_result: float
    mean_cash_received_from_auction: float
    horizons: tuple[AuctionHorizonEffect, ...]


@dataclass(frozen=True, slots=True)
class AuctionCandidateOutcome:
    candidate_bid: int | None
    action_type: str
    simulations: int
    probability_no_bid: float
    probability_candidate_bidder_wins: float
    mean_final_price: float
    mean_final_price_if_candidate_wins: float | None
    p10_final_price_if_candidate_wins: float | None
    median_final_price_if_candidate_wins: float | None
    p90_final_price_if_candidate_wins: float | None
    winner_probabilities: tuple[tuple[UUID, float], ...]
    property_valuation_at_candidate_bid: AuctionPropertyValuation | None
    property_valuation_at_candidate_bid_if_redeemed: AuctionPropertyValuation | None
    mortgage_interest_if_kept: int | None
    mortgage_redemption_cost: int | None
    player_effects: tuple[AuctionPlayerEffect, ...]


@dataclass(frozen=True, slots=True)
class AuctionDecisionAnalysis:
    auction_id: int
    auction_type: str
    tile_id: int
    property_name: str
    property_group: str
    property_market_price: int
    property_is_mortgaged: bool
    current_highest_bid: int
    current_highest_bidder: UUID | None
    current_actor_id: UUID
    seller_id: UUID | None
    bidder_cash: int
    legal_bid_minimum: int
    legal_bid_maximum: int
    candidate_bids: tuple[int, ...]
    horizons: tuple[int, ...]
    simulations: int
    seed: int
    continuation_policy: str
    outcomes: tuple[AuctionCandidateOutcome, ...]
    opponent_behavior: tuple[AuctionBehaviorProfile, ...]


def analyze_auction_decision(
    loop: GameLoop,
    config: GameConfig,
    *,
    candidate_bids: Iterable[int] | None = None,
    horizons: Iterable[int] = DEFAULT_AUCTION_HORIZONS,
    simulations: int = DEFAULT_AUCTION_SIMULATIONS,
    seed: int | None = None,
    greedy_reserve: int = 150,
    opponent_policies: Mapping[UUID, Policy] | None = None,
    include_withdraw: bool = True,
    discount_rate_per_round: float | None = None,
) -> AuctionDecisionAnalysis:
    """Compare explicit legal bids, and optionally withdrawal, by simulation.

    Each candidate is the acting player's forced first decision. Later actions
    follow supplied per-player policies or GreedyPolicy. Paired candidates use
    common random seeds. The returned metrics describe outcomes and history;
    they do not rank or select bids.
    """
    if loop.turn_phase != GameLoop.AUCTION:
        raise ValueError("A análise exige que a partida esteja na fase AUCTION.")
    actor = loop.current_decision_player_id
    auction_id = loop.current_auction_id
    tile_id = loop.pending_tile_id
    if auction_id is None or tile_id is None:
        raise ValueError("Não há leilão ativo para analisar.")
    if loop.game.player_states[actor].is_bankrupt:
        raise ValueError("Um jogador falido não pode decidir no leilão.")
    tile = loop.game.board.get_tile_by_id(tile_id)
    if not hasattr(tile, "price") or not hasattr(tile, "group"):
        raise ValueError("O imóvel em leilão não tem dados de valuation compatíveis.")
    horizon_values = _validate_positive_ints("horizons", horizons)
    _validate_positive_int("simulations", simulations)
    _validate_positive_int("greedy_reserve", greedy_reserve, allow_zero=True)
    effective_seed = _resolve_seed(seed)
    legal_actions = loop.legal_actions()
    legal_bid_actions = [a for a in legal_actions if a.action_type == ActionType.AUCTION_BID]
    bidder_cash = loop.game.player_states[actor].money
    legal_minimum = max(loop.rules.auction_min_bid, loop.auction_highest_bid + 1)
    requested_bid_values = None
    if candidate_bids is None:
        bids = {int(a.payload["amount"]) for a in legal_bid_actions}
        for amount in (
            legal_minimum,
            tile.price,
            min(bidder_cash, max(legal_minimum, tile.price // 2)),
            min(bidder_cash, max(legal_minimum, (tile.price * 3) // 4)),
            bidder_cash,
        ):
            bids.add(int(amount))
        bid_values = tuple(sorted(bids))
    else:
        requested_bid_values = _validate_candidate_bids(candidate_bids)
        bid_values = requested_bid_values
    bid_values = tuple(
        amount for amount in bid_values
        if amount >= legal_minimum and amount <= bidder_cash
    )
    for amount in bid_values:
        reason = loop.validate(GameAction(ActionType.AUCTION_BID, {"amount": amount}))
        if reason:
            raise ValueError(f"Lance candidato ${amount} ilegal: {reason}")
    if requested_bid_values is not None and len(bid_values) != len(requested_bid_values):
        raise ValueError("Todos os lances candidatos precisam estar dentro dos limites legais.")

    withdraw_legal = any(a.action_type == ActionType.AUCTION_WITHDRAW for a in legal_actions)
    alternatives: list[tuple[int | None, GameAction]] = [
        (amount, GameAction(ActionType.AUCTION_BID, {"amount": amount}))
        for amount in bid_values
    ]
    if include_withdraw and withdraw_legal:
        alternatives.append((None, GameAction(ActionType.AUCTION_WITHDRAW)))
    if not alternatives:
        raise ValueError("Não há lance legal ou opção de desistência para simular.")

    auction_record = next(
        (item for item in reversed(loop.game.auction_history) if item.auction_id == auction_id),
        None,
    )
    if auction_record is None:
        raise ValueError("O histórico da sessão de leilão não está disponível.")
    scenario_rng = random.Random(effective_seed)
    scenario_seeds = [scenario_rng.getrandbits(64) for _ in range(simulations)]
    turns_per_round = max(1, len(loop.game.active_player_ids))
    start_turn = loop.game.turn_counter
    targets = {
        rounds: start_turn + rounds * turns_per_round
        for rounds in horizon_values
    }
    action_limit = max(
        1024,
        (max(targets.values()) - start_turn) * 128
        + sum(loop.game.player_states[p].money for p in loop.game.active_player_ids),
    )
    starting_cash = {
        player_id: loop.game.player_states[player_id].money
        for player_id in loop.game.turn_order
    }
    starting_net_worth = {
        player_id: loop.game.net_worth(player_id)
        for player_id in loop.game.turn_order
    }
    outcomes = []
    for candidate_bid, initial_action in alternatives:
        simulations_results = _simulate_candidate(
            loop,
            actor,
            auction_id,
            initial_action,
            scenario_seeds,
            targets,
            action_limit,
            starting_cash,
            starting_net_worth,
            horizon_values,
            greedy_reserve,
            opponent_policies,
        )
        valuation = None
        redeemed_valuation = None
        if candidate_bid is not None:
            mortgage_costs = (
                _mortgage_costs(loop, tile_id)
                if auction_record.auction_type == "DEBT"
                and loop.game.property_states[tile_id].is_mortgaged
                else (None, None)
            )
            valuation = _value_auction_property(
                loop,
                config,
                actor,
                tile,
                candidate_bid,
                horizons=horizon_values,
                discount_rate_per_round=discount_rate_per_round,
                additional_cash_cost=(mortgage_costs[0] or 0),
            )
            if mortgage_costs[1] is not None:
                redeemed_valuation = _value_auction_property(
                    loop,
                    config,
                    actor,
                    tile,
                    candidate_bid,
                    horizons=horizon_values,
                    discount_rate_per_round=discount_rate_per_round,
                    mortgaged_override=False,
                    additional_cash_cost=mortgage_costs[1],
                )
        else:
            mortgage_costs = (None, None)
        outcomes.append(
            _summarize_candidate(
                actor=actor,
                tile_id=tile_id,
                candidate_bid=candidate_bid,
                action_type=initial_action.action_type,
                simulations_results=simulations_results,
                horizons=horizon_values,
                turns_per_round=turns_per_round,
                starting_cash=starting_cash,
                starting_net_worth=starting_net_worth,
                valuation=valuation,
                mortgaged=loop.game.property_states[tile_id].is_mortgaged,
                mortgage_costs=mortgage_costs,
                redeemed_valuation=redeemed_valuation,
            )
        )

    active_opponents = [player_id for player_id in auction_record.participants if player_id != actor]
    return AuctionDecisionAnalysis(
        auction_id=auction_id,
        auction_type=auction_record.auction_type,
        tile_id=tile_id,
        property_name=tile.name,
        property_group=tile.group,
        property_market_price=tile.price,
        property_is_mortgaged=loop.game.property_states[tile_id].is_mortgaged,
        current_highest_bid=loop.auction_highest_bid,
        current_highest_bidder=loop.auction_highest_bidder,
        current_actor_id=actor,
        seller_id=auction_record.seller_id,
        bidder_cash=bidder_cash,
        legal_bid_minimum=legal_minimum,
        legal_bid_maximum=bidder_cash,
        candidate_bids=bid_values,
        horizons=horizon_values,
        simulations=simulations,
        seed=effective_seed,
        continuation_policy=(
            "GreedyPolicy" if not opponent_policies else "per-player supplied policies; GreedyPolicy fallback"
        ),
        outcomes=tuple(outcomes),
        opponent_behavior=tuple(
            auction_behavior_profile(loop, player_id)
            for player_id in active_opponents
        ),
    )


def auction_behavior_profile(
    loop: GameLoop,
    player_id: UUID,
) -> AuctionBehaviorProfile:
    """Return descriptive, contextualized auction actions for one player."""
    if player_id not in loop.game.player_states:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    sessions = {item.auction_id: item for item in loop.game.auction_history}
    results = {item.auction_id: item for item in loop.game.auction_result_history}
    history = []
    bid_amount_ratios = []
    max_bid_ratio_by_auction: dict[int, float] = {}
    withdrawn_price_ratios = []
    bids = withdrawals = wins = 0
    participated_ids = set()
    for decision in loop.game.auction_decision_history:
        if decision.player_id != player_id:
            continue
        session = sessions.get(decision.auction_id)
        if session is None:
            continue
        participated_ids.add(session.auction_id)
        result = results.get(session.auction_id)
        history.append(AuctionHistoryEntry(session, decision, result))
        tile = loop.game.board.get_tile_by_id(session.tile_id)
        price = max(1, tile.price)
        if decision.action == "BID" and decision.bid_amount is not None:
            bids += 1
            bid_ratio = decision.bid_amount / price
            bid_amount_ratios.append(bid_ratio)
            max_bid_ratio_by_auction[session.auction_id] = max(
                max_bid_ratio_by_auction.get(session.auction_id, 0.0),
                bid_ratio,
            )
        elif decision.action == "WITHDRAW":
            withdrawals += 1
            withdrawn_price_ratios.append(decision.highest_bid_before / price)
        if result is not None and result.winner_id == player_id:
            wins += 1
    decisions_total = bids + withdrawals
    return AuctionBehaviorProfile(
        player_id=player_id,
        auctions_participated=len(participated_ids),
        bids=bids,
        withdrawals=withdrawals,
        wins=wins,
        bid_rate=bids / decisions_total if decisions_total else 0.0,
        win_rate_per_auction=wins / len(participated_ids) if participated_ids else 0.0,
        mean_bid_to_property_price_ratio=(
            float(np.mean(bid_amount_ratios)) if bid_amount_ratios else None
        ),
        mean_max_bid_to_property_price_ratio=(
            float(np.mean(tuple(max_bid_ratio_by_auction.values())))
            if max_bid_ratio_by_auction else None
        ),
        mean_withdraw_price_to_property_price_ratio=(
            float(np.mean(withdrawn_price_ratios)) if withdrawn_price_ratios else None
        ),
        history=tuple(history),
    )


def _simulate_candidate(
    loop: GameLoop,
    focal_player_id: UUID,
    auction_id: int,
    initial_action: GameAction,
    scenario_seeds: list[int],
    targets: dict[int, int],
    action_limit: int,
    starting_cash: dict[UUID, int],
    starting_net_worth: dict[UUID, int],
    horizons: tuple[int, ...],
    greedy_reserve: int,
    opponent_policies: Mapping[UUID, Policy] | None,
) -> list[dict[str, object]]:
    results = []
    for scenario_seed in scenario_seeds:
        scenario = loop.clone()
        scenario.game.record_events = False
        scenario.game.history_logs.clear()
        _reseed_scenario(scenario, random.Random(scenario_seed))
        policies = _build_policies(
            scenario,
            opponent_policies,
            random.Random(scenario_seed ^ 0xA0761D6478BD642F),
            greedy_reserve,
        )
        action_count = 0
        first = scenario.process_action(initial_action)
        if first["status"] != "SUCCESS":
            raise RuntimeError(f"A ação inicial do leilão foi rejeitada: {first['reason']}")
        action_count += 1

        auction_result = _find_result(scenario, auction_id)
        while auction_result is None:
            if action_count >= action_limit:
                raise RuntimeError("Simulação de leilão excedeu o limite de ações.")
            actor = scenario.current_decision_player_id
            action = policies[actor](scenario)
            response = scenario.process_action(action)
            if response["status"] != "SUCCESS":
                raise RuntimeError(
                    "Política oponente gerou ação inválida no leilão: "
                    f"{action.action_type}: {response['reason']}"
                )
            action_count += 1
            auction_result = _find_result(scenario, auction_id)

        while scenario.turn_phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER:
            if action_count >= action_limit:
                raise RuntimeError("Resolução hipotecária excedeu o limite de ações.")
            actor = scenario.current_decision_player_id
            action = policies[actor](scenario)
            response = scenario.process_action(action)
            if response["status"] != "SUCCESS":
                raise RuntimeError(
                    "Política não resolveu a hipoteca pós-leilão: "
                    f"{action.action_type}: {response['reason']}"
                )
            action_count += 1

        immediate = {
            player_id: {
                "cash_delta": scenario.game.player_states[player_id].money - starting_cash[player_id],
                "net_worth_delta": scenario.game.net_worth(player_id) - starting_net_worth[player_id],
                "owns_tile": scenario.game.property_states[auction_result.tile_id].owner_id == player_id,
                "money_received": (
                    auction_result.winning_bid
                    if player_id == auction_result.seller_id and auction_result.auction_type == "DEBT"
                    else 0
                ),
            }
            for player_id in scenario.game.turn_order
        }
        horizon_states: dict[int, dict[UUID, tuple[int, int, bool]]] = {}
        recorded: set[int] = set()
        while len(recorded) < len(horizons):
            for rounds in horizons:
                if rounds in recorded:
                    continue
                if scenario.game.game_over or scenario.game.turn_counter >= targets[rounds]:
                    horizon_states[rounds] = {
                        player_id: (
                            scenario.game.player_states[player_id].money - starting_cash[player_id],
                            scenario.game.net_worth(player_id) - starting_net_worth[player_id],
                            scenario.game.player_states[player_id].is_bankrupt,
                        )
                        for player_id in scenario.game.turn_order
                    }
                    recorded.add(rounds)
            if len(recorded) == len(horizons):
                break
            if action_count >= action_limit:
                raise RuntimeError("Forecast pós-leilão excedeu o limite de ações.")
            actor = scenario.current_decision_player_id
            action = policies[actor](scenario)
            response = scenario.process_action(action)
            if response["status"] != "SUCCESS":
                raise RuntimeError(
                    "Política oponente gerou ação inválida no forecast do leilão: "
                    f"{action.action_type}: {response['reason']}"
                )
            action_count += 1
        results.append({"result": auction_result, "immediate": immediate, "horizons": horizon_states})
    return results


def _summarize_candidate(
    *,
    actor: UUID,
    tile_id: int,
    candidate_bid: int | None,
    action_type: str,
    simulations_results: list[dict[str, object]],
    horizons: tuple[int, ...],
    turns_per_round: int,
    starting_cash: dict[UUID, int],
    starting_net_worth: dict[UUID, int],
    valuation: AuctionPropertyValuation | None,
    mortgaged: bool,
    mortgage_costs: tuple[int | None, int | None],
    redeemed_valuation: AuctionPropertyValuation | None,
) -> AuctionCandidateOutcome:
    count = len(simulations_results)
    auction_results = [item["result"] for item in simulations_results]
    no_bid_probability = sum(result.winner_id is None for result in auction_results) / count
    actor_win_results = [result for result in auction_results if result.winner_id == actor]
    all_prices = np.asarray([result.winning_bid for result in auction_results], dtype=np.float64)
    conditional_prices = np.asarray([result.winning_bid for result in actor_win_results], dtype=np.float64)
    players = tuple(starting_cash)
    player_effects = []
    for player_id in players:
        immediate_records = [item["immediate"][player_id] for item in simulations_results]
        horizon_effects = []
        for rounds in horizons:
            records = [item["horizons"][rounds][player_id] for item in simulations_results]
            cash_deltas = np.asarray([record[0] for record in records], dtype=np.float64)
            worth_deltas = np.asarray([record[1] for record in records], dtype=np.float64)
            p10, median, p90 = np.quantile(cash_deltas, (0.1, 0.5, 0.9))
            horizon_effects.append(
                AuctionHorizonEffect(
                    horizon_rounds=rounds,
                    requested_turns=rounds * turns_per_round,
                    mean_cash_delta=float(np.mean(cash_deltas)),
                    p10_cash_delta=float(p10),
                    median_cash_delta=float(median),
                    p90_cash_delta=float(p90),
                    bankruptcy_probability=float(np.mean([record[2] for record in records])),
                    mean_book_net_worth_delta=float(np.mean(worth_deltas)),
                )
            )
        wins = sum(result.winner_id == player_id for result in auction_results)
        player_effects.append(
            AuctionPlayerEffect(
                player_id=player_id,
                probability_wins=wins / count,
                probability_owns_auctioned_property_after_result=float(
                    np.mean([record["owns_tile"] for record in immediate_records])
                ),
                mean_cash_delta_at_result=float(
                    np.mean([record["cash_delta"] for record in immediate_records])
                ),
                mean_book_net_worth_delta_at_result=float(
                    np.mean([record["net_worth_delta"] for record in immediate_records])
                ),
                mean_cash_received_from_auction=float(
                    np.mean([record["money_received"] for record in immediate_records])
                ),
                horizons=tuple(horizon_effects),
            )
        )
    winner_ids = {result.winner_id for result in auction_results if result.winner_id is not None}
    winner_probabilities = tuple(
        (player_id, sum(result.winner_id == player_id for result in auction_results) / count)
        for player_id in sorted(winner_ids, key=str)
    )
    interest, redemption = mortgage_costs if mortgaged else (None, None)
    return AuctionCandidateOutcome(
        candidate_bid=candidate_bid,
        action_type=str(action_type),
        simulations=count,
        probability_no_bid=float(no_bid_probability),
        probability_candidate_bidder_wins=float(len(actor_win_results) / count),
        mean_final_price=float(np.mean(all_prices)),
        mean_final_price_if_candidate_wins=(
            float(np.mean(conditional_prices)) if conditional_prices.size else None
        ),
        p10_final_price_if_candidate_wins=(
            float(np.quantile(conditional_prices, 0.1)) if conditional_prices.size else None
        ),
        median_final_price_if_candidate_wins=(
            float(np.median(conditional_prices)) if conditional_prices.size else None
        ),
        p90_final_price_if_candidate_wins=(
            float(np.quantile(conditional_prices, 0.9)) if conditional_prices.size else None
        ),
        winner_probabilities=winner_probabilities,
        property_valuation_at_candidate_bid=valuation,
        property_valuation_at_candidate_bid_if_redeemed=redeemed_valuation,
        mortgage_interest_if_kept=interest,
        mortgage_redemption_cost=redemption,
        player_effects=tuple(player_effects),
    )


def _value_auction_property(
    loop: GameLoop,
    config: GameConfig,
    buyer_id: UUID,
    tile,
    bid: int,
    *,
    horizons: tuple[int, ...],
    discount_rate_per_round: float | None,
    mortgaged_override: bool | None = None,
    additional_cash_cost: int = 0,
) -> AuctionPropertyValuation:
    state = loop.game.property_states[tile.tile_id]
    mortgaged = state.is_mortgaged if mortgaged_override is None else mortgaged_override
    changes = {
        tile.tile_id: PropertyScenarioState(
            owner_id=buyer_id,
            houses=state.houses,
            is_mortgaged=mortgaged,
        )
    }
    portfolio = analyze_portfolio_scenario(
        loop,
        config,
        buyer_id,
        changes,
        immediate_cash_flow=-(bid + additional_cash_cost),
        horizons=horizons,
        discount_rate_per_round=discount_rate_per_round,
    )
    group_tiles = loop.game.board.group_members(tile.group)
    owned_before = sum(
        loop.game.property_states[item.tile_id].owner_id == buyer_id
        for item in group_tiles
    )
    owned_after = owned_before + int(state.owner_id != buyer_id)
    return AuctionPropertyValuation(
        tile_id=tile.tile_id,
        name=tile.name,
        group=tile.group,
        market_price=tile.price,
        candidate_bid=bid,
        owned_group_properties_before=owned_before,
        owned_group_properties_after=owned_after,
        group_size=len(group_tiles),
        completes_group=owned_after == len(group_tiles),
        book_asset_value=tile.price - (tile.mortgage_value if mortgaged else 0),
        portfolio=portfolio,
    )


def _mortgage_costs(loop: GameLoop, tile_id: int) -> tuple[int, int]:
    tile = loop.game.board.get_tile_by_id(tile_id)
    interest = (tile.mortgage_value * MORTGAGE_TRANSFER_INTEREST_PERCENT + 99) // 100
    return interest, loop.unmortgage_cost(tile)


def _find_result(loop: GameLoop, auction_id: int) -> AuctionResultRecord | None:
    return next(
        (item for item in reversed(loop.game.auction_result_history) if item.auction_id == auction_id),
        None,
    )


def _build_policies(
    loop: GameLoop,
    supplied: Mapping[UUID, Policy] | None,
    seed_rng: random.Random,
    greedy_reserve: int,
) -> dict[UUID, Policy]:
    result = {}
    for player_id in loop.game.turn_order:
        template = supplied.get(player_id) if supplied else None
        policy = copy.deepcopy(template) if template is not None else GreedyPolicy(greedy_reserve)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset(seed_rng.getrandbits(32))
        result[player_id] = policy
    return result


def _reseed_scenario(loop: GameLoop, rng: random.Random) -> None:
    for source in (loop.game.dice, *loop.game.decks.values()):
        source_rng = getattr(source, "_rng", None)
        reseed = getattr(source_rng, "seed", None)
        if callable(reseed):
            reseed(rng.getrandbits(64))


def _validate_candidate_bids(values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("candidate_bids deve ser uma coleção de inteiros.") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("Cada lance candidato deve ser um inteiro.")
    if any(value <= 0 for value in result):
        raise ValueError("Os lances candidatos precisam ser positivos.")
    if len(set(result)) != len(result):
        raise ValueError("Os lances candidatos não podem se repetir.")
    return tuple(sorted(result))


def _validate_positive_ints(name: str, values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} deve ser uma coleção de inteiros positivos.") from exc
    if not result or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result):
        raise ValueError(f"{name} deve conter inteiros positivos.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} não pode conter valores duplicados.")
    return tuple(sorted(result))


def _validate_positive_int(name: str, value: int, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} deve ser um inteiro.")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} deve ser {'não negativo' if allow_zero else 'positivo'}.")


def _resolve_seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().getrandbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro ou None.")
    return seed
