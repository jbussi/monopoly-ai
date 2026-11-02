"""Fixed-shape numeric encoding for action-conditioned decision contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from gymnasium import spaces

from app.domain.analytics.decision_context import DecisionContext
from app.domain.analytics.economics.auction_analysis import (
    AuctionBehaviorProfile,
    AuctionCandidateOutcome,
)
from app.domain.analytics.economics.jail_opportunity import JailChoiceForecast
from app.domain.analytics.economics.trade_analysis import (
    TradeBehaviorProfile,
    TradeOutcomeScenario,
)
from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.envs.action_space_v2 import ACTION_TYPE_TO_INDEX, ACTION_TYPES
from app.envs.observation_v2 import PHASES


@dataclass(frozen=True, slots=True)
class _Alternative:
    action: GameAction
    role: int
    outcome: int = 0
    source: Any = None


@dataclass(frozen=True, slots=True)
class EncodedDecisionContext:
    """Full analytical result paired with its fixed-shape model features."""

    context: DecisionContext
    features: dict[str, np.ndarray]


class DecisionFeaturesEncoder:
    """Encode decision context as bounded-size arrays for model inputs.

    Alternative rows are explicitly masked. Player rows are relative to the
    acting player, property rows follow board property-tile order, and horizon
    rows follow ``horizons``. The original context remains available for full
    histories and details that are summarized in these arrays.
    """

    ROLE_CANDIDATE = 0
    ROLE_BASELINE = 1
    ROLE_SCENARIO = 2
    OUTCOME_NONE = 0
    OUTCOME_ACCEPT = 1
    OUTCOME_REJECT = 2
    OUTCOME_COUNTER = 3

    def __init__(
        self,
        loop: GameLoop,
        *,
        max_players: int = 12,
        max_alternatives: int = 128,
        horizons: Iterable[int] = (1, 3, 5),
    ) -> None:
        if not 2 <= max_players <= 12 or len(loop.game.turn_order) > max_players:
            raise ValueError("max_players deve cobrir a partida e estar entre 2 e 12.")
        if isinstance(max_alternatives, bool) or not isinstance(max_alternatives, int) or max_alternatives < 1:
            raise ValueError("max_alternatives deve ser um inteiro positivo.")
        horizon_values = tuple(horizons)
        if not horizon_values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in horizon_values
        ):
            raise ValueError("horizons deve conter inteiros positivos.")
        if len(set(horizon_values)) != len(horizon_values):
            raise ValueError("horizons não pode conter valores duplicados.")

        self.max_players = max_players
        self.max_alternatives = max_alternatives
        self.horizons = tuple(sorted(horizon_values))
        self.turn_order = tuple(loop.game.turn_order)
        self.property_tiles = tuple(loop.game.board.property_tiles)
        self.property_index = {tile.tile_id: i for i, tile in enumerate(self.property_tiles)}
        self.jail_card_ids = self._jail_card_ids(loop)
        self.jail_card_index = {card_id: i for i, card_id in enumerate(self.jail_card_ids)}
        self.board_size = loop.game.board.size
        self.money_scale = float(max(1, loop.rules.starting_money, loop.rules.go_salary))

        self.action_scalar_size = 8
        self.action_size = (
            len(ACTION_TYPES)
            + self.action_scalar_size
            + 2 * len(self.property_tiles)
            + 2 * len(self.jail_card_ids)
        )
        shapes = {
            "phase": (len(PHASES),),
            "horizons": (len(self.horizons),),
            "horizon_mask": (max_alternatives, len(self.horizons)),
            "simulations": (1,),
            "alternative_mask": (max_alternatives,),
            "alternative_role": (max_alternatives, 3),
            "alternative_outcome": (max_alternatives, 4),
            "scenario_metrics": (max_alternatives, 9),
            "actions": (max_alternatives, self.action_size),
            "player_effects": (max_alternatives, max_players, 8),
            "cash_forecasts": (max_alternatives, max_players, len(self.horizons), 11),
            "portfolio_values": (max_alternatives, max_players, len(self.horizons), 10),
            "redeemed_portfolio_values": (max_alternatives, max_players, len(self.horizons), 10),
            "property_opportunities": (max_alternatives, len(self.property_tiles), len(self.horizons), 11),
            "auction_property_values": (max_alternatives, len(self.property_tiles), 8),
            "redeemed_property_values": (max_alternatives, len(self.property_tiles), 8),
            "opponent_behavior": (max_alternatives, max_players, 18),
        }
        self.observation_space = spaces.Dict({
            name: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
            for name, shape in shapes.items()
        })

    def encode(self, context: DecisionContext) -> dict[str, np.ndarray]:
        """Encode a context; raise instead of silently truncating alternatives."""
        if not isinstance(context, DecisionContext):
            raise TypeError("context deve ser DecisionContext.")
        if context.player_id not in self.turn_order:
            raise ValueError("O contexto pertence a uma partida diferente do encoder.")
        alternatives = self._alternatives(context)
        if len(alternatives) > self.max_alternatives:
            raise ValueError(
                f"O contexto tem {len(alternatives)} alternativas; limite configurado: "
                f"{self.max_alternatives}."
            )

        output = {key: np.zeros(space.shape, dtype=np.float32)
                  for key, space in self.observation_space.spaces.items()}
        output["phase"][PHASES.index(context.phase)] = 1.0
        output["horizons"][:] = np.asarray(self.horizons, dtype=np.float32)
        output["simulations"][0] = context.simulations
        for index, alternative in enumerate(alternatives):
            output["alternative_mask"][index] = 1.0
            output["alternative_role"][index, alternative.role] = 1.0
            if 0 <= alternative.outcome < 4:
                output["alternative_outcome"][index, alternative.outcome] = 1.0
            self._encode_action(output["actions"][index], alternative.action, context.player_id)
            self._encode_source(output, index, alternative, context)
        return output

    def _alternatives(self, context: DecisionContext) -> tuple[_Alternative, ...]:
        if context.debt_analysis is not None:
            return tuple(_Alternative(item.action, self.ROLE_SCENARIO, source=item)
                         for item in context.debt_analysis.scenarios)
        if context.auction_analysis is not None:
            result = []
            for item in context.auction_analysis.outcomes:
                if item.candidate_bid is None:
                    action = GameAction(ActionType.AUCTION_WITHDRAW)
                else:
                    action = GameAction(ActionType.AUCTION_BID, {"amount": item.candidate_bid})
                result.append(_Alternative(action, self.ROLE_SCENARIO, source=item))
            return tuple(result)
        if context.jail_analysis is not None:
            return tuple(
                _Alternative(GameAction(item.action_type), self.ROLE_SCENARIO, source=item)
                for item in context.jail_analysis.choices
            )
        if context.trade_analysis is not None:
            trade = context.trade_analysis
            offer_payload = {
                "target_player_id": (
                    trade.recipient_id
                    if context.candidate_action.action_type == ActionType.PROPOSE_TRADE
                    else trade.proposer_id
                ),
                "money_offered": trade.money_offered,
                "money_requested": trade.money_requested,
                "properties_offered": trade.properties_offered,
                "properties_requested": trade.properties_requested,
                "jail_cards_offered": trade.jail_cards_offered,
                "jail_cards_requested": trade.jail_cards_requested,
                "counteroffer_number": trade.counteroffer_number,
            }
            encoded_action = GameAction(context.candidate_action.action_type, offer_payload)
            return tuple(
                _Alternative(
                    encoded_action,
                    self.ROLE_SCENARIO,
                    outcome={"ACCEPT": 1, "REJECT": 2, "COUNTER": 3}.get(item.outcome, 0),
                    source=item,
                )
                for item in context.trade_analysis.scenarios
            )
        alternatives = [_Alternative(context.candidate_action, self.ROLE_CANDIDATE, source=context.candidate)]
        if context.baseline_action is not None:
            alternatives.append(_Alternative(context.baseline_action, self.ROLE_BASELINE, source=context.baseline))
        return tuple(alternatives)

    def _encode_action(self, row: np.ndarray, action: GameAction, observer_id) -> None:
        action_index = ACTION_TYPE_TO_INDEX.get(action.action_type)
        if action_index is None:
            raise ValueError(f"Tipo de ação desconhecido: {action.action_type}.")
        row[action_index] = 1.0
        offset = len(ACTION_TYPES)
        payload = action.payload
        tile_id = payload.get("tile_id", self.board_size)
        amount = payload.get("amount", 0)
        target_id = payload.get("target_player_id")
        target_slot = self._player_slot(target_id, observer_id) if target_id in self.turn_order else self.max_players
        row[offset:offset + self.action_scalar_size] = (
            tile_id / max(1, self.board_size),
            amount / self.money_scale,
            target_slot / max(1, self.max_players),
            payload.get("money_offered", 0) / self.money_scale,
            payload.get("money_requested", 0) / self.money_scale,
            len(payload.get("properties_offered", ())) / max(1, len(self.property_tiles)),
            len(payload.get("properties_requested", ())) / max(1, len(self.property_tiles)),
            payload.get("counteroffer_number", 0),
        )
        offset += self.action_scalar_size
        for side, key in enumerate(("properties_offered", "properties_requested")):
            start = offset + side * len(self.property_tiles)
            for property_id in payload.get(key, ()):
                if property_id not in self.property_index:
                    raise ValueError(f"Imóvel desconhecido na ação: {property_id}.")
                row[start + self.property_index[property_id]] = 1.0
        offset += 2 * len(self.property_tiles)
        for side, key in enumerate(("jail_cards_offered", "jail_cards_requested")):
            start = offset + side * len(self.jail_card_ids)
            for card_id in payload.get(key, ()):
                if card_id not in self.jail_card_index:
                    raise ValueError(f"Carta de prisão desconhecida na ação: {card_id}.")
                row[start + self.jail_card_index[card_id]] = 1.0

    def _encode_source(self, output, alternative_index: int, alternative: _Alternative, context: DecisionContext) -> None:
        source = alternative.source
        if source is None:
            return
        observer_id = context.player_id
        if hasattr(source, "cash_forecast"):
            self._forecast(
                output, alternative_index, source.cash_forecast.player_id,
                source.cash_forecast, observer_id,
            )
        if hasattr(source, "portfolio") and source.portfolio is not None:
            self._portfolio(output, alternative_index, source.portfolio.player_id, source.portfolio, observer_id)
        if hasattr(source, "participants"):
            for participant in source.participants:
                self._participant(output, alternative_index, participant, observer_id)
        if hasattr(source, "impacts"):
            for impact in source.impacts:
                slot = self._player_slot(impact.player_id, observer_id)
                self._impact(output, alternative_index, slot, impact)
                self._forecast(output, alternative_index, impact.player_id, impact.cash_forecast, observer_id)
        if hasattr(source, "player_effects"):
            self._auction_effects(output, alternative_index, source, observer_id)
            self._auction_scenario_metrics(output, alternative_index, source)
            if context.auction_analysis is not None:
                for profile in context.auction_analysis.opponent_behavior:
                    self._auction_behavior(output, alternative_index, profile, observer_id)
        if isinstance(source, JailChoiceForecast):
            self._jail_choice(output, alternative_index, source, observer_id)
        if isinstance(source, TradeOutcomeScenario):
            self._trade_outcome(
                output, alternative_index, source, observer_id, context.trade_analysis
            )

    def _participant(self, output, alternative_index: int, participant, observer_id) -> None:
        slot = self._player_slot(participant.player_id, observer_id)
        # Participant scenarios include a portfolio and cash forecast.
        self._impact(output, alternative_index, slot, participant)
        self._forecast(output, alternative_index, participant.player_id, participant.cash_forecast, observer_id)
        self._portfolio(output, alternative_index, participant.player_id, participant.portfolio, observer_id)

    def _impact(self, output, alternative_index: int, slot: int, impact) -> None:
        row = output["player_effects"][alternative_index, slot]
        cash_before = getattr(impact, "cash_before", 0)
        cash_after = getattr(impact, "cash_after_immediate_exchange", getattr(impact, "cash_after", cash_before))
        cash_delta = getattr(impact, "cash_delta", cash_after - cash_before)
        nw_before = getattr(impact, "book_net_worth_before", getattr(impact, "net_worth_before", 0))
        nw_delta = getattr(impact, "book_net_worth_delta", getattr(impact, "net_worth_delta", 0))
        row[:5] = (cash_before / self.money_scale, cash_delta / self.money_scale,
                   nw_before / self.money_scale, nw_delta / self.money_scale,
                   float(getattr(impact, "is_bankrupt_after_action", False)))

    def _forecast(self, output, alternative_index: int, player_id, forecast, observer_id) -> None:
        slot = self._player_slot(player_id, observer_id)
        self._write_forecast(output, alternative_index, slot, forecast)

    def _write_forecast(self, output, alternative_index: int, slot: int, forecast) -> None:
        for horizon in forecast.horizons:
            index = self._horizon_index(horizon.horizon_rounds)
            output["horizon_mask"][alternative_index, index] = 1.0
            output["cash_forecasts"][alternative_index, slot, index] = (
                horizon.mean_net_cash_flow / self.money_scale,
                horizon.std_net_cash_flow / self.money_scale,
                horizon.p10_net_cash_flow / self.money_scale,
                horizon.median_net_cash_flow / self.money_scale,
                horizon.p90_net_cash_flow / self.money_scale,
                horizon.mean_ending_cash / self.money_scale,
                horizon.p10_ending_cash / self.money_scale,
                horizon.median_ending_cash / self.money_scale,
                horizon.p90_ending_cash / self.money_scale,
                horizon.bankruptcy_probability,
                horizon.game_over_probability,
            )

    def _portfolio(self, output, alternative_index: int, player_id, portfolio, observer_id,
                   *, redeemed: bool = False) -> None:
        slot = self._player_slot(player_id, observer_id)
        key = "redeemed_portfolio_values" if redeemed else "portfolio_values"
        for horizon in portfolio.horizons:
            index = self._horizon_index(horizon.horizon_rounds)
            output["horizon_mask"][alternative_index, index] = 1.0
            output[key][alternative_index, slot, index] = (
                portfolio.current_rent_income_per_round / self.money_scale,
                portfolio.projected_rent_income_per_round / self.money_scale,
                portfolio.incremental_rent_income_per_round / self.money_scale,
                horizon.incremental_operating_cash_flow / self.money_scale,
                horizon.immediate_cash_flow / self.money_scale,
                horizon.expected_net_cash_flow / self.money_scale,
                horizon.roi_on_cash_outlay or 0.0,
                float(horizon.roi_on_cash_outlay is not None),
                (horizon.net_present_value or 0.0) / self.money_scale,
                float(horizon.net_present_value is not None),
            )

    def _auction_effects(self, output, alternative_index: int, outcome: AuctionCandidateOutcome, observer_id) -> None:
        for effect in outcome.player_effects:
            slot = self._player_slot(effect.player_id, observer_id)
            row = output["player_effects"][alternative_index, slot]
            row[1] = effect.mean_cash_delta_at_result / self.money_scale
            row[3] = effect.mean_book_net_worth_delta_at_result / self.money_scale
            row[5:8] = (
                effect.probability_wins,
                effect.probability_owns_auctioned_property_after_result,
                effect.mean_cash_received_from_auction / self.money_scale,
            )
            for horizon in effect.horizons:
                index = self._horizon_index(horizon.horizon_rounds)
                output["horizon_mask"][alternative_index, index] = 1.0
                output["cash_forecasts"][alternative_index, slot, index, 0:5] = (
                    horizon.mean_cash_delta / self.money_scale,
                    0.0,
                    horizon.p10_cash_delta / self.money_scale,
                    horizon.median_cash_delta / self.money_scale,
                    horizon.p90_cash_delta / self.money_scale,
                )
                output["cash_forecasts"][alternative_index, slot, index, 9] = horizon.bankruptcy_probability
        valuation = outcome.property_valuation_at_candidate_bid
        if valuation is not None:
            self._auction_valuation(output, alternative_index, valuation, observer_id)
        redeemed_valuation = outcome.property_valuation_at_candidate_bid_if_redeemed
        if redeemed_valuation is not None:
            self._auction_valuation(
                output, alternative_index, redeemed_valuation, observer_id, redeemed=True
            )

    def _auction_scenario_metrics(self, output, alternative_index: int, outcome) -> None:
        output["scenario_metrics"][alternative_index] = (
            outcome.probability_no_bid,
            outcome.probability_candidate_bidder_wins,
            outcome.mean_final_price / self.money_scale,
            (outcome.mean_final_price_if_candidate_wins or 0.0) / self.money_scale,
            (outcome.p10_final_price_if_candidate_wins or 0.0) / self.money_scale,
            (outcome.median_final_price_if_candidate_wins or 0.0) / self.money_scale,
            (outcome.p90_final_price_if_candidate_wins or 0.0) / self.money_scale,
            (outcome.mortgage_interest_if_kept or 0) / self.money_scale,
            (outcome.mortgage_redemption_cost or 0) / self.money_scale,
        )

    def _auction_valuation(self, output, alternative_index: int, valuation, observer_id,
                           *, redeemed: bool = False) -> None:
        prop_index = self.property_index[valuation.tile_id]
        values = (
            valuation.market_price / self.money_scale,
            valuation.candidate_bid / self.money_scale,
            valuation.owned_group_properties_before,
            valuation.owned_group_properties_after,
            valuation.group_size,
            float(valuation.completes_group),
            valuation.book_asset_value / self.money_scale,
            1.0,
        )
        key = "redeemed_property_values" if redeemed else "auction_property_values"
        output[key][alternative_index, prop_index] = values
        self._portfolio(
            output, alternative_index, valuation.portfolio.player_id,
            valuation.portfolio, observer_id, redeemed=redeemed,
        )

    def _jail_choice(self, output, alternative_index: int, choice: JailChoiceForecast, observer_id) -> None:
        slot = self._player_slot(observer_id, observer_id)
        for horizon in choice.horizons:
            index = self._horizon_index(horizon.horizon_rounds)
            output["horizon_mask"][alternative_index, index] = 1.0
            output["cash_forecasts"][alternative_index, slot, index, 0:5] = (
                horizon.mean_cash_delta / self.money_scale,
                horizon.std_cash_delta / self.money_scale,
                horizon.p10_cash_delta / self.money_scale,
                horizon.median_cash_delta / self.money_scale,
                horizon.p90_cash_delta / self.money_scale,
            )
            output["cash_forecasts"][alternative_index, slot, index, 9] = horizon.bankruptcy_probability
            for opportunity in horizon.property_opportunities:
                prop_index = self.property_index[opportunity.tile_id]
                row = output["property_opportunities"][alternative_index, prop_index, index]
                row[:3] = (opportunity.probability_offered_to_player,
                           opportunity.probability_bought_by_greedy_policy,
                           opportunity.probability_bought_if_offered)
                valuation = opportunity.acquisition_valuation
                row[3:8] = (
                    valuation.acquisition_price / self.money_scale,
                    valuation.owned_group_properties_before,
                    valuation.owned_group_properties_after,
                    valuation.group_size,
                    float(valuation.completes_group),
                )
                weighted_values = dict(opportunity.greedy_weighted_positive_net_value)
                row[8] = weighted_values.get(horizon.horizon_rounds, 0.0) / self.money_scale
                row[9] = valuation.projected_group_rent_income_per_round / self.money_scale
                row[10] = 1.0
                self._portfolio(output, alternative_index, observer_id,
                                valuation.portfolio_impact, observer_id)

    def _trade_outcome(self, output, alternative_index: int, outcome: TradeOutcomeScenario,
                       observer_id, analysis) -> None:
        if analysis is not None:
            for profile in (analysis.proposer_behavior, analysis.recipient_behavior):
                self._trade_behavior(output, alternative_index, profile, observer_id)

    def _trade_behavior(self, output, alternative_index: int, profile: TradeBehaviorProfile, observer_id) -> None:
        slot = self._player_slot(profile.player_id, observer_id)
        output["opponent_behavior"][alternative_index, slot, :9] = (
            profile.offers_received, profile.accepted, profile.rejected, profile.countered,
            profile.offers_made, profile.mean_counteroffer_number_when_responding or 0.0,
            profile.accepted / max(1, profile.offers_received),
            profile.rejected / max(1, profile.offers_received),
            profile.countered / max(1, profile.offers_received),
        )

    def _auction_behavior(self, output, alternative_index: int, profile: AuctionBehaviorProfile, observer_id) -> None:
        slot = self._player_slot(profile.player_id, observer_id)
        output["opponent_behavior"][alternative_index, slot, 9:18] = (
            profile.auctions_participated, profile.bids, profile.withdrawals, profile.wins,
            profile.bid_rate, profile.win_rate_per_auction,
            profile.mean_bid_to_property_price_ratio or 0.0,
            profile.mean_max_bid_to_property_price_ratio or 0.0,
            profile.mean_withdraw_price_to_property_price_ratio or 0.0,
        )

    def _player_slot(self, player_id, observer_id) -> int:
        relative = [self.turn_order[(self.turn_order.index(observer_id) + i) % len(self.turn_order)]
                    for i in range(len(self.turn_order))]
        return relative.index(player_id)

    def _horizon_index(self, value: int) -> int:
        try:
            return self.horizons.index(value)
        except ValueError as exc:
            raise ValueError(f"Horizonte {value} não está configurado no encoder.") from exc

    @staticmethod
    def _jail_card_ids(loop: GameLoop) -> tuple[str, ...]:
        card_ids = []
        for deck_name, deck in loop.game.decks.items():
            for card in (*deck._cards, *deck._discard_pile):
                if card.action == "get_out_of_jail":
                    card_ids.append(f"{deck_name}:{card.id}")
        for state in loop.game.player_states.values():
            card_ids.extend(f"{deck_name}:{card.id}" for deck_name, card in state.jail_cards)
        return tuple(sorted(set(card_ids)))
