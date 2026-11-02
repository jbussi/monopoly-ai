"""Monte Carlo forecasts for the jailed player's next own turns.

The forecast replays the real game loop for the focal player while holding
opponents' positions and decisions fixed. It is a decision feature, not a
recommendation or a full-match simulation.
"""

from __future__ import annotations

import random
from uuid import UUID

import numpy as np

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop


JAIL_FORECAST_HORIZONS = (1, 3, 5)
JAIL_FORECAST_PLANS = ("try_doubles", "pay_fine", "use_jail_card")
DEFAULT_JAIL_FORECAST_SAMPLES = 64


def empty_jail_forecast(loop: GameLoop) -> dict[str, np.ndarray]:
    """Return correctly shaped neutral forecasts when jail analysis is inactive."""
    plan_count = len(JAIL_FORECAST_PLANS)
    horizon_count = len(JAIL_FORECAST_HORIZONS)
    board_size = loop.game.board.size
    property_count = len(loop.game.board.property_tiles)
    return {
        "available": np.zeros(plan_count, dtype=np.float32),
        "expected_cash_delta": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "cash_delta_std": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "expected_debt": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "probability_in_jail": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "probability_of_debt": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "expected_jail_cards_remaining": np.zeros(
            (plan_count, horizon_count), dtype=np.float32
        ),
        "expected_rent_charged": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "expected_tax_charged": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "expected_go_salary": np.zeros((plan_count, horizon_count), dtype=np.float32),
        "property_opportunities": np.zeros(
            (plan_count, horizon_count, property_count), dtype=np.float32
        ),
        "position_probability": np.zeros(
            (plan_count, horizon_count, board_size), dtype=np.float32
        ),
    }


def forecast_jail_decisions(
    loop: GameLoop,
    player_id: UUID,
    *,
    samples: int = DEFAULT_JAIL_FORECAST_SAMPLES,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Forecast jail alternatives over 1, 3, and 5 of the player's own turns.

    Scenarios are: keep trying for doubles, pay the fine now, or use a jail
    card now. After the first turn, a jailed player keeps trying for doubles.
    Unowned properties are declined and auctions receive no bids. Opponents do
    not take turns in this first version; their current assets remain in place.
    Paths stop at the first unresolved debt rather than inventing a liquidation
    policy. Future card order is randomized from the currently public pile
    composition, so forecasts do not use the engine's hidden deck order.
    """
    if samples < 2:
        raise ValueError("samples deve ser pelo menos 2.")
    result = empty_jail_forecast(loop)
    state = loop.game.player_states[player_id]
    if (
        not state.is_in_jail
        or loop.turn_phase != GameLoop.BEFORE_ROLL
        or loop.current_decision_player_id != player_id
        or loop.game.debts
        or loop.game.pending_trade is not None
        or loop.game.game_over
    ):
        return result

    result["available"][:] = (1.0, float(state.money >= loop.rules.jail_fine), float(bool(state.jail_cards)))
    if not result["available"][1] and not result["available"][2]:
        plans = ("try_doubles",)
    else:
        plans = JAIL_FORECAST_PLANS

    property_tiles = tuple(loop.game.board.property_tiles)
    property_index = {tile.tile_id: index for index, tile in enumerate(property_tiles)}
    horizon_index = {horizon: index for index, horizon in enumerate(JAIL_FORECAST_HORIZONS)}

    for plan_index, plan in enumerate(JAIL_FORECAST_PLANS):
        if not result["available"][plan_index]:
            continue
        if plan not in plans:
            continue
        samples_by_horizon = [
            {key: [] for key in (
                "cash_delta", "debt", "in_jail", "debt_occurred", "jail_cards",
                "rent_charged",
                "tax_charged", "go_salary", "property_opportunities", "position",
            )}
            for _ in JAIL_FORECAST_HORIZONS
        ]
        for sample_index in range(samples):
            sample_seed = int(seed) + sample_index * 104729
            sim = loop.clone()
            _prepare_simulation(sim, sample_seed)
            initial_money = sim.game.player_states[player_id].money
            cumulative = {
                "rent_charged": 0.0,
                "tax_charged": 0.0,
                "go_salary": 0.0,
                "property_opportunities": np.zeros(len(property_tiles), dtype=np.float64),
            }
            stopped = False

            for own_turn in range(1, max(JAIL_FORECAST_HORIZONS) + 1):
                if not stopped:
                    if own_turn == 1 and plan == "pay_fine":
                        stopped = not _apply(sim, player_id, ActionType.PAY_JAIL_FINE)
                    elif own_turn == 1 and plan == "use_jail_card":
                        stopped = not _apply(sim, player_id, ActionType.USE_JAIL_CARD)

                    if not stopped:
                        stopped = not _play_focal_turn(
                            sim, player_id, property_index, cumulative
                        )

                if own_turn in horizon_index:
                    h = horizon_index[own_turn]
                    player_state = sim.game.player_states[player_id]
                    sample_values = samples_by_horizon[h]
                    sample_values["cash_delta"].append(player_state.money - initial_money)
                    sample_values["debt"].append(sim.game.amount_owed(player_id))
                    sample_values["in_jail"].append(float(player_state.is_in_jail))
                    sample_values["debt_occurred"].append(float(
                        sim.game.amount_owed(player_id) > 0
                        or player_state.is_bankrupt
                    ))
                    sample_values["jail_cards"].append(float(len(player_state.jail_cards)))
                    sample_values["rent_charged"].append(cumulative["rent_charged"])
                    sample_values["tax_charged"].append(cumulative["tax_charged"])
                    sample_values["go_salary"].append(cumulative["go_salary"])
                    sample_values["property_opportunities"].append(
                        cumulative["property_opportunities"].copy()
                    )
                    position = sim.game.position_of(player_id)
                    sample_values["position"].append(position)

                if not stopped and own_turn < max(JAIL_FORECAST_HORIZONS):
                    stopped = not _start_next_focal_turn(sim, player_id)

        for horizon, h in horizon_index.items():
            values = samples_by_horizon[h]
            cash = np.asarray(values["cash_delta"], dtype=np.float64)
            result["expected_cash_delta"][plan_index, h] = cash.mean()
            result["cash_delta_std"][plan_index, h] = cash.std(ddof=1)
            result["expected_debt"][plan_index, h] = np.mean(values["debt"])
            result["probability_in_jail"][plan_index, h] = np.mean(values["in_jail"])
            result["probability_of_debt"][plan_index, h] = np.mean(values["debt_occurred"])
            result["expected_jail_cards_remaining"][plan_index, h] = np.mean(values["jail_cards"])
            result["expected_rent_charged"][plan_index, h] = np.mean(values["rent_charged"])
            result["expected_tax_charged"][plan_index, h] = np.mean(values["tax_charged"])
            result["expected_go_salary"][plan_index, h] = np.mean(values["go_salary"])
            result["property_opportunities"][plan_index, h] = np.mean(
                np.stack(values["property_opportunities"]), axis=0
            )
            for position in values["position"]:
                result["position_probability"][plan_index, h, position] += 1.0 / samples

    return result


def _prepare_simulation(loop: GameLoop, seed: int) -> None:
    """Seed cloned randomness and remove any dependence on hidden pile order."""
    loop.game.record_events = True
    loop.game.history_logs = []
    loop.game.dice._rng = random.Random(seed ^ 0xD1CE)
    for deck_index, deck_name in enumerate(sorted(loop.game.decks)):
        deck = loop.game.decks[deck_name]
        deck._cards.sort(key=lambda card: card.id)
        deck._discard_pile.sort(key=lambda card: card.id)
        for state in loop.game.player_states.values():
            state.jail_cards.sort(key=lambda item: (item[0], item[1].id))
        deck._rng = random.Random(seed + 7919 * (deck_index + 1))
        deck.shuffle()


def _apply(loop: GameLoop, player_id: UUID, action_type: str) -> bool:
    if loop.current_decision_player_id != player_id:
        return False
    return loop.process_action(GameAction(action_type))["status"] == "SUCCESS"


def _play_focal_turn(
    loop: GameLoop,
    player_id: UUID,
    property_index: dict[int, int],
    cumulative: dict[str, np.ndarray | float],
) -> bool:
    """Replay movement and deterministic landing effects until the turn ends."""
    for _ in range(64):
        if loop.game.game_over or loop.current_decision_player_id != player_id:
            return False
        phase = loop.turn_phase
        if phase == GameLoop.BEFORE_ROLL:
            if not _apply(loop, player_id, ActionType.ROLL_DICE):
                return False
        elif phase == GameLoop.AWAITING_PROPERTY_DECISION:
            tile_id = loop.pending_tile_id
            if tile_id in property_index:
                cumulative["property_opportunities"][property_index[tile_id]] += 1.0
            if not _apply(loop, player_id, ActionType.DECLINE_PROPERTY):
                return False
        elif phase == GameLoop.AUCTION:
            bidder = loop.current_decision_player_id
            if bidder == loop.auction_highest_bidder:
                return False
            if loop.process_action(GameAction(ActionType.AUCTION_WITHDRAW))["status"] != "SUCCESS":
                return False
        elif phase == GameLoop.AFTER_ROLL:
            _accumulate_events(loop, player_id, cumulative)
            return not bool(loop.game.amount_owed(player_id))
        elif phase in (GameLoop.RESOLVE_DEBT, GameLoop.RESOLVE_MORTGAGE_TRANSFER, GameLoop.GAME_OVER):
            _accumulate_events(loop, player_id, cumulative)
            return False
        else:
            return False

        if loop.game.debts or loop.turn_phase in (
            GameLoop.RESOLVE_DEBT,
            GameLoop.RESOLVE_MORTGAGE_TRANSFER,
            GameLoop.GAME_OVER,
        ):
            _accumulate_events(loop, player_id, cumulative)
            return False

    return False


def _accumulate_events(
    loop: GameLoop,
    player_id: UUID,
    cumulative: dict[str, np.ndarray | float],
) -> None:
    for event in loop.game.history_logs:
        if event.get("_forecast_counted"):
            continue
        data = event["data"]
        if data.get("player_id") != player_id:
            continue
        if event["event"] == "RENT_PAYMENT":
            cumulative["rent_charged"] += data.get("amount", 0)
        elif event["event"] == "TAX_PAYMENT":
            cumulative["tax_charged"] += data.get("amount", 0)
        elif event["event"] == "PASSED_GO":
            cumulative["go_salary"] += data.get("salary", 0)
        event["_forecast_counted"] = True


def _start_next_focal_turn(loop: GameLoop, player_id: UUID) -> bool:
    """Advance the shadow forecast while leaving opponents' state fixed."""
    active_count = max(1, len(loop.game.active_player_ids))
    max_turns = loop.rules.max_turns
    if max_turns is not None and loop.game.turn_counter + active_count - 1 >= max_turns:
        loop.game.game_over = True
        loop.game.truncated = True
        loop.turn_phase = GameLoop.GAME_OVER
        return False
    loop.game.current_player_id = player_id
    loop.turn_phase = GameLoop.BEFORE_ROLL
    loop.double_count = 0
    loop._pending_extra_roll = False
    loop._pending_tile_id = None
    loop.game.turn_counter += active_count
    return True
