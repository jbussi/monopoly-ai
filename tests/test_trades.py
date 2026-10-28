from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import Debt


def trade_payload(**values):
    return {
        "target_player_id": values.pop("target_player_id", None),
        "money_offered": 0,
        "money_requested": 0,
        "properties_offered": [],
        "properties_requested": [],
        "jail_cards_offered": [],
        "jail_cards_requested": [],
        **values,
    }


def send(loop, action_type, payload=None):
    result = loop.process_action(GameAction(action_type, payload or {}))
    assert result["status"] == "SUCCESS", result
    return result


def test_trade_money_offer_can_be_countered_and_accepted(make_game):
    loop = make_game()
    p0, p1 = loop.game.turn_order
    assert loop.available_trade_actions() == (ActionType.PROPOSE_TRADE,)
    payload = trade_payload(target_player_id=p1, money_offered=100)

    send(loop, ActionType.PROPOSE_TRADE, payload)
    assert loop.current_decision_player_id == p1
    assert loop.available_trade_actions() == (
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.COUNTER_TRADE,
    )
    assert loop.game.player_states[p0].money == 1500
    assert loop.game.player_states[p1].money == 1500

    send(loop, ActionType.COUNTER_TRADE, trade_payload(money_offered=50))
    assert loop.current_decision_player_id == p0
    assert loop.game.pending_trade.counteroffers_made == 1
    assert loop.game.pending_trade.current_offer.money_offered == 50

    send(loop, ActionType.ACCEPT_TRADE)
    assert loop.game.player_states[p0].money == 1550
    assert loop.game.player_states[p1].money == 1450
    assert loop.game.pending_trade is None
    assert not loop._trade_window_open
    assert [offer.counteroffer_number for offer in loop.game.trade_history] == [0, 1]


def test_trade_counteroffer_limit_is_three(make_game):
    loop = make_game()
    p0, p1 = loop.game.turn_order
    send(
        loop,
        ActionType.PROPOSE_TRADE,
        trade_payload(target_player_id=p1, money_offered=100),
    )
    for amount in (90, 80, 70):
        send(loop, ActionType.COUNTER_TRADE, trade_payload(money_offered=amount))
    # The default limit is reached; a fourth counteroffer is rejected.
    result = loop.process_action(
        GameAction(ActionType.COUNTER_TRADE, trade_payload(money_offered=60))
    )
    assert result["status"] == "REJECTED"
    assert loop.game.pending_trade.counteroffers_made == 3
    # Either current recipient may still accept the last version.
    send(loop, ActionType.ACCEPT_TRADE)
    assert loop.game.pending_trade is None


def test_trade_rejects_property_from_group_with_buildings(make_game):
    loop = make_game()
    p0, p1 = loop.game.turn_order
    tile = loop.game.board.property_tiles[0]
    group = loop.game.board.group_members(tile.group)
    loop.game.property_states[tile.tile_id].owner_id = p0
    loop.game.property_states[group[1].tile_id].owner_id = p0
    loop.game.property_states[group[0].tile_id].houses = 1

    result = loop.process_action(
        GameAction(
            ActionType.PROPOSE_TRADE,
            trade_payload(target_player_id=p1, properties_offered=[tile.tile_id]),
        )
    )
    assert result["status"] == "REJECTED"
    assert "construções" in result["reason"]
    assert loop.game.pending_trade is None


def test_trade_transfers_mortgaged_property_with_existing_redemption_cost(make_game):
    loop = make_game()
    p0, p1 = loop.game.turn_order
    tile = loop.game.board.property_tiles[0]
    state = loop.game.property_states[tile.tile_id]
    state.owner_id = p0
    state.is_mortgaged = True

    send(
        loop,
        ActionType.PROPOSE_TRADE,
        trade_payload(target_player_id=p1, properties_offered=[tile.tile_id]),
    )
    send(loop, ActionType.ACCEPT_TRADE)

    assert state.owner_id == p1
    assert state.is_mortgaged
    assert loop.unmortgage_cost(tile) == tile.unmortgage_cost


def test_rejected_trade_closes_window_and_next_turn_reopens_it(make_game):
    loop = make_game()
    p0, p1 = loop.game.turn_order
    send(
        loop,
        ActionType.PROPOSE_TRADE,
        trade_payload(target_player_id=p1, money_offered=10),
    )
    send(loop, ActionType.REJECT_TRADE)
    assert not loop._trade_window_open

    loop._next_turn()
    assert loop.game.current_player_id == p1
    assert loop._trade_window_open


def test_pending_trade_requires_explicit_response(make_game):
    loop = make_game()
    p1 = loop.game.turn_order[1]
    send(
        loop,
        ActionType.PROPOSE_TRADE,
        trade_payload(target_player_id=p1, money_offered=10),
    )
    result = loop.process_action(GameAction(ActionType.ROLL_DICE))
    assert result["status"] == "REJECTED"
    assert {a.action_type for a in loop.trade_response_actions()} == {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.COUNTER_TRADE,
    }


def test_debt_trade_must_be_answered_and_can_settle_debt(make_game):
    loop = make_game()
    debtor, creditor = loop.game.turn_order
    loop.game.player_states[debtor].money = 0
    loop.game.debts.append(Debt(debtor, None, 80))
    loop.turn_phase = GameLoop.RESOLVE_DEBT
    loop._settle()

    assert loop.current_decision_player_id == debtor
    assert loop.available_trade_actions() == (ActionType.PROPOSE_TRADE,)
    send(
        loop,
        ActionType.PROPOSE_TRADE,
        trade_payload(target_player_id=creditor, money_requested=100),
    )
    assert loop.current_decision_player_id == creditor
    assert loop.available_trade_actions() == (
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.COUNTER_TRADE,
    )

    result = loop.process_action(GameAction(ActionType.MORTGAGE, {"tile_id": 1}))
    assert result["status"] == "REJECTED"
    send(loop, ActionType.ACCEPT_TRADE)
    assert loop.game.pending_trade is None
    assert loop.game.debts == []
    assert loop.game.player_states[debtor].money == 20
