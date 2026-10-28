from app.domain.game.actions import ActionType as A, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import Debt


def act(loop, kind, **payload):
    result = loop.process_action(GameAction(kind, payload))
    assert result["status"] == "SUCCESS", result
    return result


def propose_mortgaged_property(loop, tile_id):
    proposer, recipient = loop.game.turn_order
    loop.game.property_states[tile_id].owner_id = proposer
    loop.game.property_states[tile_id].is_mortgaged = True
    act(
        loop,
        A.PROPOSE_TRADE,
        target_player_id=recipient,
        properties_offered=[tile_id],
    )
    act(loop, A.ACCEPT_TRADE)
    return proposer, recipient


def test_new_owner_can_keep_transferred_mortgage(make_game):
    loop = make_game()
    tile = loop.game.board.property_tiles[0]
    _, recipient = propose_mortgaged_property(loop, tile.tile_id)
    interest = (tile.mortgage_value * 10 + 99) // 100

    assert loop.turn_phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER
    assert loop.current_decision_player_id == recipient
    assert {action.action_type for action in loop.legal_actions()} == {
        A.KEEP_MORTGAGE,
        A.REDEEM_MORTGAGE,
    }

    act(loop, A.KEEP_MORTGAGE, tile_id=tile.tile_id)

    assert loop.game.property_states[tile.tile_id].owner_id == recipient
    assert loop.game.property_states[tile.tile_id].is_mortgaged
    assert loop.game.player_states[recipient].money == 1500 - interest
    assert loop.turn_phase == GameLoop.BEFORE_ROLL


def test_new_owner_can_redeem_transferred_mortgage_immediately(make_game):
    loop = make_game()
    tile = loop.game.board.property_tiles[0]
    _, recipient = propose_mortgaged_property(loop, tile.tile_id)
    redemption_cost = loop.unmortgage_cost(tile)

    act(loop, A.REDEEM_MORTGAGE, tile_id=tile.tile_id)

    assert loop.game.property_states[tile.tile_id].owner_id == recipient
    assert not loop.game.property_states[tile.tile_id].is_mortgaged
    assert loop.game.player_states[recipient].money == 1500 - redemption_cost
    assert loop.turn_phase == GameLoop.BEFORE_ROLL


def test_unpaid_transfer_interest_enters_debt_resolution(make_game):
    loop = make_game()
    tile = loop.game.board.property_tiles[0]
    _, recipient = propose_mortgaged_property(loop, tile.tile_id)
    interest = (tile.mortgage_value * 10 + 99) // 100
    loop.game.player_states[recipient].money = interest - 1

    legal = {action.action_type for action in loop.legal_actions()}
    assert legal == {A.KEEP_MORTGAGE}

    act(loop, A.KEEP_MORTGAGE, tile_id=tile.tile_id)

    assert loop.game.property_states[tile.tile_id].is_mortgaged
    assert loop.game.amount_owed(recipient) == interest
    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    assert loop.current_decision_player_id == recipient


def test_bankruptcy_to_bank_auctions_repossessed_mortgaged_property(make_game):
    loop = make_game()
    debtor, bidder = loop.game.turn_order
    tile = loop.game.board.property_tiles[0]
    prop = loop.game.property_states[tile.tile_id]
    prop.owner_id = debtor
    prop.is_mortgaged = True
    loop.game.player_states[debtor].money = 0
    loop.game.debts.append(Debt(debtor, None, 1000))
    loop.turn_phase = GameLoop.RESOLVE_DEBT
    loop._settle()

    act(loop, A.END_DEBT_NEGOTIATIONS)
    act(loop, A.AUCTION_DEBT_PROPERTY, tile_id=tile.tile_id)
    assert loop.current_decision_player_id == bidder
    act(loop, A.AUCTION_WITHDRAW)
    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    assert prop.owner_id == debtor

    act(loop, A.DECLARE_BANKRUPTCY)

    assert loop.turn_phase == GameLoop.AUCTION
    assert loop._auction_for_bankruptcy
    assert loop.current_decision_player_id == bidder
    assert prop.owner_id is None
    assert not prop.is_mortgaged

    act(loop, A.AUCTION_BID, amount=1)

    assert prop.owner_id == bidder
    assert not prop.is_mortgaged
    assert loop.game.player_states[bidder].money == 1499
    assert loop.game.game_over
    assert loop.game.winner_id == bidder


def test_settled_debt_resumes_phase_after_mortgage_transfer(make_game):
    loop = make_game()
    debtor, bidder = loop.game.turn_order
    tile = loop.game.board.property_tiles[0]
    prop = loop.game.property_states[tile.tile_id]
    prop.owner_id = debtor
    prop.is_mortgaged = True
    loop.game.player_states[debtor].money = 0
    loop.game.debts.append(Debt(debtor, None, 1))
    loop.turn_phase = GameLoop.RESOLVE_DEBT
    loop._settle()

    act(loop, A.END_DEBT_NEGOTIATIONS)
    act(loop, A.AUCTION_DEBT_PROPERTY, tile_id=tile.tile_id)
    assert loop.current_decision_player_id == bidder
    act(loop, A.AUCTION_BID, amount=1)
    assert loop.turn_phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER

    act(loop, A.KEEP_MORTGAGE, tile_id=tile.tile_id)

    assert loop.game.debts == []
    assert loop.turn_phase == GameLoop.AFTER_ROLL
