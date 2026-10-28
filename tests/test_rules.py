import pytest

from app.application.game_setup import create_game
from app.domain.game.actions import ActionType as A
from app.domain.game.actions import GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import Debt, HOTEL_LEVEL

from .conftest import act

MEDITERRANEAN, BALTIC, READING, ORIENTAL, ELECTRIC = 1, 3, 5, 6, 12


def ids(loop):
    return loop.game.turn_order


def money(loop, seat):
    return loop.game.player_states[ids(loop)[seat]].money


def own(loop, seat, *tile_ids):
    for tile_id in tile_ids:
        loop.game.property_states[tile_id].owner_id = ids(loop)[seat]


def test_board_built_from_config(make_game):
    board = make_game().game.board
    assert board.size == 40
    assert len(board.property_tiles) == 28
    assert len({tile.group for tile in board.property_tiles}) == 10


def test_game_player_count_is_between_two_and_twelve(config):
    with pytest.raises(ValueError, match="pelo menos 2 jogadores"):
        create_game(["P0"], config)

    loop = create_game([f"P{i}" for i in range(12)], config, seed=0)
    assert len(loop.game.turn_order) == 12

    with pytest.raises(ValueError, match="no máximo 12 jogadores"):
        create_game([f"P{i}" for i in range(13)], config)


def test_buy_property(make_game):
    loop = make_game(dice=[1, 2])
    act(loop, A.ROLL_DICE)
    assert loop.turn_phase == GameLoop.AWAITING_PROPERTY_DECISION
    act(loop, A.BUY_PROPERTY)
    assert loop.game.property_states[BALTIC].owner_id == ids(loop)[0]
    assert money(loop, 0) == 1500 - 60 and loop.turn_phase == GameLoop.AFTER_ROLL


def test_decline_without_auction_stays_with_bank(make_game):
    loop = make_game(dice=[1, 2], auction_enabled=False)
    act(loop, A.ROLL_DICE)
    act(loop, A.DECLINE_PROPERTY)
    assert loop.game.property_states[BALTIC].owner_id is None
    assert loop.turn_phase == GameLoop.AFTER_ROLL


def test_auction_flow(make_game):
    loop = make_game(3, dice=[1, 2])
    p0, p1, p2 = ids(loop)
    act(loop, A.ROLL_DICE)
    act(loop, A.DECLINE_PROPERTY)
    assert loop.turn_phase == GameLoop.AUCTION and loop.current_decision_player_id == p0
    act(loop, A.AUCTION_WITHDRAW)
    act(loop, A.AUCTION_BID, amount=20)        # P1
    act(loop, A.AUCTION_BID, amount=30)        # P2
    assert loop.current_decision_player_id == p1
    act(loop, A.AUCTION_WITHDRAW)              # P1 desiste; P2 vence
    assert loop.game.property_states[BALTIC].owner_id == p2
    assert loop.game.player_states[p2].money == 1470
    assert loop.turn_phase == GameLoop.AFTER_ROLL and loop.current_decision_player_id == p0


def test_cannot_end_turn_before_rolling(make_game):
    loop = make_game()
    result = loop.process_action(GameAction(A.END_TURN))
    assert result["status"] == "REJECTED"
    assert loop.turn_phase == GameLoop.BEFORE_ROLL


def test_pass_go_salary(make_game):
    loop = make_game(dice=[3, 4])
    loop.game.set_position(ids(loop)[0], 36)
    act(loop, A.ROLL_DICE)
    assert loop.game.position_of(ids(loop)[0]) == 3 and money(loop, 0) == 1700


def test_street_rent(make_game):
    loop = make_game(dice=[1, 2])
    own(loop, 1, BALTIC)
    act(loop, A.ROLL_DICE)
    assert money(loop, 0) == 1496 and money(loop, 1) == 1504
    tile = loop.game.board.get_tile_by_id(BALTIC)
    own(loop, 1, MEDITERRANEAN)
    assert loop.game.rent_for(tile) == 8           # valor "monopoly" do board.json
    loop.game.property_states[BALTIC].houses = 3
    assert loop.game.rent_for(tile) == 180
    loop.game.property_states[BALTIC].is_mortgaged = True
    assert loop.game.rent_for(tile) == 0


def test_railroad_and_utility_rent(make_game):
    loop = make_game()
    board = loop.game.board
    for i, tile in enumerate(board.group_members("railroad")):
        own(loop, 1, tile.tile_id)
        assert loop.game.rent_for(tile) == 25 * 2 ** i
    electric, water = board.group_members("utility")
    own(loop, 1, electric.tile_id)
    assert loop.game.rent_for(electric, dice_total=7) == 28
    own(loop, 1, water.tile_id)
    assert loop.game.rent_for(electric, dice_total=7) == 70


def test_income_tax_takes_cheaper_option(make_game):
    loop = make_game(dice=[1, 3])
    act(loop, A.ROLL_DICE)
    assert money(loop, 0) == 1500 - 150             # 10% de 1500 < 200


def test_doubles_extra_roll_and_three_doubles_jail(make_game):
    loop = make_game(dice=[2, 2, 3, 3, 4, 4])
    act(loop, A.ROLL_DICE)                          # 0 -> 4 (imposto)
    assert loop.turn_phase == GameLoop.BEFORE_ROLL
    act(loop, A.ROLL_DICE)                          # 4 -> 10 (visitando)
    assert loop.turn_phase == GameLoop.BEFORE_ROLL
    act(loop, A.ROLL_DICE)                          # terceiro duplo
    state = loop.game.player_states[ids(loop)[0]]
    assert state.is_in_jail and loop.game.position_of(ids(loop)[0]) == 10
    assert loop.turn_phase == GameLoop.AFTER_ROLL


def test_jail_forced_fine_on_third_attempt(make_game):
    loop = make_game(dice=[1, 2, 1, 2, 1, 2])
    p0 = ids(loop)[0]
    state = loop.game.player_states[p0]
    state.is_in_jail = True
    loop.game.set_position(p0, 10)
    for _ in range(2):
        act(loop, A.ROLL_DICE)
        assert state.is_in_jail
        loop.turn_phase = GameLoop.BEFORE_ROLL
    act(loop, A.ROLL_DICE)
    assert not state.is_in_jail and loop.game.position_of(p0) == 13
    assert state.money == 1450


def test_jail_escape_with_double_has_no_extra_roll(make_game):
    loop = make_game(dice=[3, 3], auction_enabled=False)
    p0 = ids(loop)[0]
    loop.game.player_states[p0].is_in_jail = True
    loop.game.set_position(p0, 10)
    act(loop, A.ROLL_DICE)
    act(loop, A.DECLINE_PROPERTY)
    assert loop.turn_phase == GameLoop.AFTER_ROLL


def test_jail_card_returns_to_deck(make_game):
    loop = make_game()
    deck = loop.game.decks["chance"]
    card = next(c for c in deck._cards if c.action == "get_out_of_jail")
    deck._cards.remove(card)
    state = loop.game.player_states[ids(loop)[0]]
    state.jail_cards.append(("chance", card))
    state.is_in_jail = True
    act(loop, A.USE_JAIL_CARD)
    assert not state.is_in_jail and not state.jail_cards and card in deck._discard_pile


def test_even_building_and_bank_supply(make_game):
    loop = make_game()
    assert loop.process_action(GameAction(A.BUILD_HOUSE, {"tile_id": MEDITERRANEAN}))["status"] == "REJECTED"
    own(loop, 0, MEDITERRANEAN, BALTIC)
    act(loop, A.BUILD_HOUSE, tile_id=MEDITERRANEAN)
    assert loop.process_action(GameAction(A.BUILD_HOUSE, {"tile_id": MEDITERRANEAN}))["status"] == "REJECTED"
    act(loop, A.BUILD_HOUSE, tile_id=BALTIC)
    assert loop.game.houses_in_bank == 30 and money(loop, 0) == 1400
    assert loop.process_action(GameAction(A.MORTGAGE, {"tile_id": MEDITERRANEAN}))["status"] == "REJECTED"

    for tile_id in (MEDITERRANEAN, BALTIC):
        loop.game.property_states[tile_id].houses = 4
    act(loop, A.BUILD_HOUSE, tile_id=MEDITERRANEAN)
    assert loop.game.property_states[MEDITERRANEAN].houses == HOTEL_LEVEL
    assert loop.game.hotels_in_bank == 11


def test_mortgage_and_unmortgage(make_game):
    loop = make_game()
    own(loop, 0, MEDITERRANEAN)
    act(loop, A.MORTGAGE, tile_id=MEDITERRANEAN)
    assert money(loop, 0) == 1530
    act(loop, A.UNMORTGAGE, tile_id=MEDITERRANEAN)
    assert money(loop, 0) == 1530 - 33              # unmortgageCost do board.json


def test_debt_resolution_by_mortgage(make_game):
    loop = make_game(dice=[1, 2])
    own(loop, 1, BALTIC)
    own(loop, 0, ORIENTAL)
    loop.game.player_states[ids(loop)[0]].money = 2
    act(loop, A.ROLL_DICE)
    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    kinds = {a.action_type for a in loop.legal_actions()}
    assert kinds == {A.MORTGAGE, A.END_DEBT_NEGOTIATIONS}
    result = loop.process_action(GameAction(A.DECLARE_BANKRUPTCY))
    assert result["status"] == "REJECTED"
    assert "Encerre as negociações" in result["reason"]
    assert not loop.game.player_states[ids(loop)[0]].is_bankrupt
    act(loop, A.MORTGAGE, tile_id=ORIENTAL)
    assert loop.turn_phase == GameLoop.AFTER_ROLL
    assert money(loop, 0) == 2 + 50 - 4 and money(loop, 1) == 1504


def test_insolvent_player_can_auction_before_bankruptcy_to_creditor(make_game):
    loop = make_game(dice=[1, 2])
    own(loop, 1, BALTIC)
    own(loop, 0, ORIENTAL)
    loop.game.property_states[ORIENTAL].is_mortgaged = True
    loop.game.player_states[ids(loop)[0]].money = 1
    act(loop, A.ROLL_DICE)
    game = loop.game
    debtor, creditor = ids(loop)[:2]
    assert not game.player_states[debtor].is_bankrupt
    act(loop, A.END_DEBT_NEGOTIATIONS)
    assert any(
        action.action_type == A.AUCTION_DEBT_PROPERTY and action.payload["tile_id"] == ORIENTAL
        for action in loop.legal_actions()
    )
    act(loop, A.AUCTION_DEBT_PROPERTY, tile_id=ORIENTAL)
    assert loop.current_decision_player_id == creditor
    act(loop, A.AUCTION_BID, amount=1)
    assert loop.game.property_states[ORIENTAL].owner_id == creditor
    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    act(loop, A.DECLARE_BANKRUPTCY)
    assert game.player_states[debtor].is_bankrupt
    assert game.game_over and game.winner_id == creditor
    assert money(loop, 1) >= 1500


def test_hotel_can_be_sold_without_house_inventory_during_debt(make_game):
    loop = make_game()
    debtor, creditor = ids(loop)
    tile = loop.game.board.property_tiles[0]
    loop.game.property_states[tile.tile_id].owner_id = debtor
    loop.game.property_states[tile.tile_id].houses = 5
    loop.game.houses_in_bank = 0
    loop.game.player_states[debtor].money = 0
    loop.game.debts.append(Debt(debtor, creditor, 500))
    loop.turn_phase = GameLoop.RESOLVE_DEBT

    actions = loop.legal_actions()
    assert A.SELL_HOUSE in {action.action_type for action in actions}
    assert A.DECLARE_BANKRUPTCY not in {action.action_type for action in actions}
    assert loop.validate(GameAction(A.SELL_HOUSE, {"tile_id": tile.tile_id})) is None

    money_before = loop.game.player_states[debtor].money
    act(loop, A.SELL_HOUSE, tile_id=tile.tile_id)
    assert loop.game.property_states[tile.tile_id].houses == HOTEL_LEVEL - 1
    assert loop.game.player_states[debtor].money == money_before + (tile.hotel_cost or tile.house_cost) // 2
    assert loop.game.houses_in_bank == 0
    assert loop.game.house_shortage == HOTEL_LEVEL - 1

    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    act(loop, A.END_DEBT_NEGOTIATIONS)
    assert loop.game.debt_resolution.negotiations_closed
    for _ in range(HOTEL_LEVEL - 1):
        act(loop, A.SELL_HOUSE, tile_id=tile.tile_id)
    assert loop.game.house_shortage == 0
    assert A.AUCTION_DEBT_PROPERTY in {action.action_type for action in loop.legal_actions()}


def test_birthday_debtor_acts_outside_own_turn(make_game):
    loop = make_game(3, dice=[1, 1])
    p0, p1, p2 = ids(loop)
    deck = loop.game.decks["community_chest"]
    card = next(c for c in deck._cards if c.id == "cc_birthday")
    deck._cards.remove(card)
    deck._cards.insert(0, card)
    own(loop, 2, ELECTRIC)
    loop.game.player_states[p2].money = 5
    act(loop, A.ROLL_DICE)                          # duplo 1+1 -> Cofre -> aniversário
    assert loop.turn_phase == GameLoop.RESOLVE_DEBT
    assert loop.current_decision_player_id == p2 and loop.game.current_player_id == p0
    act(loop, A.MORTGAGE, tile_id=ELECTRIC)
    assert loop.turn_phase == GameLoop.BEFORE_ROLL   # volta à rolagem extra do duplo
    assert loop.current_decision_player_id == p0
    assert money(loop, 0) == 1520 and money(loop, 2) == 5 + 75 - 10


def test_truncation(make_game):
    loop = make_game(dice=[1, 2] * 10, max_turns=3, auction_enabled=False)
    for _ in range(3):
        act(loop, A.ROLL_DICE)
        if loop.turn_phase == GameLoop.AWAITING_PROPERTY_DECISION:
            act(loop, A.DECLINE_PROPERTY)
        act(loop, A.END_TURN)
    assert loop.game.game_over and loop.game.truncated and loop.game.winner_id is None
