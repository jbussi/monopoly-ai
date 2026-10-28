import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from app.infraestructure.game_config.loader import load_game_config
from app.application.game_setup import build_board, create_game
from app.domain.game.board import TaxTile


CLASSIC_CONFIG = Path(__file__).parents[1] / "configs" / "games" / "classic_us"


@pytest.mark.parametrize(
    ("action", "card_data"),
    [
        ("advance_to", {}),
        ("advance_to", {"position": 40}),
        ("advance_to", {"position": True}),
        ("advance_to", {"position": 0, "collectGoSalary": "yes"}),
        ("advance_to_nearest", {"type": "station"}),
        ("advance_to_nearest", {"type": "railroad", "rentMultiplier": 0}),
        ("collect", {}),
        ("collect", {"amount": True}),
        ("go_back", {"spaces": 0}),
        ("repairs", {"houseCost": 25}),
    ],
)
def test_loader_rejects_invalid_card_data(tmp_path, action, card_data):
    for filename in ("board.json", "cards.json", "settings.json"):
        shutil.copy(CLASSIC_CONFIG / filename, tmp_path / filename)

    cards_path = tmp_path / "cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    cards["chance"][0]["action"] = action
    cards["chance"][0]["data"] = card_data
    cards_path.write_text(json.dumps(cards), encoding="utf-8")

    with pytest.raises(ValueError, match="Dados da carta"):
        load_game_config(tmp_path)


def test_loader_accepts_classic_game_config():
    config = load_game_config(CLASSIC_CONFIG)

    assert len(config.chance_cards) == 16
    assert len(config.community_chest_cards) == 16
    assert config.settings.min_players_to_start == 2
    assert config.settings.max_players == 12


@pytest.mark.parametrize(
    ("key", "value"),
    [("minPlayersToStart", 1), ("maxPlayers", 13)],
)
def test_loader_rejects_player_limits_outside_two_to_twelve(tmp_path, key, value):
    for filename in ("board.json", "cards.json", "settings.json"):
        shutil.copy(CLASSIC_CONFIG / filename, tmp_path / filename)

    settings_path = tmp_path / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings[key] = value
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    with pytest.raises(ValueError):
        load_game_config(tmp_path)


def test_settings_override_duplicate_tax_values_and_jail_position():
    config = load_game_config(CLASSIC_CONFIG)
    settings = replace(
        config.settings,
        income_tax_amount=175,
        income_tax_percent=7,
        luxury_tax_amount=35,
        go_to_jail_position=31,
    )
    configured = replace(config, settings=settings)

    board = build_board(configured)
    income_tax = next(tile for tile in board.tiles if tile.name == "Income Tax")
    luxury_tax = next(tile for tile in board.tiles if tile.name == "Luxury Tax")
    assert isinstance(income_tax, TaxTile)
    assert (income_tax.tax_amount, income_tax.percent_option) == (175, 7)
    assert isinstance(luxury_tax, TaxTile)
    assert luxury_tax.tax_amount == 35

    loop = create_game(["A", "B"], configured, seed=0)
    player_id = loop.game.turn_order[0]
    loop.game.set_position(player_id, 30)
    loop._land(player_id)
    assert not loop.game.player_states[player_id].is_in_jail

    loop.game.set_position(player_id, 31)
    loop._land(player_id)
    assert loop.game.player_states[player_id].is_in_jail
