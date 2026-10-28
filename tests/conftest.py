import pytest

from app.application.game_setup import create_game, load_selected_game
from app.domain.game.actions import GameAction
from app.domain.game.dice import DiceMechanism, DiceRoll


class ScriptedDice(DiceMechanism):
    """Dados com resultados pré-definidos, para testar regras."""

    def __init__(self, values):
        self.values = list(values)

    def roll(self) -> DiceRoll:
        return DiceRoll(self.values.pop(0), self.values.pop(0))


@pytest.fixture(scope="session")
def config():
    return load_selected_game()


@pytest.fixture
def make_game(config):
    def _make(num_players=2, dice=(), **kwargs):
        loop = create_game([f"P{i}" for i in range(num_players)], config, seed=0, **kwargs)
        loop.game.dice = ScriptedDice(dice)
        return loop
    return _make


def act(loop, kind, **payload):
    result = loop.process_action(GameAction(kind, payload))
    assert result["status"] == "SUCCESS", result
    return result
