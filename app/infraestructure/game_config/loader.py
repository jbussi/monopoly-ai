"""Carregamento das configurações externas do Monopoly."""

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from models import GameConfig
from validators import (
    validate_board,
    validate_cards,
    validate_settings,
)


def _read_json(path: Path) -> Any:
    """Lê um arquivo JSON e apresenta erros com contexto."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido em {path}, linha {exc.lineno}, "
            f"coluna {exc.colno}: {exc.msg}"
        ) from exc


def load_game_config(
    config_dir: Path | None = None,
    variant: str | None = None,
) -> GameConfig:
    """Carrega e valida os arquivos de uma variante do jogo."""
    selected_variant = variant or settings.game_variant

    directory = (
        config_dir
        if config_dir is not None
        else settings.configs_dir / "games" / selected_variant
    )

    board = _read_json(directory / "board.json")
    cards = _read_json(directory / "cards.json")
    game_settings = _read_json(directory / "settings.json")

    validate_board(board)
    validate_cards(cards)
    validate_settings(game_settings)

    return GameConfig(
        board=board,
        cards=cards,
        settings=game_settings,
        variant=selected_variant,
    )