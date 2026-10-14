"""Validações estruturais da configuração do jogo."""

from typing import Any


def validate_board(data: Any) -> None:
    if not isinstance(data, list) or not data:
        raise ValueError("O tabuleiro deve ser uma lista não vazia.")

    for index, space in enumerate(data):
        if not isinstance(space, dict):
            raise ValueError(
                f"A casa na posição {index} deve ser um objeto JSON."
            )


def validate_cards(data: Any) -> None:
    if not isinstance(data, dict) or not data:
        raise ValueError("O arquivo de cartas deve conter um objeto JSON.")


def validate_settings(data: Any) -> None:
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "O arquivo de configurações deve conter um objeto JSON."
        )