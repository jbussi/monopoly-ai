
import json
from pathlib import Path
from typing import Any

from app.infraestructure.game_config.models import (
    BoardSpace,
    CardDefinition,
    GameConfig,
    PropertyDefinition,
    RailroadDefinition,
    RentSchedule,
    TaxDefinition,
    UtilityDefinition,
    GameSettings
)

from app.domain.game.card_actions import CardAction


SUPPORTED_SPACE_TYPES = {
    "go",
    "property",
    "community_chest",
    "tax",
    "railroad",
    "chance",
    "jail",
    "utility",
    "free_parking",
    "go_to_jail",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido em {path}, linha {exc.lineno}: {exc.msg}"
        ) from exc


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} deve ser um objeto JSON.")
    return value


def _require_int(
    data: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{context}: '{key}' deve ser um inteiro. Valor: {value!r}"
        )

    if value < minimum:
        raise ValueError(
            f"{context}: '{key}' deve ser >= {minimum}. Valor: {value}"
        )

    return value


def _require_text(
    data: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: '{key}' deve ser um texto não vazio.")

    return value


def _parse_rent(data: dict[str, Any]) -> RentSchedule:
    rent = _require_object(data.get("rent"), f"Aluguéis de {data.get('name')!r}")
    context = f"Aluguéis de {data.get('name')!r}"

    return RentSchedule(
        base=_require_int(rent, "base", context),
        monopoly=_require_int(rent, "monopoly", context),
        one_house=_require_int(rent, "oneHouse", context),
        two_houses=_require_int(rent, "twoHouses", context),
        three_houses=_require_int(rent, "threeHouses", context),
        four_houses=_require_int(rent, "fourHouses", context),
        hotel=_require_int(rent, "hotel", context),
    )


def _parse_special_rent(
    data: dict[str, Any],
    context: str,
) -> dict[str, int]:
    rent = _require_object(data.get("rent"), context)

    if not rent:
        raise ValueError(f"{context}: o aluguel não pode estar vazio.")

    parsed: dict[str, int] = {}

    for key, value in rent.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context}: chave de aluguel inválida.")

        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{context}: o aluguel '{key}' deve ser um inteiro >= 0."
            )

        parsed[key] = value

    return parsed


def _parse_space(raw: Any) -> BoardSpace:
    data = _require_object(raw, "Cada casa do tabuleiro")
    position = _require_int(data, "position", "Casa do tabuleiro")
    name = _require_text(data, "name", f"Casa na posição {position}")
    space_type = _require_text(data, "type", f"Casa {name!r}")

    if space_type not in SUPPORTED_SPACE_TYPES:
        raise ValueError(
            f"Tipo de casa desconhecido em {name!r}: {space_type!r}"
        )

    description = data.get("description")

    if description is not None and not isinstance(description, str):
        raise ValueError(f"A descrição de {name!r} deve ser texto.")

    common = {
        "position": position,
        "type": space_type,
        "name": name,
        "description": description,
    }

    if space_type == "property":
        color_group = _require_text(
            data, "colorGroup", f"Propriedade {name!r}"
        )

        return PropertyDefinition(
            **common,
            color_group=color_group,
            price=_require_int(data, "price", name),
            house_cost=_require_int(data, "houseCost", name),
            hotel_cost=_require_int(data, "hotelCost", name),
            mortgage=_require_int(data, "mortgage", name),
            unmortgage_cost=_require_int(data, "unmortgageCost", name),
            rent=_parse_rent(data),
        )

    if space_type == "railroad":
        return RailroadDefinition(
            **common,
            price=_require_int(data, "price", name),
            mortgage=_require_int(data, "mortgage", name),
            unmortgage_cost=_require_int(data, "unmortgageCost", name),
            rent=_parse_special_rent(data, f"Aluguéis de {name!r}"),
        )

    if space_type == "utility":
        return UtilityDefinition(
            **common,
            price=_require_int(data, "price", name),
            mortgage=_require_int(data, "mortgage", name),
            unmortgage_cost=_require_int(data, "unmortgageCost", name),
            rent=_parse_special_rent(data, f"Aluguéis de {name!r}"),
        )

    if space_type == "tax":
        percent_option = data.get("percentOption")
        tax_type = data.get("taxType")

        if percent_option is not None and (
            isinstance(percent_option, bool)
            or not isinstance(percent_option, int)
            or percent_option < 0
        ):
            raise ValueError(
                f"'percentOption' inválido na casa {name!r}."
            )

        if tax_type is not None and not isinstance(tax_type, str):
            raise ValueError(f"'taxType' inválido na casa {name!r}.")

        return TaxDefinition(
            **common,
            amount=_require_int(data, "amount", name),
            percent_option=percent_option,
            tax_type=tax_type,
        )

    return BoardSpace(**common)


def _validate_board(board: tuple[BoardSpace, ...]) -> None:
    if not board:
        raise ValueError("O tabuleiro não pode estar vazio.")

    positions = [space.position for space in board]
    expected_positions = list(range(len(board)))

    if positions != expected_positions:
        raise ValueError(
            "As casas devem estar ordenadas por posição, começando em zero, "
            "sem lacunas ou duplicações."
        )


def _parse_cards(raw: Any, deck_name: str) -> tuple[CardDefinition, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"O baralho {deck_name!r} deve ser uma lista.")

    cards: list[CardDefinition] = []

    for index, item in enumerate(raw):
        context = f"Carta {index} do baralho {deck_name!r}"
        data = _require_object(item, context)

        card_id = _require_text(data, "id", context)
        text = _require_text(data, "text", context)
        action = _require_text(data, "action", context)
        supported_actions = {action.value for action in CardAction}

        if action not in supported_actions:
            raise ValueError(
                f"{context}: ação '{action}' não suportada "
                f"na carta '{card_id}'."
            )
        card_data = _require_object(data.get("data"), f"Dados de {card_id!r}")

        cards.append(
            CardDefinition(
                id=card_id,
                text=text,
                action=action,
                data=card_data,
            )
        )

    ids = [card.id for card in cards]

    if len(ids) != len(set(ids)):
        raise ValueError(f"O baralho {deck_name!r} contém IDs duplicados.")

    return tuple(cards)


def _parse_settings(raw: dict[str, Any]) -> GameSettings:
    def integer(key: str) -> int:
        return _require_int(raw, key, "settings.json")

    def boolean(key: str) -> bool:
        value = raw.get(key)

        if not isinstance(value, bool):
            raise ValueError(
                f"settings.json: '{key}' deve ser booleano."
            )

        return value

    player_colors = raw.get("playerColors")
    player_tokens = raw.get("playerTokens")

    if not isinstance(player_colors, list):
        raise ValueError(
            "settings.json: 'playerColors' deve ser uma lista."
        )

    if not all(isinstance(color, dict) for color in player_colors):
        raise ValueError(
            "settings.json: cada item de 'playerColors' deve ser um objeto."
        )

    if not isinstance(player_tokens, list) or not all(
        isinstance(token, str) and token.strip()
        for token in player_tokens
    ):
        raise ValueError(
            "settings.json: 'playerTokens' deve ser uma lista de textos."
        )

    return GameSettings(
        starting_money=integer("startingMoney"),
        go_salary=integer("goSalary"),
        income_tax_amount=integer("incomeTaxAmount"),
        income_tax_percent=integer("incomeTaxPercent"),
        income_tax_choice=boolean("incomeTaxChoice"),
        luxury_tax_amount=integer("luxuryTaxAmount"),
        jail_fine=integer("jailFine"),
        jail_max_turns=integer("jailMaxTurns"),
        go_to_jail_position=integer("goToJailPosition"),
        jail_position=integer("jailPosition"),
        max_houses_in_bank=integer("maxHousesInBank"),
        max_hotels_in_bank=integer("maxHotelsInBank"),
        auction_enabled=boolean("auctionEnabled"),
        auction_min_bid=integer("auctionMinBid"),
        auction_timeout_seconds=integer("auctionTimeoutSeconds"),
        free_parking_jackpot=boolean("freeParkingJackpot"),
        max_players=integer("maxPlayers"),
        min_players_to_start=integer("minPlayersToStart"),
        allow_spectators=boolean("allowSpectators"),
        trade_enabled=boolean("tradeEnabled"),
        bankruptcy_to_bank=boolean("bankruptcyToBank"),
        player_colors=tuple(dict(color) for color in player_colors),
        player_tokens=tuple(player_tokens),
    )


def _validate_settings(settings: GameSettings, board_size: int) -> None:
    if settings.min_players_to_start < 1:
        raise ValueError("É necessário permitir ao menos um jogador.")

    if settings.max_players < settings.min_players_to_start:
        raise ValueError(
            "'maxPlayers' não pode ser menor que 'minPlayersToStart'."
        )

    if settings.jail_max_turns < 1:
        raise ValueError("'jailMaxTurns' deve ser positivo.")

    if settings.auction_enabled and settings.auction_timeout_seconds < 1:
        raise ValueError(
            "'auctionTimeoutSeconds' deve ser positivo "
            "quando os leilões estão habilitados."
        )

    for field_name, position in (
        ("goToJailPosition", settings.go_to_jail_position),
        ("jailPosition", settings.jail_position),
    ):
        if not 0 <= position < board_size:
            raise ValueError(
                f"'{field_name}' aponta para uma posição inexistente: {position}."
            )




def load_game_config(
    config_dir: Path,
    variant: str = "classic_us",
) -> GameConfig:
    config_dir = Path(config_dir)

    raw_board = _read_json(config_dir / "board.json")
    raw_cards = _read_json(config_dir / "cards.json")
    raw_settings = _read_json(config_dir / "settings.json")

    if not isinstance(raw_board, list):
        raise ValueError("'board.json' deve conter uma lista.")

    raw_cards = _require_object(raw_cards, "'cards.json'")
    raw_settings = _require_object(raw_settings, "'settings.json'")

    board = tuple(_parse_space(item) for item in raw_board)
    _validate_board(board)

    chance_cards = _parse_cards(
        raw_cards.get("chance"), "chance"
    )
    community_chest_cards = _parse_cards(
        raw_cards.get("communityChest"), "communityChest"
    )

    settings = _parse_settings(
        {
            key: value
            for key, value in raw_settings.items()
            if not key.startswith("_")
        }
    )
    _validate_settings(settings, len(board))

    return GameConfig(
        board=board,
        chance_cards=chance_cards,
        community_chest_cards=community_chest_cards,
        settings=settings,
        variant=variant,
    )




