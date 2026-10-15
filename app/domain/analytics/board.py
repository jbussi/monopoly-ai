
from collections import Counter, defaultdict
from typing import Any
from statistics import mean

from app.infraestructure.game_config.models import GameConfig


def analyze_board(game_config: GameConfig) -> dict[str, Any]:
    """Resume a estrutura do tabuleiro a partir da configuração validada."""

    spaces = game_config.board
    type_counts = Counter(space.type for space in spaces)

    properties = [
        space for space in spaces
        if space.type == "property"
    ]

    property_prices = [space.price for space in properties]
    property_rents = [
        space.rent.base
        for space in properties
        if space.rent is not None
    ]

    groups: dict[str, list[Any]] = defaultdict(list)

    for space in properties:
        groups[space.color_group].append(space)

    return {
        "board_size": len(spaces),
        "space_type_counts": dict(type_counts),
        "property_count": len(properties),
        "property_groups": {
            group: len(group_properties)
            for group, group_properties in groups.items()
        },
        "property_price_total": sum(property_prices),
        "property_price_mean": (
            sum(property_prices) / len(property_prices)
            if property_prices else None
        ),
        "base_rent_mean": (
            sum(property_rents) / len(property_rents)
            if property_rents else None
        ),
    }


def analyze_property_costs(
    game_config: GameConfig,
) -> list[dict[str, Any]]:
    """Gera indicadores básicos de preço e aluguel das propriedades."""

    results = []

    for space in game_config.board:
        if space.type != "property":
            continue

        rent = space.rent
        base_rent = rent.base if rent is not None else None

        results.append({
            "position": space.position,
            "name": space.name,
            "color_group": space.color_group,
            "price": space.price,
            "house_cost": space.house_cost,
            "hotel_cost": space.hotel_cost,
            "mortgage": space.mortgage,
            "unmortgage_cost": space.unmortgage_cost,
            "base_rent": base_rent,
            "base_rent_to_price": (
                base_rent / space.price
                if base_rent is not None and space.price > 0
                else None
            ),
        })

    return results

def build_property_table(
    game_config: GameConfig,
) -> list[dict[str, Any]]:
    """Produz registros tabulares das propriedades do tabuleiro."""

    records: list[dict[str, Any]] = []

    for space in game_config.board:
        if space.type != "property":
            continue

        rent = space.rent

        if rent is None:
            raise ValueError(
                f"Propriedade na posição {space.position} "
                "não possui tabela de aluguel."
            )

        records.append({
            "position": space.position,
            "name": space.name,
            "color_group": space.color_group,
            "price": space.price,
            "house_cost": space.house_cost,
            "hotel_cost": space.hotel_cost,
            "mortgage": space.mortgage,
            "unmortgage_cost": space.unmortgage_cost,
            "rent_base": rent.base,
            "rent_monopoly": rent.monopoly,
            "rent_one_house": rent.one_house,
            "rent_two_houses": rent.two_houses,
            "rent_three_houses": rent.three_houses,
            "rent_four_houses": rent.four_houses,
            "rent_hotel": rent.hotel,
        })

    return records



def summarize_property_groups(
    property_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resume preços e aluguéis-base por grupo de cores."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for property_data in property_metrics:
        groups[property_data["color_group"]].append(property_data)

    summaries = []

    for group, properties in groups.items():
        prices = [p["price"] for p in properties]
        rents = [
            p["base_rent"]
            for p in properties
            if p["base_rent"] is not None
        ]

        summaries.append({
            "color_group": group,
            "property_count": len(properties),
            "total_purchase_cost": sum(prices),
            "mean_purchase_price": sum(prices) / len(prices),
            "mean_base_rent": (
                sum(rents) / len(rents) if rents else None
            ),
        })

    return summaries


def build_space_table(game_config: GameConfig) -> list[dict[str, Any]]:
    """Converte as casas do tabuleiro em registros tabulares."""
    records: list[dict[str, Any]] = []

    for space in game_config.board:
        record: dict[str, Any] = {
            "position": space.position,
            "name": space.name,
            "type": space.type,
            "description": getattr(space, "description", None),
        }

        # Campos específicos de propriedades, ferrovias e utilidades.
        for attr, column in (
            ("color_group", "color_group"),
            ("price", "price"),
            ("house_cost", "house_cost"),
            ("hotel_cost", "hotel_cost"),
            ("mortgage", "mortgage"),
            ("unmortgage_cost", "unmortgage_cost"),
            ("amount", "tax_amount"),
            ("tax_type", "tax_type"),
            ("percent_option", "percent_option"),
        ):
            if hasattr(space, attr):
                record[column] = getattr(space, attr)

        rent = getattr(space, "rent", None)
        if rent is not None:
            if hasattr(rent, "base"):
                rent_values = {
                    "rent_base": rent.base,
                    "rent_monopoly": rent.monopoly,
                    "rent_one_house": rent.one_house,
                    "rent_two_houses": rent.two_houses,
                    "rent_three_houses": rent.three_houses,
                    "rent_four_houses": rent.four_houses,
                    "rent_hotel": rent.hotel,
                }
            elif isinstance(rent, dict):
                rent_values = {f"rent_{key}": value for key, value in rent.items()}
            else:
                rent_values = {}
            record.update(rent_values)

        records.append(record)

    return records


def build_property_table(game_config: GameConfig) -> list[dict[str, Any]]:
    """Retorna somente propriedades, com a tabela completa de aluguel."""
    records: list[dict[str, Any]] = []

    for space in game_config.board:
        if space.type != "property":
            continue

        rent = getattr(space, "rent", None)
        if rent is None or not hasattr(rent, "base"):
            raise ValueError(
                f"Propriedade na posição {space.position} "
                "não possui uma tabela de aluguel compatível."
            )
        if not space.price:
            raise ValueError(
                f"Propriedade na posição {space.position} tem preço inválido."
            )

        records.append({
            "position": space.position,
            "name": space.name,
            "color_group": space.color_group,
            "price": space.price,
            "house_cost": space.house_cost,
            "hotel_cost": space.hotel_cost,
            "mortgage": space.mortgage,
            "unmortgage_cost": space.unmortgage_cost,
            "rent_base": rent.base,
            "rent_monopoly": rent.monopoly,
            "rent_one_house": rent.one_house,
            "rent_two_houses": rent.two_houses,
            "rent_three_houses": rent.three_houses,
            "rent_four_houses": rent.four_houses,
            "rent_hotel": rent.hotel,
            "base_rent_to_price": rent.base / space.price,
            "monopoly_rent_to_price": rent.monopoly / space.price,
        })

    return records





def summarize_property_groups(
    property_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resume preço, aluguel e custo de aquisição por grupo de cor."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in property_metrics:
        groups[item["color_group"]].append(item)

    summaries: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        prices = [item["price"] for item in items]
        base_rents = [
            item["rent_base"]
            for item in items
            if item.get("rent_base") is not None
        ]
        summaries.append({
            "color_group": group,
            "property_count": len(items),
            "total_purchase_cost": sum(prices),
            "mean_purchase_price": mean(prices) if prices else None,
            "mean_base_rent": mean(base_rents) if base_rents else None,
            "mean_base_rent_to_price": (
                mean(item["base_rent_to_price"] for item in items)
                if items else None
            ),
            "house_cost": items[0].get("house_cost"),
            "hotel_cost": items[0].get("hotel_cost"),
        })

    return summaries
