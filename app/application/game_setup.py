import random
import re
from collections.abc import Sequence
from uuid import UUID

from app.core.config import get_app_settings
from app.domain.game.board import (
    Board,
    PropertyKind,
    PropertyTile,
    SpecialTile,
    TaxTile,
    Tile,
)
from app.domain.game.cards import StandardCardDeck
from app.domain.game.dice import StandardDice
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import GameState
from app.domain.game.pieces import Piece
from app.domain.game.player import Player
from app.domain.game.rules import ConfiguredRules
from app.infraestructure.game_config.loader import load_game_config
from app.infraestructure.game_config.models import (
    BoardSpace,
    GameConfig,
    GameSettings,
    PropertyDefinition,
    RailroadDefinition,
    TaxDefinition,
    UtilityDefinition,
)


def load_selected_game() -> GameConfig:
    settings = get_app_settings()

    return load_game_config(
        config_dir=settings.game_config_dir,
        variant=settings.game_variant,
    )


def build_board(config: GameConfig) -> Board:
    """Converte as definições da variante em casas do domínio."""
    return Board([_build_tile(space, config.settings) for space in config.board])


def _build_tile(space: BoardSpace, settings: GameSettings) -> Tile:
    if isinstance(space, PropertyDefinition):
        rent = space.rent
        return PropertyTile(
            tile_id=space.position,
            name=space.name,
            price=space.price,
            rent_values=(
                rent.base,
                rent.one_house,
                rent.two_houses,
                rent.three_houses,
                rent.four_houses,
                rent.hotel,
            ),
            group=space.color_group,
            mortgage_value=space.mortgage,
            kind=PropertyKind.STREET,
            house_cost=space.house_cost,
            hotel_cost=space.hotel_cost,
            unmortgage_cost=space.unmortgage_cost,
            monopoly_rent=rent.monopoly,
        )

    if isinstance(space, (RailroadDefinition, UtilityDefinition)):
        kind = PropertyKind.RAILROAD if isinstance(space, RailroadDefinition) else PropertyKind.UTILITY
        # Chaves "owned1..4" / "multiplier1..2": ordena pelo número final
        values = tuple(
            value
            for _, value in sorted(space.rent.items(), key=lambda item: int(re.search(r"\d+$", item[0]).group()))
        )
        return PropertyTile(
            tile_id=space.position,
            name=space.name,
            price=space.price,
            rent_values=values,
            group=kind,
            mortgage_value=space.mortgage,
            kind=kind,
            unmortgage_cost=space.unmortgage_cost,
        )

    if isinstance(space, TaxDefinition):
        amount = space.amount
        percent_option = space.percent_option
        if space.tax_type == "income":
            amount = settings.income_tax_amount
            percent_option = settings.income_tax_percent
        elif space.tax_type == "luxury":
            amount = settings.luxury_tax_amount
        return TaxTile(
            tile_id=space.position,
            name=space.name,
            tax_amount=amount,
            percent_option=percent_option,
        )

    return SpecialTile(tile_id=space.position, name=space.name, tile_type=space.type)


def build_rules(settings: GameSettings, max_turns: int | None = None,
                auction_enabled: bool | None = None,
                trade_enabled: bool | None = None,
                debt_resolution_enabled: bool | None = None) -> ConfiguredRules:
    return ConfiguredRules(
        starting_money_value=settings.starting_money,
        go_salary_value=settings.go_salary,
        jail_fine_value=settings.jail_fine,
        max_jail_attempts_value=settings.jail_max_turns,
        auction_enabled=settings.auction_enabled if auction_enabled is None else auction_enabled,
        go_to_jail_position_value=settings.go_to_jail_position,
        jail_position_value=settings.jail_position,
        max_houses_value=settings.max_houses_in_bank,
        max_hotels_value=settings.max_hotels_in_bank,
        auction_min_bid_value=settings.auction_min_bid,
        income_tax_choice_value=settings.income_tax_choice,
        max_turns_value=max_turns,
        trade_enabled_value=(
            settings.trade_enabled if trade_enabled is None else trade_enabled
        ),
        trade_max_counteroffers_value=settings.trade_max_counteroffers,
        debt_resolution_enabled_value=(
            settings.debt_resolution_enabled
            if debt_resolution_enabled is None
            else debt_resolution_enabled
        ),
        debt_negotiations_per_player_value=settings.debt_negotiations_per_player,
        bankruptcy_to_bank_value=settings.bankruptcy_to_bank,
    )


def create_game(
    player_names: Sequence[str],
    config: GameConfig | None = None,
    seed: int | None = None,
    max_turns: int | None = None,
    auction_enabled: bool | None = None,
    trade_enabled: bool | None = None,
    debt_resolution_enabled: bool | None = None,
    record_events: bool = True,
) -> GameLoop:
    """Monta uma partida pronta para receber ações.

    `seed` controla dados, embaralhamento e os IDs dos jogadores, então a mesma
    seed com as mesmas ações reproduz a mesma partida.
    """
    config = config or load_selected_game()
    player_count = len(player_names)
    if player_count < config.settings.min_players_to_start:
        raise ValueError(
            f"A partida precisa de pelo menos "
            f"{config.settings.min_players_to_start} jogadores."
        )
    if player_count > config.settings.max_players:
        raise ValueError(
            f"A variante permite no máximo {config.settings.max_players} jogadores."
        )

    rng = random.Random(seed)

    players = [Player(name, player_id=_uuid(rng)) for name in player_names]
    pieces = [
        Piece(f"piece_{i}", [player.player_id], piece_id=_uuid(rng))
        for i, player in enumerate(players)
    ]
    decks = {
        "chance": StandardCardDeck(list(config.chance_cards), seed=rng.getrandbits(32)),
        "community_chest": StandardCardDeck(list(config.community_chest_cards), seed=rng.getrandbits(32)),
    }
    state = GameState(
        players=players,
        pieces=pieces,
        board=build_board(config),
        rules=build_rules(
            config.settings, max_turns, auction_enabled, trade_enabled,
            debt_resolution_enabled,
        ),
        dice=StandardDice(seed=rng.getrandbits(32)),
        decks=decks,
        record_events=record_events,
    )
    return GameLoop(state)


def _uuid(rng: random.Random) -> UUID:
    return UUID(int=rng.getrandbits(128), version=4)
