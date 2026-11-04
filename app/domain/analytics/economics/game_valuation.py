"""Long-run rent valuations applied to a live Monopoly game state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping
from uuid import UUID

from app.domain.analytics.markov.stationary_board import (
    StationaryBoardAnalysis,
    calculate_stationary_board_analysis,
)
from app.domain.game.board import PropertyKind, PropertyTile
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import HOTEL_LEVEL
from app.infraestructure.game_config.models import GameConfig


DEFAULT_VALUE_HORIZONS = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class PropertyScenarioState:
    """Complete ownership/building state for one property in a scenario."""

    owner_id: UUID | None
    houses: int
    is_mortgaged: bool


@dataclass(frozen=True, slots=True)
class PortfolioHorizonValue:
    horizon_rounds: int
    incremental_operating_cash_flow: float
    immediate_cash_flow: float
    expected_net_cash_flow: float
    roi_on_cash_outlay: float | None
    net_present_value: float | None


@dataclass(frozen=True, slots=True)
class PortfolioScenarioValuation:
    player_id: UUID
    active_opponents: int
    immediate_cash_flow: float
    current_rent_income_per_round: float
    projected_rent_income_per_round: float
    incremental_rent_income_per_round: float
    horizons: tuple[PortfolioHorizonValue, ...]

    def for_rounds(self, rounds: int) -> PortfolioHorizonValue:
        """Return the valuation entry calculated for the given horizon."""
        for value in self.horizons:
            if value.horizon_rounds == rounds:
                return value
        raise KeyError(f"Horizonte não calculado: {rounds} rodada(s).")


@dataclass(frozen=True, slots=True)
class PropertyAcquisitionValuation:
    tile_id: int
    name: str
    group: str
    acquisition_price: float
    owned_group_properties_before: int
    owned_group_properties_after: int
    group_size: int
    completes_group: bool
    current_group_rent_income_per_round: float
    projected_group_rent_income_per_round: float
    portfolio_impact: PortfolioScenarioValuation


@dataclass(frozen=True, slots=True)
class PropertyGroupAnalysis:
    player_id: UUID
    group: str
    group_size: int
    owned_group_properties: int
    expected_group_rent_income_per_round: float
    available_acquisitions: tuple[PropertyAcquisitionValuation, ...]


def analyze_portfolio_scenario(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    property_changes: Mapping[int, PropertyScenarioState],
    *,
    immediate_cash_flow: float = 0.0,
    horizons: Iterable[int] = DEFAULT_VALUE_HORIZONS,
    discount_rate_per_round: float | None = None,
    stationary: StationaryBoardAnalysis | None = None,
) -> PortfolioScenarioValuation:
    """Value an explicit property-state scenario against the current portfolio.

    The long-run property configuration stays fixed after the hypothetical
    change. Every other active player contributes one stationary movement
    stream per table round. Immediate money spent or received is supplied as a
    signed cash flow (for example, a purchase price is negative).
    """
    horizon_values = _validate_horizons(horizons)
    immediate_cash_flow = _finite_number("immediate_cash_flow", immediate_cash_flow)
    if discount_rate_per_round is not None:
        discount_rate_per_round = _finite_number(
            "discount_rate_per_round", discount_rate_per_round
        )
        if discount_rate_per_round <= -1:
            raise ValueError("discount_rate_per_round deve ser maior que -1.")
    _validate_scenario(loop, config, player_id, property_changes)

    traffic = stationary or calculate_stationary_board_analysis(config, loop.rules)
    if traffic.positions != tuple(range(loop.game.board.size)):
        raise ValueError("A análise estacionária não corresponde ao tabuleiro desta partida.")

    active_opponents = sum(
        not loop.game.player_states[other_id].is_bankrupt
        for other_id in loop.game.turn_order
        if other_id != player_id
    )
    current_income = _expected_rent_income(
        loop,
        traffic,
        player_id,
        active_opponents,
        {},
    )
    projected_income = _expected_rent_income(
        loop,
        traffic,
        player_id,
        active_opponents,
        property_changes,
    )
    incremental_income = projected_income - current_income
    cash_outlay = max(0.0, -immediate_cash_flow)

    horizon_values_result = []
    for rounds in horizon_values:
        operating_cash_flow = incremental_income * rounds
        net_cash_flow = immediate_cash_flow + operating_cash_flow
        roi = net_cash_flow / cash_outlay if cash_outlay > 0 else None
        npv = None
        if discount_rate_per_round is not None:
            discounted_flow = sum(
                incremental_income / (1 + discount_rate_per_round) ** period
                for period in range(1, rounds + 1)
            )
            npv = immediate_cash_flow + discounted_flow
        horizon_values_result.append(
            PortfolioHorizonValue(
                horizon_rounds=rounds,
                incremental_operating_cash_flow=float(operating_cash_flow),
                immediate_cash_flow=immediate_cash_flow,
                expected_net_cash_flow=float(net_cash_flow),
                roi_on_cash_outlay=(float(roi) if roi is not None else None),
                net_present_value=(float(npv) if npv is not None else None),
            )
        )

    return PortfolioScenarioValuation(
        player_id=player_id,
        active_opponents=active_opponents,
        immediate_cash_flow=immediate_cash_flow,
        current_rent_income_per_round=current_income,
        projected_rent_income_per_round=projected_income,
        incremental_rent_income_per_round=incremental_income,
        horizons=tuple(horizon_values_result),
    )


def value_property_acquisition(
    loop: GameLoop,
    config: GameConfig,
    buyer_id: UUID,
    tile_id: int,
    *,
    acquisition_price: float | None = None,
    horizons: Iterable[int] = DEFAULT_VALUE_HORIZONS,
    discount_rate_per_round: float | None = None,
    stationary: StationaryBoardAnalysis | None = None,
) -> PropertyAcquisitionValuation:
    """Estimate the long-run incremental value of buying an unowned property."""
    tile = loop.game.board.get_tile_by_id(tile_id)
    if not isinstance(tile, PropertyTile):
        raise ValueError(f"A casa {tile_id} não é uma propriedade comprável.")
    current_state = loop.game.property_states[tile_id]
    if current_state.owner_id is not None:
        raise ValueError("A aquisição só pode avaliar uma propriedade sem dono.")
    traffic = stationary or calculate_stationary_board_analysis(config, loop.rules)
    price = tile.price if acquisition_price is None else _finite_number(
        "acquisition_price", acquisition_price
    )
    if price < 0:
        raise ValueError("acquisition_price não pode ser negativo.")

    changes = {
        tile_id: PropertyScenarioState(
            owner_id=buyer_id,
            houses=current_state.houses,
            is_mortgaged=False,
        )
    }
    portfolio_impact = analyze_portfolio_scenario(
        loop,
        config,
        buyer_id,
        changes,
        immediate_cash_flow=-price,
        horizons=horizons,
        discount_rate_per_round=discount_rate_per_round,
        stationary=traffic,
    )
    group_tiles = loop.game.board.group_members(tile.group)
    owned_before = sum(
        loop.game.property_states[member.tile_id].owner_id == buyer_id
        for member in group_tiles
    )
    owned_after = owned_before + 1
    current_group_income = _expected_rent_income(
        loop,
        traffic,
        buyer_id,
        portfolio_impact.active_opponents,
        {},
        group=tile.group,
    )
    projected_group_income = _expected_rent_income(
        loop,
        traffic,
        buyer_id,
        portfolio_impact.active_opponents,
        changes,
        group=tile.group,
    )

    return PropertyAcquisitionValuation(
        tile_id=tile_id,
        name=tile.name,
        group=tile.group,
        acquisition_price=price,
        owned_group_properties_before=owned_before,
        owned_group_properties_after=owned_after,
        group_size=len(group_tiles),
        completes_group=owned_after == len(group_tiles),
        current_group_rent_income_per_round=current_group_income,
        projected_group_rent_income_per_round=projected_group_income,
        portfolio_impact=portfolio_impact,
    )


def analyze_player_group(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    group: str,
    *,
    horizons: Iterable[int] = DEFAULT_VALUE_HORIZONS,
    discount_rate_per_round: float | None = None,
    stationary: StationaryBoardAnalysis | None = None,
) -> PropertyGroupAnalysis:
    """Summarize current group income and value each remaining bank property."""
    group_tiles = loop.game.board.group_members(group)
    traffic = stationary or calculate_stationary_board_analysis(config, loop.rules)
    active_opponents = sum(
        not loop.game.player_states[other_id].is_bankrupt
        for other_id in loop.game.turn_order
        if other_id != player_id
    )
    owned_count = sum(
        loop.game.property_states[tile.tile_id].owner_id == player_id
        for tile in group_tiles
    )
    current_income = _expected_rent_income(
        loop,
        traffic,
        player_id,
        active_opponents,
        {},
        group=group,
    )
    acquisitions = tuple(
        value_property_acquisition(
            loop,
            config,
            player_id,
            tile.tile_id,
            horizons=horizons,
            discount_rate_per_round=discount_rate_per_round,
            stationary=traffic,
        )
        for tile in group_tiles
        if loop.game.property_states[tile.tile_id].owner_id is None
    )
    return PropertyGroupAnalysis(
        player_id=player_id,
        group=group,
        group_size=len(group_tiles),
        owned_group_properties=owned_count,
        expected_group_rent_income_per_round=current_income,
        available_acquisitions=acquisitions,
    )


def _expected_rent_income(
    loop: GameLoop,
    traffic: StationaryBoardAnalysis,
    player_id: UUID,
    active_opponents: int,
    changes: Mapping[int, PropertyScenarioState],
    *,
    group: str | None = None,
) -> float:
    owner_by_tile = {
        tile.tile_id: changes[tile.tile_id].owner_id
        if tile.tile_id in changes
        else loop.game.property_states[tile.tile_id].owner_id
        for tile in loop.game.board.property_tiles
    }
    houses_by_tile = {
        tile.tile_id: changes[tile.tile_id].houses
        if tile.tile_id in changes
        else loop.game.property_states[tile.tile_id].houses
        for tile in loop.game.board.property_tiles
    }
    mortgaged_by_tile = {
        tile.tile_id: changes[tile.tile_id].is_mortgaged
        if tile.tile_id in changes
        else loop.game.property_states[tile.tile_id].is_mortgaged
        for tile in loop.game.board.property_tiles
    }
    owned_counts: dict[tuple[UUID, str], int] = {}
    for tile in loop.game.board.property_tiles:
        owner = owner_by_tile[tile.tile_id]
        if owner is not None:
            key = owner, tile.group
            owned_counts[key] = owned_counts.get(key, 0) + 1

    expected_income = 0.0
    for tile in loop.game.board.property_tiles:
        if owner_by_tile[tile.tile_id] != player_id or mortgaged_by_tile[tile.tile_id]:
            continue
        if group is not None and tile.group != group:
            continue
        rent_events = traffic.rent_events_for(tile.tile_id)
        if not rent_events:
            continue
        rent_per_event = 0.0
        group_count = owned_counts.get((player_id, tile.group), 0)
        for event in rent_events:
            if tile.kind == PropertyKind.STREET:
                level = houses_by_tile[tile.tile_id]
                if level:
                    rent = tile.rent_for_level(level)
                elif group_count == len(loop.game.board.group_members(tile.group)):
                    rent = (
                        tile.monopoly_rent
                        if tile.monopoly_rent is not None
                        else 2 * tile.rent_values[0]
                    )
                else:
                    rent = tile.rent_values[0]
            elif tile.kind == PropertyKind.UTILITY:
                rent = tile.rent_values[min(group_count, len(tile.rent_values)) - 1]
                rent *= event.rent_dice_total
            else:
                rent = tile.rent_values[min(group_count, len(tile.rent_values)) - 1]
            rent_per_event += (
                event.expected_arrivals_per_turn
                * rent
                * event.rent_multiplier
            )
        expected_income += active_opponents * rent_per_event
    return float(expected_income)


def _validate_scenario(
    loop: GameLoop,
    config: GameConfig,
    player_id: UUID,
    property_changes: Mapping[int, PropertyScenarioState],
) -> None:
    if player_id not in loop.game.player_states:
        raise KeyError(f"Jogador desconhecido: {player_id!r}")
    if loop.game.player_states[player_id].is_bankrupt:
        raise ValueError("Não é possível avaliar o portfólio de um jogador falido.")
    if not isinstance(property_changes, Mapping):
        raise TypeError("property_changes deve ser um mapeamento de tile_id para estado.")

    config_property_ids = {
        space.position for space in config.board if space.type in {
            "property", "railroad", "utility"
        }
    }
    for tile_id, state in property_changes.items():
        if tile_id not in config_property_ids or tile_id not in loop.game.property_states:
            raise KeyError(f"tile_id não é uma propriedade deste tabuleiro: {tile_id!r}")
        if not isinstance(state, PropertyScenarioState):
            raise TypeError("Cada alteração deve ser PropertyScenarioState.")
        if state.owner_id is not None and state.owner_id not in loop.game.player_states:
            raise KeyError(f"Dono desconhecido: {state.owner_id!r}")
        if state.owner_id is not None and loop.game.player_states[state.owner_id].is_bankrupt:
            raise ValueError("Uma propriedade não pode ser atribuída a um jogador falido.")
        if isinstance(state.houses, bool) or not isinstance(state.houses, int):
            raise TypeError("houses deve ser um inteiro.")
        if not 0 <= state.houses <= HOTEL_LEVEL:
            raise ValueError("houses deve estar entre 0 e 5.")
        if not isinstance(state.is_mortgaged, bool):
            raise TypeError("is_mortgaged deve ser booleano.")
        if state.is_mortgaged and state.houses:
            raise ValueError("Uma propriedade hipotecada não pode ter construções.")
        if state.owner_id is None and (state.houses or state.is_mortgaged):
            raise ValueError("Uma propriedade do banco não pode ter construções nem hipoteca.")


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(horizons)
    except TypeError as exc:
        raise TypeError("horizons deve ser uma coleção de inteiros positivos.") from exc
    if not values:
        raise ValueError("Informe ao menos um horizonte.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("Cada horizonte deve ser um inteiro.")
    if any(value <= 0 for value in values):
        raise ValueError("Os horizontes devem ser positivos.")
    if len(set(values)) != len(values):
        raise ValueError("Os horizontes não podem se repetir.")
    return tuple(sorted(values))


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} deve ser finito.")
    return result
