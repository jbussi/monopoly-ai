"""Stationary board traffic and comparable property portfolio valuations.

The movement chain advances once per dice roll. Card draws are approximated as
independent, uniform draws with replacement; this keeps the state space finite
without encoding each deck permutation. Jail policy is to attempt doubles and
pay on the final allowed attempt, matching the engine's automatic roll path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from collections.abc import Mapping

import numpy as np

from app.domain.analytics.markov import (
    StateSpace,
    TransitionMatrix,
    TransitionModel,
    calculate_stationary_distribution,
)
from app.infraestructure.game_config.models import (
    BoardSpace,
    GameConfig,
    PropertyDefinition,
    RailroadDefinition,
    UtilityDefinition,
)


_STATIONARY_ANALYSIS_CACHE: dict[
    tuple[str, tuple[int, int]], "StationaryBoardAnalysis"
] = {}


@dataclass(frozen=True, slots=True)
class BoardMovementState:
    position: int
    doubles_streak: int = 0
    jail_attempts: int = 0
    in_jail: bool = False


@dataclass(frozen=True, slots=True)
class _Outcome:
    state: BoardMovementState
    probability: float
    landed_property_id: int | None = None
    utility_dice_value: float = 0.0
    rent_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class PropertyTraffic:
    tile_id: int
    name: str
    group: str
    kind: str
    arrivals_per_roll: float
    utility_dice_total_per_roll: float
    railroad_rent_multiplier_per_roll: float


@dataclass(frozen=True, slots=True)
class StationaryBoardAnalysis:
    variant: str
    state_count: int
    probabilities: tuple[float, ...]
    states: tuple[BoardMovementState, ...]
    position_probabilities: tuple[float, ...]
    expected_rolls_per_turn: float
    property_traffic: tuple[PropertyTraffic, ...]
    assumptions: tuple[str, ...]

    def traffic_for(self, tile_id: int) -> PropertyTraffic:
        """Return modeled landing and rent traffic for one board tile."""
        for traffic in self.property_traffic:
            if traffic.tile_id == tile_id:
                return traffic
        raise KeyError(f"Propriedade sem taxa de chegada: {tile_id!r}")


@dataclass(frozen=True, slots=True)
class GroupPortfolioScenario:
    group: str
    owned_tile_ids: tuple[int, ...]
    owned_count: int
    development_level: int
    investment_cost: float
    gross_income_per_turn: float
    gross_yield_per_turn: float | None
    horizon_turns: int | None
    expected_gross_income: float | None
    expected_net_profit: float | None
    roi: float | None


@dataclass(frozen=True, slots=True)
class MarginalPurchaseValuation:
    group: str
    candidate_tile_id: int
    purchase_price: float
    owned_before: tuple[int, ...]
    owned_after: tuple[int, ...]
    income_before_per_turn: float
    income_after_per_turn: float
    marginal_income_per_turn: float
    marginal_yield_per_turn: float | None
    horizon_turns: int | None
    expected_marginal_income: float | None
    expected_net_profit: float | None
    roi: float | None


class _ConfiguredBoardMovementModel(TransitionModel[BoardMovementState]):
    def __init__(self, config: GameConfig, dice_sides: tuple[int, int]):
        self.config = config
        self.board = tuple(sorted(config.board, key=lambda item: item.position))
        self.board_size = len(self.board)
        self.by_position = {space.position: space for space in self.board}
        self.rules = config.settings
        self.dice_sides = dice_sides
        self.max_doubles = 3
        self.max_jail_attempts = config.settings.jail_max_turns
        if self.max_jail_attempts < 1:
            raise ValueError("jail_max_turns deve ser maior que zero.")
        if self.board_size < 2 or set(self.by_position) != set(range(self.board_size)):
            raise ValueError("O tabuleiro deve ter posições consecutivas começando em zero.")
        if not 0 <= self.rules.jail_position < self.board_size:
            raise ValueError("jail_position fora do tabuleiro.")
        if not 0 <= self.rules.go_to_jail_position < self.board_size:
            raise ValueError("go_to_jail_position fora do tabuleiro.")

        states = [
            BoardMovementState(position=position, doubles_streak=streak)
            for position in range(self.board_size)
            for streak in range(self.max_doubles)
        ]
        states.extend(
            BoardMovementState(
                position=self.rules.jail_position,
                jail_attempts=attempt,
                in_jail=True,
            )
            for attempt in range(self.max_jail_attempts)
        )
        super().__init__(StateSpace(states))
        self._transition_cache: dict[BoardMovementState, tuple[_Outcome, ...]] = {}
        self.decks = {
            "chance": tuple(config.chance_cards),
            "community_chest": tuple(config.community_chest_cards),
        }
        self._validate_card_destinations()
        self.property_by_position = {
            item.position: item
            for item in self.board
            if isinstance(item, (PropertyDefinition, RailroadDefinition, UtilityDefinition))
        }
        self.mean_dice_total = (dice_sides[0] + 1) / 2 + (dice_sides[1] + 1) / 2

    def transitions_from(self, source: BoardMovementState):
        """Return next movement states and their aggregated probabilities."""
        outcomes = self._outcomes(source)
        probabilities: dict[BoardMovementState, float] = defaultdict(float)
        for outcome in outcomes:
            probabilities[outcome.state] += outcome.probability
        return tuple(probabilities.items())

    def probability(self, source: BoardMovementState, target: BoardMovementState) -> float:
        """Return the probability of transitioning between two board states."""
        return dict(self.transitions_from(source)).get(target, 0.0)

    def outcomes_from(self, source: BoardMovementState) -> tuple[_Outcome, ...]:
        """Return detailed dice and card outcomes for a movement state."""
        return self._outcomes(source)

    def _outcomes(self, source: BoardMovementState) -> tuple[_Outcome, ...]:
        cached = self._transition_cache.get(source)
        if cached is not None:
            return cached

        d1_sides, d2_sides = self.dice_sides
        outcome_mass: dict[tuple[BoardMovementState, int | None, float, float], float] = defaultdict(float)
        roll_probability = 1.0 / (d1_sides * d2_sides)
        for die1 in range(1, d1_sides + 1):
            for die2 in range(1, d2_sides + 1):
                total = die1 + die2
                is_double = die1 == die2
                if source.in_jail:
                    next_attempt = source.jail_attempts + 1
                    if is_double or next_attempt >= self.max_jail_attempts:
                        position = (self.rules.jail_position + total) % self.board_size
                        resolved = self._resolve_position(position, 0, float(total), 1.0, 0)
                    else:
                        resolved = (_Outcome(
                            BoardMovementState(
                                position=self.rules.jail_position,
                                jail_attempts=next_attempt,
                                in_jail=True,
                            ),
                            1.0,
                        ),)
                else:
                    streak = source.doubles_streak + 1 if is_double else 0
                    if is_double and streak >= self.max_doubles:
                        resolved = (_Outcome(
                            BoardMovementState(
                                position=self.rules.jail_position,
                                in_jail=True,
                            ),
                            1.0,
                        ),)
                    else:
                        position = (source.position + total) % self.board_size
                        resolved = self._resolve_position(position, streak, float(total), 1.0, 0)

                for outcome in resolved:
                    key = (
                        outcome.state, outcome.landed_property_id,
                        outcome.utility_dice_value, outcome.rent_multiplier,
                    )
                    outcome_mass[key] += roll_probability * outcome.probability

        outcomes = tuple(
            _Outcome(state, probability, tile_id, utility_value, rent_multiplier)
            for (state, tile_id, utility_value, rent_multiplier), probability in outcome_mass.items()
        )
        self._transition_cache[source] = outcomes
        return outcomes

    def _resolve_position(
        self,
        position: int,
        doubles_streak: int,
        utility_dice_value: float,
        rent_multiplier: float,
        depth: int,
    ) -> tuple[_Outcome, ...]:
        if depth > self.board_size:
            raise ValueError("Ciclo de movimento entre cartas excede o tamanho do tabuleiro.")
        if position == self.rules.go_to_jail_position:
            return (_Outcome(
                BoardMovementState(position=self.rules.jail_position, in_jail=True),
                1.0,
            ),)

        property_tile = self.property_by_position.get(position)
        if property_tile is not None:
            utility_value = utility_dice_value if isinstance(property_tile, UtilityDefinition) else 0.0
            return (_Outcome(
                BoardMovementState(position=position, doubles_streak=doubles_streak),
                1.0,
                property_tile.position,
                utility_value,
                rent_multiplier if isinstance(property_tile, RailroadDefinition) else 1.0,
            ),)

        space = self.by_position[position]
        cards = self.decks.get(space.type, ())
        if not cards:
            return (_Outcome(
                BoardMovementState(position=position, doubles_streak=doubles_streak),
                1.0,
            ),)

        outcomes: list[_Outcome] = []
        card_probability = 1.0 / len(cards)
        for card in cards:
            action = card.action
            data = card.data
            if action == "go_to_jail":
                card_outcomes = (_Outcome(
                    BoardMovementState(position=self.rules.jail_position, in_jail=True), 1.0
                ),)
            elif action == "advance_to":
                target = int(data["position"])
                card_outcomes = self._resolve_position(target, doubles_streak, utility_dice_value, 1.0, depth + 1)
            elif action == "go_back":
                target = (position - int(data["spaces"])) % self.board_size
                card_outcomes = self._resolve_position(target, doubles_streak, utility_dice_value, 1.0, depth + 1)
            elif action == "advance_to_nearest":
                target = self._nearest_property(position, str(data["type"]))
                if data["type"] == "utility":
                    utility_value = self.mean_dice_total * float(data.get("rentMultiplier", 1))
                    card_multiplier = 1.0
                else:
                    utility_value = utility_dice_value
                    card_multiplier = float(data.get("rentMultiplier", 1))
                card_outcomes = self._resolve_position(
                    target, doubles_streak, utility_value, card_multiplier, depth + 1
                )
            else:
                # Financial effects and get-out-of-jail cards do not change position.
                card_outcomes = (_Outcome(
                    BoardMovementState(position=position, doubles_streak=doubles_streak), 1.0
                ),)
            outcomes.extend(
                _Outcome(
                    item.state,
                    item.probability * card_probability,
                    item.landed_property_id,
                    item.utility_dice_value,
                    item.rent_multiplier,
                )
                for item in card_outcomes
            )
        return tuple(outcomes)

    def _nearest_property(self, position: int, kind: str) -> int:
        for distance in range(1, self.board_size + 1):
            candidate = (position + distance) % self.board_size
            space = self.by_position[candidate]
            if kind == "railroad" and isinstance(space, RailroadDefinition):
                return candidate
            if kind == "utility" and isinstance(space, UtilityDefinition):
                return candidate
        raise ValueError(f"Não há imóvel do tipo {kind!r} no tabuleiro.")

    def _validate_card_destinations(self) -> None:
        for deck in self.decks.values():
            for card in deck:
                if card.action == "advance_to":
                    target = card.data.get("position")
                    if not isinstance(target, int) or isinstance(target, bool) or target not in self.by_position:
                        raise ValueError(f"Destino inválido na carta {card.id!r}: {target!r}.")
                elif card.action == "advance_to_nearest":
                    if card.data.get("type") not in ("railroad", "utility"):
                        raise ValueError(f"Tipo de destino inválido na carta {card.id!r}.")
                    self._nearest_property(0, str(card.data["type"]))
                elif card.action == "go_back":
                    spaces = card.data.get("spaces")
                    if not isinstance(spaces, int) or isinstance(spaces, bool) or spaces < 0:
                        raise ValueError(f"Distância inválida na carta {card.id!r}.")


def calculate_stationary_board_analysis(
    config: GameConfig,
    dice_sides: int | tuple[int, int] = 6,
) -> StationaryBoardAnalysis:
    """Calculate long-run board occupancy and property arrivals per dice roll."""
    if isinstance(dice_sides, int) and not isinstance(dice_sides, bool):
        dice_sides = (dice_sides, dice_sides)
    if (
        not isinstance(dice_sides, tuple)
        or len(dice_sides) != 2
        or any(not isinstance(side, int) or isinstance(side, bool) or side < 2 for side in dice_sides)
    ):
        raise ValueError("dice_sides deve ser um inteiro >=2 ou uma tupla de dois inteiros >=2.")

    cache_key = (repr(config), dice_sides)
    cached = _STATIONARY_ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = _ConfiguredBoardMovementModel(config, dice_sides)
    matrix = TransitionMatrix(model)
    stationary = calculate_stationary_distribution(matrix)

    position_probabilities = np.zeros(model.board_size, dtype=np.float64)
    arrivals: dict[int, float] = defaultdict(float)
    utility_dice_totals: dict[int, float] = defaultdict(float)
    railroad_multipliers: dict[int, float] = defaultdict(float)
    turn_start_probability = 0.0

    for index, state in enumerate(model.state_space.states):
        state_probability = float(stationary[index])
        position_probabilities[state.position] += state_probability
        if state.in_jail or state.doubles_streak == 0:
            turn_start_probability += state_probability
        for outcome in model.outcomes_from(state):
            mass = state_probability * outcome.probability
            if outcome.landed_property_id is not None:
                arrivals[outcome.landed_property_id] += mass
                utility_dice_totals[outcome.landed_property_id] += mass * outcome.utility_dice_value
                railroad_multipliers[outcome.landed_property_id] += mass * outcome.rent_multiplier

    expected_rolls_per_turn = 1.0 / turn_start_probability
    properties = [
        space for space in model.board
        if isinstance(space, (PropertyDefinition, RailroadDefinition, UtilityDefinition))
    ]
    traffic = tuple(
        PropertyTraffic(
            tile_id=space.position,
            name=space.name,
            group=_group_for(space),
            kind=space.type,
            arrivals_per_roll=float(arrivals[space.position]),
            utility_dice_total_per_roll=float(utility_dice_totals[space.position]),
            railroad_rent_multiplier_per_roll=float(railroad_multipliers[space.position]),
        )
        for space in properties
    )
    result = StationaryBoardAnalysis(
        variant=config.variant,
        state_count=model.state_space.size,
        probabilities=tuple(float(value) for value in stationary),
        states=model.state_space.states,
        position_probabilities=tuple(float(value) for value in position_probabilities),
        expected_rolls_per_turn=float(expected_rolls_per_turn),
        property_traffic=traffic,
        assumptions=(
            "Um passo da cadeia representa um lançamento de dados; as taxas de chegada são por lançamento.",
            "Cartas são amostradas uniformemente com reposição; efeitos financeiros não alteram posição.",
            "O jogador tenta dados duplos na prisão e se move após dupla ou na tentativa final.",
            "Dados são justos e independentes; o modelo não depende de posições ou ativos dos adversários.",
        ),
    )
    _STATIONARY_ANALYSIS_CACHE[cache_key] = result
    return result


def evaluate_group_portfolios(
    config: GameConfig,
    analysis: StationaryBoardAnalysis,
    *,
    number_of_opponents: int = 1,
    horizon_turns: int | None = None,
) -> tuple[GroupPortfolioScenario, ...]:
    """Evaluate all concrete ownership subsets and full street build levels.

    Income is gross expected rent per owner turn, based on the requested
    number of active opponents. No operating costs,
    ownership changes, liquidity effects, or terminal resale value are assumed.
    """
    _validate_horizon(number_of_opponents, horizon_turns)
    groups: dict[str, list[BoardSpace]] = defaultdict(list)
    for space in config.board:
        if isinstance(space, (PropertyDefinition, RailroadDefinition, UtilityDefinition)):
            groups[_group_for(space)].append(space)

    scenarios: list[GroupPortfolioScenario] = []
    for group, properties in sorted(groups.items()):
        properties.sort(key=lambda item: item.position)
        full_ids = tuple(item.position for item in properties)
        for count in range(1, len(properties) + 1):
            for subset in combinations(properties, count):
                ids = tuple(item.position for item in subset)
                scenarios.append(_portfolio_scenario(
                    group, properties, ids, 0, analysis,
                    number_of_opponents, horizon_turns,
                ))
        if all(isinstance(item, PropertyDefinition) for item in properties):
            for level in range(1, 6):
                scenarios.append(_portfolio_scenario(
                    group, properties, full_ids, level, analysis,
                    number_of_opponents, horizon_turns,
                ))
    return tuple(scenarios)


def evaluate_marginal_purchase(
    config: GameConfig,
    analysis: StationaryBoardAnalysis,
    candidate_tile_id: int,
    currently_owned_tile_ids: tuple[int, ...] | list[int],
    *,
    price: float | None = None,
    mortgaged_tile_ids: tuple[int, ...] | list[int] = (),
    number_of_opponents: int = 1,
    horizon_turns: int | None = None,
) -> MarginalPurchaseValuation:
    """Compare a group's expected rent before and after buying one property.

    ``price`` can be the listed purchase price or a candidate auction bid.
    """
    _validate_horizon(number_of_opponents, horizon_turns)
    definitions = {
        item.position: item for item in config.board
        if isinstance(item, (PropertyDefinition, RailroadDefinition, UtilityDefinition))
    }
    if candidate_tile_id not in definitions:
        raise ValueError(f"tile_id={candidate_tile_id} não é uma propriedade.")
    candidate = definitions[candidate_tile_id]
    group = _group_for(candidate)
    group_properties = sorted(
        (item for item in definitions.values() if _group_for(item) == group),
        key=lambda item: item.position,
    )
    owned_before = tuple(sorted(set(currently_owned_tile_ids)))
    group_ids = {item.position for item in group_properties}
    if not set(owned_before).issubset(group_ids):
        raise ValueError("currently_owned_tile_ids deve conter apenas imóveis do grupo candidato.")
    if candidate_tile_id in owned_before:
        raise ValueError("O imóvel candidato já consta entre os imóveis possuídos.")
    mortgaged = set(mortgaged_tile_ids)
    if not mortgaged.issubset(set(owned_before)):
        raise ValueError("mortgaged_tile_ids deve ser subconjunto dos imóveis já possuídos no grupo.")

    investment = float(candidate.price if price is None else price)
    if not isfinite(investment) or investment <= 0:
        raise ValueError("price deve ser um número finito positivo.")
    owned_after = tuple(sorted((*owned_before, candidate_tile_id)))
    income_before = _portfolio_income(group_properties, owned_before, 0, analysis, mortgaged)
    income_after = _portfolio_income(group_properties, owned_after, 0, analysis, mortgaged)
    marginal_income = (income_after - income_before) * number_of_opponents * analysis.expected_rolls_per_turn
    marginal_yield = marginal_income / investment
    expected_income = marginal_income * horizon_turns if horizon_turns is not None else None
    net_profit = expected_income - investment if expected_income is not None else None
    roi = net_profit / investment if net_profit is not None else None
    return MarginalPurchaseValuation(
        group=group,
        candidate_tile_id=candidate_tile_id,
        purchase_price=investment,
        owned_before=owned_before,
        owned_after=owned_after,
        income_before_per_turn=(
            income_before * number_of_opponents * analysis.expected_rolls_per_turn
        ),
        income_after_per_turn=(
            income_after * number_of_opponents * analysis.expected_rolls_per_turn
        ),
        marginal_income_per_turn=marginal_income,
        marginal_yield_per_turn=marginal_yield,
        horizon_turns=horizon_turns,
        expected_marginal_income=expected_income,
        expected_net_profit=net_profit,
        roi=roi,
    )


def estimate_group_state_income_per_turn(
    config: GameConfig,
    analysis: StationaryBoardAnalysis,
    owned_tile_ids: tuple[int, ...] | list[int],
    building_levels: Mapping[int, int],
    mortgaged_tile_ids: tuple[int, ...] | list[int],
    *,
    number_of_opponents: int,
) -> float:
    """Estimate gross rent per turn for one player's current group state."""
    _validate_horizon(number_of_opponents, None)
    definitions = {
        item.position: item for item in config.board
        if isinstance(item, (PropertyDefinition, RailroadDefinition, UtilityDefinition))
    }
    owned = set(owned_tile_ids)
    mortgaged = set(mortgaged_tile_ids)
    if not mortgaged.issubset(owned):
        raise ValueError("Hipotecas devem pertencer a imóveis possuídos.")
    if not owned:
        return 0.0
    candidate = next(iter(owned))
    if candidate not in definitions:
        raise ValueError(f"Imóvel desconhecido: {candidate!r}.")
    group = _group_for(definitions[candidate])
    group_properties = [
        item for item in definitions.values()
        if _group_for(item) == group
    ]
    if not owned.issubset({item.position for item in group_properties}):
        raise ValueError("A carteira deve conter imóveis de um único grupo.")
    complete_group = owned == {item.position for item in group_properties}
    owned_count = len(owned)
    income_per_roll = 0.0
    for item in group_properties:
        if item.position not in owned or item.position in mortgaged:
            continue
        traffic = analysis.traffic_for(item.position)
        if isinstance(item, PropertyDefinition):
            level = building_levels.get(item.position, 0)
            if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 5:
                raise ValueError("Nível de construção deve ser um inteiro entre 0 e 5.")
            rent = (
                item.rent.monopoly
                if complete_group and level == 0
                else item.rent.base if level == 0
                else (
                    item.rent.one_house,
                    item.rent.two_houses,
                    item.rent.three_houses,
                    item.rent.four_houses,
                    item.rent.hotel,
                )[level - 1]
            )
            income_per_roll += traffic.arrivals_per_roll * rent
        elif isinstance(item, RailroadDefinition):
            rent = float(item.rent.get(f"owned{owned_count}", 0))
            income_per_roll += traffic.railroad_rent_multiplier_per_roll * rent
        elif isinstance(item, UtilityDefinition):
            multiplier = float(item.rent.get(f"multiplier{owned_count}", 0))
            income_per_roll += traffic.utility_dice_total_per_roll * multiplier
    return income_per_roll * number_of_opponents * analysis.expected_rolls_per_turn


def _portfolio_scenario(group, properties, owned_ids, level, analysis, opponents, horizon):
    income_per_roll = _portfolio_income(properties, owned_ids, level, analysis)
    income_per_turn = income_per_roll * opponents * analysis.expected_rolls_per_turn
    investment = sum(float(item.price) for item in properties if item.position in owned_ids)
    if level:
        investment += sum(
            float(item.house_cost) * min(level, 4)
            + (float(item.hotel_cost or item.house_cost) if level == 5 else 0.0)
            for item in properties if item.position in owned_ids
        )
    yield_per_turn = income_per_turn / investment if investment > 0 else None
    gross_income = income_per_turn * horizon if horizon is not None else None
    net_profit = gross_income - investment if gross_income is not None else None
    roi = net_profit / investment if net_profit is not None and investment > 0 else None
    return GroupPortfolioScenario(
        group=group,
        owned_tile_ids=tuple(owned_ids),
        owned_count=len(owned_ids),
        development_level=level,
        investment_cost=investment,
        gross_income_per_turn=income_per_turn,
        gross_yield_per_turn=yield_per_turn,
        horizon_turns=horizon,
        expected_gross_income=gross_income,
        expected_net_profit=net_profit,
        roi=roi,
    )


def _portfolio_income(properties, owned_ids, level, analysis, mortgaged_ids=()):
    owned = set(owned_ids)
    if not owned:
        return 0.0
    all_ids = {item.position for item in properties}
    complete_group = owned == all_ids
    group_count = len(owned)
    income = 0.0
    for item in properties:
        if item.position not in owned or item.position in mortgaged_ids:
            continue
        traffic = analysis.traffic_for(item.position)
        if isinstance(item, PropertyDefinition):
            if level == 0:
                rent = item.rent.monopoly if complete_group else item.rent.base
            else:
                rent = (
                    item.rent.one_house,
                    item.rent.two_houses,
                    item.rent.three_houses,
                    item.rent.four_houses,
                    item.rent.hotel,
                )[level - 1]
            income += traffic.arrivals_per_roll * rent
        elif isinstance(item, RailroadDefinition):
            rent = float(item.rent.get(f"owned{group_count}", 0))
            income += traffic.railroad_rent_multiplier_per_roll * rent
        elif isinstance(item, UtilityDefinition):
            multiplier = float(item.rent.get(f"multiplier{group_count}", 0))
            income += traffic.utility_dice_total_per_roll * multiplier
    return income


def _group_for(space: BoardSpace) -> str:
    if isinstance(space, PropertyDefinition):
        return space.color_group
    if isinstance(space, RailroadDefinition):
        return "railroad"
    if isinstance(space, UtilityDefinition):
        return "utility"
    raise TypeError(f"Espaço não é propriedade: {space!r}")


def _validate_horizon(number_of_opponents: int, horizon_turns: int | None) -> None:
    if not isinstance(number_of_opponents, int) or isinstance(number_of_opponents, bool) or number_of_opponents < 0:
        raise ValueError("number_of_opponents deve ser um inteiro não negativo.")
    if horizon_turns is not None and (
        not isinstance(horizon_turns, int) or isinstance(horizon_turns, bool) or horizon_turns < 1
    ):
        raise ValueError("horizon_turns deve ser None ou um inteiro positivo.")
