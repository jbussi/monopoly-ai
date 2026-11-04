"""Long-run landing frequencies for a configured Monopoly board.

Dice outcomes and configured movement cards are modeled explicitly. Card draws
are independent uniform draws (the deck order and held jail cards are not part
of the Markov state), and jail follows the roll-doubles / forced-release path
used by the original board-analysis notebook. This is a movement model, not a
full economic or player-policy model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.domain.game.rules import Rules
from app.infraestructure.game_config.models import GameConfig

from .state_space import StateSpace
from .stationary import calculate_stationary_distribution
from .transition_matrix import TransitionMatrix
from .transition_model import TransitionModel


@dataclass(frozen=True, slots=True)
class MonopolyMovementState:
    """Movement state observed immediately before a dice roll."""

    position: int
    in_jail: bool = False
    jail_attempts: int = 0
    doubles_streak: int = 0


@dataclass(frozen=True, slots=True)
class MonopolyMovementOutcome:
    """One possible state transition and the landing/rent event it creates."""

    state: MonopolyMovementState
    probability: float
    landing_position: int | None
    rent_dice_total: float
    rent_multiplier: float = 1.0
    turn_ended: bool = True


@dataclass(frozen=True, slots=True)
class LandingEventFrequency:
    """Long-run rate for one type of arrival at a board position."""

    tile_id: int
    rent_dice_total: float
    rent_multiplier: float
    probability_per_roll: float
    expected_arrivals_per_turn: float


@dataclass(frozen=True, slots=True)
class StationaryBoardAnalysis:
    """Stationary states and board landing frequencies under the model."""

    positions: tuple[int, ...]
    stationary_states: tuple[MonopolyMovementState, ...]
    stationary_state_probabilities: tuple[float, ...]
    landing_probabilities_per_roll: tuple[float, ...]
    expected_landings_per_turn: tuple[float, ...]
    landing_events_per_turn: tuple[LandingEventFrequency, ...]
    turn_end_probability_per_roll: float
    expected_rolls_per_turn: float

    def probability_per_roll(self, tile_id: int) -> float:
        """Return the long-run probability of landing on a tile per dice roll."""
        try:
            index = self.positions.index(tile_id)
        except ValueError as exc:
            raise KeyError(f"Casa desconhecida: {tile_id}.") from exc
        return self.landing_probabilities_per_roll[index]

    def arrivals_per_turn(self, tile_id: int) -> float:
        """Return expected landings on a tile per complete table turn."""
        try:
            index = self.positions.index(tile_id)
        except ValueError as exc:
            raise KeyError(f"Casa desconhecida: {tile_id}.") from exc
        return self.expected_landings_per_turn[index]

    def rent_events_for(self, tile_id: int) -> tuple[LandingEventFrequency, ...]:
        """List rent-relevant landing frequencies for a property tile."""
        return tuple(event for event in self.landing_events_per_turn if event.tile_id == tile_id)


class MonopolyBoardModel(TransitionModel[MonopolyMovementState]):
    """Markov movement model built from the selected board and its card decks."""

    def __init__(
        self,
        config: GameConfig,
        rules: Rules | None = None,
        *,
        dice_sides: tuple[int, int] = (6, 6),
        max_card_chain: int = 4,
    ) -> None:
        if not config.board:
            raise ValueError("A configuração precisa conter um tabuleiro.")
        if (
            len(dice_sides) != 2
            or any(isinstance(side, bool) or not isinstance(side, int) or side < 2 for side in dice_sides)
        ):
            raise ValueError("dice_sides deve conter duas quantidades de faces >= 2.")
        if isinstance(max_card_chain, bool) or not isinstance(max_card_chain, int) or max_card_chain <= 0:
            raise ValueError("max_card_chain deve ser um inteiro positivo.")

        self.config = config
        self.board_size = len(config.board)
        self.positions = tuple(sorted(space.position for space in config.board))
        if self.positions != tuple(range(self.board_size)):
            raise ValueError("As posições do tabuleiro devem ser contíguas, de 0 a N-1.")
        self.space_by_position = {space.position: space for space in config.board}
        self.cards_by_type = {
            "chance": tuple(config.chance_cards),
            "community_chest": tuple(config.community_chest_cards),
        }
        if any(not cards for cards in self.cards_by_type.values()):
            raise ValueError("Cada casa de cartas precisa ter um baralho configurado.")

        settings = config.settings
        self.jail_position = (
            rules.jail_position if rules is not None else settings.jail_position
        )
        self.go_to_jail_position = (
            rules.go_to_jail_position
            if rules is not None
            else settings.go_to_jail_position
        )
        self.max_consecutive_doubles = (
            rules.max_consecutive_doubles if rules is not None else 3
        )
        self.max_jail_attempts = (
            rules.max_jail_attempts if rules is not None else settings.jail_max_turns
        )
        if not 0 <= self.jail_position < self.board_size:
            raise ValueError("jail_position está fora do tabuleiro.")
        if not 0 <= self.go_to_jail_position < self.board_size:
            raise ValueError("go_to_jail_position está fora do tabuleiro.")
        if self.max_consecutive_doubles <= 0 or self.max_jail_attempts <= 0:
            raise ValueError("As tentativas de prisão e duplas devem ser positivas.")

        self.dice_sides = dice_sides
        self.max_card_chain = max_card_chain
        self.expected_dice_total = (dice_sides[0] + 1 + dice_sides[1] + 1) / 2
        states = [
            MonopolyMovementState(position, False, 0, doubles_streak)
            for position in self.positions
            for doubles_streak in range(self.max_consecutive_doubles)
        ]
        states.extend(
            MonopolyMovementState(self.jail_position, True, attempts, 0)
            for attempts in range(self.max_jail_attempts)
        )
        super().__init__(StateSpace(states))
        self._outcome_cache: dict[
            MonopolyMovementState, tuple[MonopolyMovementOutcome, ...]
        ] = {}
        self._transition_cache: dict[
            MonopolyMovementState,
            tuple[tuple[MonopolyMovementState, float], ...],
        ] = {}

    def outcomes_from(
        self,
        source: MonopolyMovementState,
    ) -> tuple[MonopolyMovementOutcome, ...]:
        """Return dice and card outcomes reachable from a movement state."""
        self._validate_state(source)
        if source not in self._outcome_cache:
            self._outcome_cache[source] = self._calculate_outcomes(source)
        return self._outcome_cache[source]

    def transitions_from(
        self,
        source: MonopolyMovementState,
    ) -> tuple[tuple[MonopolyMovementState, float], ...]:
        """Aggregate outcome probabilities by resulting movement state."""
        self._validate_state(source)
        if source not in self._transition_cache:
            transitions: dict[MonopolyMovementState, float] = defaultdict(float)
            for outcome in self.outcomes_from(source):
                transitions[outcome.state] += outcome.probability
            self._transition_cache[source] = tuple(transitions.items())
        return self._transition_cache[source]

    def probability(
        self,
        source: MonopolyMovementState,
        target: MonopolyMovementState,
    ) -> float:
        """Return the one-roll transition probability between two states."""
        self._validate_state(target)
        return dict(self.transitions_from(source)).get(target, 0.0)

    def _validate_state(self, state: MonopolyMovementState) -> None:
        if not self.state_space.contains(state):
            raise ValueError(f"Estado não pertence ao modelo: {state!r}")

    def _calculate_outcomes(
        self,
        source: MonopolyMovementState,
    ) -> tuple[MonopolyMovementOutcome, ...]:
        outcomes: list[MonopolyMovementOutcome] = []
        sides_a, sides_b = self.dice_sides
        roll_probability = 1.0 / (sides_a * sides_b)

        for die_a in range(1, sides_a + 1):
            for die_b in range(1, sides_b + 1):
                dice_total = die_a + die_b
                is_double = die_a == die_b
                if source.in_jail:
                    self._jail_outcomes(
                        source,
                        dice_total,
                        is_double,
                        roll_probability,
                        outcomes,
                    )
                    continue

                doubles_streak = source.doubles_streak + 1 if is_double else 0
                if is_double and doubles_streak >= self.max_consecutive_doubles:
                    outcomes.append(
                        MonopolyMovementOutcome(
                            self._jailed_state(),
                            roll_probability,
                            None,
                            float(dice_total),
                            turn_ended=True,
                        )
                    )
                    continue

                destination = (source.position + dice_total) % self.board_size
                outcomes.extend(
                    self._resolve_landing(
                        destination,
                        doubles_streak,
                        roll_continues=is_double,
                        dice_total=float(dice_total),
                        probability=roll_probability,
                        rent_multiplier=1.0,
                        utility_reroll=False,
                        card_depth=0,
                    )
                )

        return tuple(outcomes)

    def _jail_outcomes(
        self,
        source: MonopolyMovementState,
        dice_total: int,
        is_double: bool,
        probability: float,
        outcomes: list[MonopolyMovementOutcome],
    ) -> None:
        next_attempt = source.jail_attempts + 1
        released = is_double or next_attempt >= self.max_jail_attempts
        if not released:
            outcomes.append(
                MonopolyMovementOutcome(
                    MonopolyMovementState(
                        self.jail_position,
                        True,
                        next_attempt,
                        0,
                    ),
                    probability,
                    None,
                    float(dice_total),
                    turn_ended=True,
                )
            )
            return

        destination = (self.jail_position + dice_total) % self.board_size
        outcomes.extend(
            self._resolve_landing(
                destination,
                0,
                roll_continues=False,
                dice_total=float(dice_total),
                probability=probability,
                rent_multiplier=1.0,
                utility_reroll=False,
                card_depth=0,
            )
        )

    def _resolve_landing(
        self,
        position: int,
        doubles_streak: int,
        *,
        roll_continues: bool,
        dice_total: float,
        probability: float,
        rent_multiplier: float,
        utility_reroll: bool,
        card_depth: int,
    ) -> list[MonopolyMovementOutcome]:
        if position == self.go_to_jail_position:
            return [
                MonopolyMovementOutcome(
                    self._jailed_state(),
                    probability,
                    None,
                    dice_total,
                    turn_ended=True,
                )
            ]

        space = self.space_by_position[position]
        deck_name = space.type if space.type in self.cards_by_type else None
        if deck_name is None:
            return [
                MonopolyMovementOutcome(
                    MonopolyMovementState(position, False, 0, doubles_streak),
                    probability,
                    position,
                    self.expected_dice_total if utility_reroll else dice_total,
                    rent_multiplier,
                    turn_ended=not roll_continues,
                )
            ]

        if card_depth >= self.max_card_chain:
            raise ValueError(
                "A cadeia de cartas excedeu max_card_chain; há um ciclo de "
                "movimentos que este modelo de draws independentes não representa."
            )

        cards = self.cards_by_type[deck_name]
        card_probability = probability / len(cards)
        outcomes: list[MonopolyMovementOutcome] = []
        for card in cards:
            action = card.action
            data = card.data
            if action == "go_to_jail":
                outcomes.append(
                    MonopolyMovementOutcome(
                        self._jailed_state(),
                        card_probability,
                        None,
                        dice_total,
                        turn_ended=True,
                    )
                )
                continue

            destination, moves = self._card_destination(position, action, data)
            if not moves:
                outcomes.append(
                    MonopolyMovementOutcome(
                        MonopolyMovementState(position, False, 0, doubles_streak),
                        card_probability,
                        position,
                        dice_total,
                        rent_multiplier,
                        turn_ended=not roll_continues,
                    )
                )
                continue

            card_rent_multiplier = float(data.get("rentMultiplier", 1))
            card_utility_reroll = action == "advance_to_nearest" and data.get("type") == "utility"
            outcomes.extend(
                self._resolve_landing(
                    destination,
                    doubles_streak,
                    roll_continues=roll_continues,
                    dice_total=dice_total,
                    probability=card_probability,
                    rent_multiplier=card_rent_multiplier,
                    utility_reroll=card_utility_reroll,
                    card_depth=card_depth + 1,
                )
            )
        return outcomes

    def _card_destination(
        self,
        position: int,
        action: str,
        data: Any,
    ) -> tuple[int, bool]:
        if action == "advance_to":
            destination = data.get("position")
            if not isinstance(destination, int) or destination not in self.space_by_position:
                raise ValueError(f"Carta advance_to tem posição inválida: {destination!r}.")
            return destination, True
        if action == "go_back":
            spaces = data.get("spaces")
            if not isinstance(spaces, int) or spaces < 0:
                raise ValueError(f"Carta go_back tem distância inválida: {spaces!r}.")
            return (position - spaces) % self.board_size, True
        if action == "advance_to_nearest":
            target_type = data.get("type")
            for offset in range(1, self.board_size + 1):
                destination = (position + offset) % self.board_size
                if self.space_by_position[destination].type == target_type:
                    return destination, True
            raise ValueError(f"Não há casas do tipo {target_type!r} no tabuleiro.")
        return position, False

    def _jailed_state(self) -> MonopolyMovementState:
        return MonopolyMovementState(self.jail_position, True, 0, 0)


def calculate_stationary_board_analysis(
    config: GameConfig,
    rules: Rules | None = None,
    *,
    dice_sides: tuple[int, int] = (6, 6),
    tolerance: float = 1e-10,
) -> StationaryBoardAnalysis:
    """Calculate tile landing probabilities per roll and arrivals per turn."""
    model = MonopolyBoardModel(config, rules, dice_sides=dice_sides)
    matrix = TransitionMatrix(model)
    stationary = calculate_stationary_distribution(matrix, tolerance=tolerance)
    per_roll = [0.0] * model.board_size
    end_turn_probability = 0.0
    event_rates: dict[tuple[int, float, float], float] = defaultdict(float)

    for state_index, state in enumerate(model.state_space):
        state_probability = float(stationary[state_index])
        for outcome in model.outcomes_from(state):
            transition_flow = state_probability * outcome.probability
            if outcome.turn_ended:
                end_turn_probability += transition_flow
            if outcome.landing_position is None:
                continue
            per_roll[outcome.landing_position] += transition_flow
            key = (
                outcome.landing_position,
                outcome.rent_dice_total,
                outcome.rent_multiplier,
            )
            event_rates[key] += transition_flow

    if not end_turn_probability > 0:
        raise ValueError("O modelo não encontrou encerramentos de turno.")

    events = tuple(
        LandingEventFrequency(
            tile_id=tile_id,
            rent_dice_total=dice_total,
            rent_multiplier=multiplier,
            probability_per_roll=probability,
            expected_arrivals_per_turn=probability / end_turn_probability,
        )
        for (tile_id, dice_total, multiplier), probability in sorted(event_rates.items())
    )
    return StationaryBoardAnalysis(
        positions=model.positions,
        stationary_states=model.state_space.states,
        stationary_state_probabilities=tuple(float(value) for value in stationary),
        landing_probabilities_per_roll=tuple(per_roll),
        expected_landings_per_turn=tuple(
            probability / end_turn_probability for probability in per_roll
        ),
        landing_events_per_turn=events,
        turn_end_probability_per_roll=float(end_turn_probability),
        expected_rolls_per_turn=1.0 / end_turn_probability,
    )
