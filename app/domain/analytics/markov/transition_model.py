
from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import isclose, isfinite
from typing import Generic, TypeVar

from .state_space import StateSpace


State = TypeVar("State", bound=object)


class TransitionModel(ABC, Generic[State]):
    """Contrato para modelos de transição de uma cadeia de Markov."""

    def __init__(self, state_space: StateSpace[State]) -> None:
        self._state_space = state_space

    @property
    def state_space(self) -> StateSpace[State]:
        return self._state_space

    @abstractmethod
    def probability(self, source: State, target: State) -> float:
        """Retorna P(X[t+1] = target | X[t] = source)."""
        raise NotImplementedError

    @abstractmethod
    def transitions_from(self, source: State) -> tuple[tuple[State, float], ...]:
        """Retorna os destinos alcançáveis e suas probabilidades."""
        raise NotImplementedError


class TabularTransitionModel(TransitionModel[State]):
    """Modelo de transição definido por probabilidades explícitas."""

    def __init__(
        self,
        state_space: StateSpace[State],
        transitions: Mapping[State, Mapping[State, float]],
        tolerance: float = 1e-9,
    ) -> None:
        super().__init__(state_space)

        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance deve ser finita e não negativa.")

        if set(transitions.keys()) != set(state_space.states):
            raise ValueError(
                "Deve existir uma definição de transição para cada estado."
            )

        self._transitions: dict[State, tuple[tuple[State, float], ...]] = {}

        for source in state_space:
            destinations = transitions[source]

            if not destinations:
                raise ValueError(
                    f"O estado {source!r} não possui transições definidas."
                )

            if not set(destinations.keys()).issubset(state_space.states):
                raise ValueError(
                    f"O estado {source!r} possui destino fora do espaço de estados."
                )

            validated: list[tuple[State, float]] = []

            for target, probability in destinations.items():
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    raise TypeError("As probabilidades devem ser numéricas.")

                probability = float(probability)

                if not isfinite(probability) or probability < 0:
                    raise ValueError(
                        "As probabilidades devem ser finitas e não negativas."
                    )

                validated.append((target, probability))

            total_probability = sum(
                probability for _, probability in validated
            )

            if not isclose(
                total_probability,
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    f"As probabilidades de saída de {source!r} "
                    f"somam {total_probability}, e não 1."
                )

            self._transitions[source] = tuple(validated)

    def probability(self, source: State, target: State) -> float:
        """Retorna a probabilidade de uma transição específica."""

        self._validate_state(source)
        self._validate_state(target)

        for destination, probability in self._transitions[source]:
            if destination == target:
                return probability

        return 0.0

    def transitions_from(
        self,
        source: State,
    ) -> tuple[tuple[State, float], ...]:
        """Retorna as transições de saída de um estado."""

        self._validate_state(source)
        return self._transitions[source]

    def _validate_state(self, state: State) -> None:
        if not self.state_space.contains(state):
            raise ValueError(
                f"Estado não pertence ao espaço: {state!r}"
            )
