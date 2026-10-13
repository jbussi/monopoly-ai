
from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar


State = TypeVar("State", bound=Hashable)


class StateSpace(Generic[State]):
    """Representa um espaço finito e ordenado de estados."""

    def __init__(self, states: Iterable[State]) -> None:
        ordered_states = tuple(states)

        if not ordered_states:
            raise ValueError("O espaço de estados não pode ser vazio.")

        if len(set(ordered_states)) != len(ordered_states):
            raise ValueError("O espaço de estados contém estados duplicados.")

        self._states: tuple[State, ...] = ordered_states
        self._index_by_state: dict[State, int] = {
            state: index
            for index, state in enumerate(self._states)
        }

    @property
    def states(self) -> tuple[State, ...]:
        """Retorna os estados na ordem em que foram definidos."""
        return self._states

    @property
    def size(self) -> int:
        """Retorna a quantidade de estados."""
        return len(self._states)

    def index_of(self, state: State) -> int:
        """Retorna o índice de um estado."""
        try:
            return self._index_by_state[state]
        except KeyError as exc:
            raise ValueError(
                f"Estado não pertence ao espaço: {state!r}"
            ) from exc

    def state_at(self, index: int) -> State:
        """Retorna o estado correspondente a um índice."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("O índice deve ser um número inteiro.")

        if not 0 <= index < self.size:
            raise IndexError(
                f"Índice {index} fora dos limites do espaço de estados."
            )

        return self._states[index]

    def contains(self, state: State) -> bool:
        """Verifica se um estado pertence ao espaço."""
        return state in self._index_by_state

    def __len__(self) -> int:
        return self.size

    def __contains__(self, state: object) -> bool:
        return state in self._index_by_state

    def __iter__(self):
        return iter(self._states)
