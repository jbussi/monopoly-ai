
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiceRoll:
    """Resultado imutável de um lançamento de dois dados."""

    die1: int
    die2: int

    @property
    def total(self) -> int:
        return self.die1 + self.die2

    @property
    def is_double(self) -> bool:
        return self.die1 == self.die2


class DiceMechanism(ABC):
    """Contrato para mecanismos de lançamento de dois dados."""

    @abstractmethod
    def roll(self) -> DiceRoll:
        """Gera e retorna um resultado de lançamento."""
        raise NotImplementedError


class StandardDice(DiceMechanism):
    """Dois dados justos com número de faces configurável."""

    def __init__(
        self,
        sides: int | tuple[int, int] = 6,
        seed: int | None = None,
    ) -> None:
        if isinstance(sides, int):
            sides = (sides, sides)

        if (
            len(sides) != 2
            or any(
                not isinstance(n, int)
                or isinstance(n, bool)
                or n < 2
                for n in sides
            )
        ):
            raise ValueError(
                "Informe dois números inteiros de faces, "
                "ambos maiores ou iguais a 2."
            )

        self._sides = sides
        self._rng = random.Random(seed)

    @property
    def sides(self) -> tuple[int, int]:
        return self._sides

    def roll(self) -> DiceRoll:
        die1 = self._rng.randint(1, self._sides[0])
        die2 = self._rng.randint(1, self._sides[1])

        return DiceRoll(die1=die1, die2=die2)
