
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class CardEffect(ABC):
    """Contrato para representar um efeito de carta."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Representa o efeito em formato serializável."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Card:
    """Definição imutável de uma carta."""

    card_id: str
    text: str
    effect: CardEffect


class CardDeck(ABC):
    """Contrato para baralhos de cartas."""

    @abstractmethod
    def draw(self) -> Card:
        """Retira uma carta do baralho."""
        raise NotImplementedError

    @abstractmethod
    def return_card(self, card: Card) -> None:
        """Devolve uma carta ao baralho."""
        raise NotImplementedError

    @abstractmethod
    def shuffle(self) -> None:
        """Embaralha as cartas disponíveis."""
        raise NotImplementedError


class StandardCardDeck(CardDeck):
    """
    Baralho com compra, descarte e embaralhamento.

    As cartas retiradas permanecem fora do baralho até serem
    devolvidas explicitamente ou o descarte ser reciclado.
    """

    def __init__(
        self,
        cards: list[Card],
        seed: int | None = None,
        recycle_discard: bool = True,
    ) -> None:
        if not cards:
            raise ValueError("O baralho deve possuir ao menos uma carta.")

        ids = [card.card_id for card in cards]
        if len(ids) != len(set(ids)):
            raise ValueError("Os IDs das cartas devem ser únicos no baralho.")

        self._cards = list(cards)
        self._discard_pile: list[Card] = []
        self._rng = random.Random(seed)
        self._recycle_discard = recycle_discard

        self.shuffle()

    @property
    def remaining(self) -> int:
        return len(self._cards)

    @property
    def discarded(self) -> int:
        return len(self._discard_pile)

    def draw(self) -> Card:
        if not self._cards:
            if not self._recycle_discard or not self._discard_pile:
                raise LookupError("Não há cartas disponíveis para compra.")

            self._cards = self._discard_pile
            self._discard_pile = []
            self.shuffle()

        return self._cards.pop(0)

    def return_card(self, card: Card) -> None:
        self._discard_pile.append(card)

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)
