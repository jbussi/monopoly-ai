
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tile(ABC):
    """Definição abstrata de uma casa do tabuleiro."""

    tile_id: int
    name: str


@dataclass(frozen=True, slots=True)
class SpecialTile(Tile):
    """Casa com uma função especial definida pela variante do jogo."""

    tile_type: str


@dataclass(frozen=True, slots=True)
class TaxTile(Tile):
    """Casa que cobra um valor definido."""

    tax_amount: int


@dataclass(frozen=True, slots=True)
class PropertyTile(Tile):
    """Definição de um imóvel disponível no tabuleiro."""

    price: int
    rent_values: tuple[int, ...]
    group: str
    mortgage_value: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("O preço não pode ser negativo.")

        if self.mortgage_value < 0:
            raise ValueError("O valor de hipoteca não pode ser negativo.")

        if not self.rent_values:
            raise ValueError("A tabela de aluguéis não pode ser vazia.")

        if any(value < 0 for value in self.rent_values):
            raise ValueError("Os aluguéis não podem ser negativos.")

    def rent_for_level(self, level: int) -> int:
        """Retorna o aluguel correspondente a um nível de construção."""
        if not 0 <= level < len(self.rent_values):
            raise ValueError("Nível de construção inválido.")

        return self.rent_values[level]


class Board:
    """Tabuleiro ordenado, com quantidade variável de casas."""

    def __init__(self, tiles: list[Tile]) -> None:
        if not tiles:
            raise ValueError("O tabuleiro deve possuir ao menos uma casa.")

        ids = [tile.tile_id for tile in tiles]

        if len(ids) != len(set(ids)):
            raise ValueError("Os IDs das casas devem ser únicos.")

        self._tiles = tuple(
            sorted(tiles, key=lambda tile: tile.tile_id)
        )
        self._position_by_id = {
            tile.tile_id: position
            for position, tile in enumerate(self._tiles)
        }

    @property
    def tiles(self) -> tuple[Tile, ...]:
        return self._tiles

    @property
    def size(self) -> int:
        return len(self._tiles)

    def get_tile(self, position: int) -> Tile:
        """Obtém a casa pela posição, considerando o circuito circular."""
        return self._tiles[position % self.size]

    def get_tile_by_id(self, tile_id: int) -> Tile:
        """Obtém uma casa por seu identificador."""
        try:
            return self._tiles[self._position_by_id[tile_id]]
        except KeyError as exc:
            raise ValueError(
                f"Não existe casa com tile_id={tile_id}."
            ) from exc
