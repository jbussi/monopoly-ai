
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class Piece:
    """
    Representa uma peça que percorre o tabuleiro.

    Uma peça pode estar associada a vários jogadores.
    Na configuração inicial do jogo, associaremos apenas um.
    """

    name: str
    player_ids: list[UUID]
    piece_id: UUID = field(default_factory=uuid4)
    position: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("O nome da peça não pode ser vazio.")

        if self.position < 0:
            raise ValueError("A posição inicial não pode ser negativa.")

        if not self.player_ids:
            raise ValueError(
                "Uma peça deve possuir ao menos um jogador associado."
            )

        if len(self.player_ids) != len(set(self.player_ids)):
            raise ValueError(
                "Uma peça não pode possuir jogadores duplicados."
            )

    def add_player(self, player_id: UUID) -> None:
        """Associa um jogador à peça."""
        if player_id in self.player_ids:
            raise ValueError(
                "O jogador já está associado a esta peça."
            )

        self.player_ids.append(player_id)

    def remove_player(self, player_id: UUID) -> None:
        """Remove a associação de um jogador à peça."""
        if player_id not in self.player_ids:
            raise ValueError(
                "O jogador não está associado a esta peça."
            )

        if len(self.player_ids) == 1:
            raise ValueError(
                "Uma peça deve possuir ao menos um jogador associado."
            )

        self.player_ids.remove(player_id)
