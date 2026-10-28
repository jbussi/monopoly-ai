
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Player:
    """
    Representa um participante da partida.

    As decisões do jogador serão definidas posteriormente
    por agentes ou políticas de decisão.
    """

    name: str
    player_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("O nome do jogador não pode ser vazio.")
