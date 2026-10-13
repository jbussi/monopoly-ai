
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GameAction:
    """Representa um comando enviado à engine."""

    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_type.strip():
            raise ValueError("O tipo da ação não pode ser vazio.")

        if not isinstance(self.payload, dict):
            raise TypeError("O payload da ação deve ser um dicionário.")
