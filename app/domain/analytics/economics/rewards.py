
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class RewardType(StrEnum):
    """Categorias de resultados econômicos."""

    RENT = "rent"
    PROPERTY_PURCHASE = "property_purchase"
    HOUSE_PURCHASE = "house_purchase"
    HOTEL_PURCHASE = "hotel_purchase"
    TAX = "tax"
    SALARY = "salary"
    TRADE = "trade"
    MORTGAGE = "mortgage"
    MORTGAGE_REDEMPTION = "mortgage_redemption"
    BANKRUPTCY = "bankruptcy"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Reward:
    """
    Resultado econômico de um evento sob a perspectiva de um agente.

    amount:
        Valor assinado. Positivo representa entrada econômica;
        negativo representa saída econômica.

    player_id:
        Identificador do agente cuja perspectiva está sendo registrada.

    event_type:
        Categoria do evento.

    turn:
        Turno em que o evento ocorreu, se conhecido.

    event_id:
        Identificador opcional para relacionar registros do mesmo evento.
    """

    amount: float
    player_id: str
    event_type: RewardType
    turn: int | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.amount, bool) or not isinstance(
            self.amount, (int, float)
        ):
            raise TypeError("amount deve ser numérico.")
        if not isfinite(self.amount):
            raise ValueError("amount deve ser finito.")

        object.__setattr__(self, "amount", float(self.amount))

        if not isinstance(self.player_id, str) or not self.player_id.strip():
            raise ValueError("player_id deve ser uma string não vazia.")

        if not isinstance(self.event_type, RewardType):
            raise TypeError("event_type deve ser um RewardType.")

        if self.turn is not None:
            if (
                not isinstance(self.turn, int)
                or isinstance(self.turn, bool)
            ):
                raise TypeError("turn deve ser um inteiro ou None.")
            if self.turn < 0:
                raise ValueError("turn não pode ser negativo.")

        if self.event_id is not None and (
            not isinstance(self.event_id, str) or not self.event_id.strip()
        ):
            raise ValueError("event_id deve ser None ou uma string não vazia.")


@dataclass(frozen=True, slots=True)
class RewardSummary:
    """Resumo dos resultados econômicos de um agente."""

    player_id: str
    total_reward: float
    total_inflows: float
    total_outflows: float
    n_events: int
    reward_by_type: tuple[tuple[RewardType, float], ...]


class RewardLedger:
    """
    Agrega recompensas sem alterar o estado financeiro do jogo.

    Cada registro representa a perspectiva de um único agente.
    """

    def __init__(self, rewards: Iterable[Reward] = ()) -> None:
        self._rewards: list[Reward] = []
        for reward in rewards:
            self.add(reward)

    @property
    def rewards(self) -> tuple[Reward, ...]:
        """Retorna os registros em uma sequência imutável."""
        return tuple(self._rewards)

    def add(self, reward: Reward) -> None:
        if not isinstance(reward, Reward):
            raise TypeError("O registro deve ser uma instância de Reward.")
        self._rewards.append(reward)

    def summarize(self, player_id: str) -> RewardSummary:
        """Resume os resultados econômicos de um agente."""
        if not isinstance(player_id, str) or not player_id.strip():
            raise ValueError("player_id deve ser uma string não vazia.")

        player_rewards = [
            reward
            for reward in self._rewards
            if reward.player_id == player_id
        ]

        reward_by_type: dict[RewardType, float] = defaultdict(float)
        total_inflows = 0.0
        total_outflows = 0.0

        for reward in player_rewards:
            reward_by_type[reward.event_type] += reward.amount

            if reward.amount > 0:
                total_inflows += reward.amount
            elif reward.amount < 0:
                total_outflows += abs(reward.amount)

        return RewardSummary(
            player_id=player_id,
            total_reward=sum(r.amount for r in player_rewards),
            total_inflows=total_inflows,
            total_outflows=total_outflows,
            n_events=len(player_rewards),
            reward_by_type=tuple(sorted(
                reward_by_type.items(),
                key=lambda item: item[0].value,
            )),
        )

    def rewards_for(self, player_id: str) -> tuple[Reward, ...]:
        """Retorna os registros de um agente, preservando a ordem."""
        return tuple(
            reward
            for reward in self._rewards
            if reward.player_id == player_id
        )
