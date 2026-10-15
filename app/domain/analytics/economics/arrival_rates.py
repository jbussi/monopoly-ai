
from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from math import isfinite

from ..markov.transition_matrix import TransitionMatrix
from ..markov.stationary import calculate_stationary_distribution


State = Hashable


@dataclass(frozen=True, slots=True)
class PropertyArrivalRate:
    """Taxa de chegada esperada a uma propriedade por passo da cadeia."""

    property_id: str
    expected_arrivals_per_step: float


@dataclass(frozen=True, slots=True)
class ArrivalRatesResult:
    """Taxas de chegada estimadas para as propriedades mapeadas."""

    rates: tuple[PropertyArrivalRate, ...]
    total_mapped_arrivals_per_step: float

    def rate_for(self, property_id: str) -> float:
        """Retorna a taxa de chegada de uma propriedade."""
        for rate in self.rates:
            if rate.property_id == property_id:
                return rate.expected_arrivals_per_step
        raise KeyError(f"Propriedade sem taxa calculada: {property_id!r}")


def calculate_property_arrival_rates(
    transition_matrix: TransitionMatrix,
    property_by_transition: Mapping[tuple[State, State], str],
    tolerance: float = 1e-9,
) -> ArrivalRatesResult:
    """
    Calcula taxas de chegada a propriedades no regime estacionário.

    property_by_transition associa cada transição que representa uma
    chegada a um identificador de propriedade.

    Cada passo da cadeia representa a unidade temporal definida pelo
    modelo. Por exemplo, pode representar uma jogada de dados ou um
    turno completo, mas essa escolha deve ser consistente no modelo.

    Pressupõe uma distribuição estacionária única e que a matriz
    representa corretamente a dinâmica relevante do jogo.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance deve ser finita e não negativa.")

    state_space = transition_matrix.state_space
    matrix = transition_matrix.matrix

    validated_mapping: dict[tuple[State, State], str] = {}

    for transition, property_id in property_by_transition.items():
        if not isinstance(transition, tuple) or len(transition) != 2:
            raise ValueError(
                "Cada chave deve ser uma tupla (estado_origem, estado_destino)."
            )

        source, target = transition

        if not state_space.contains(source):
            raise ValueError(f"Estado de origem desconhecido: {source!r}")
        if not state_space.contains(target):
            raise ValueError(f"Estado de destino desconhecido: {target!r}")
        if not isinstance(property_id, str) or not property_id.strip():
            raise ValueError("property_id deve ser uma string não vazia.")

        source_index = state_space.index_of(source)
        target_index = state_space.index_of(target)

        if matrix[source_index, target_index] <= 0:
            raise ValueError(
                f"A transição {transition!r} tem probabilidade zero."
            )

        validated_mapping[transition] = property_id

    stationary = calculate_stationary_distribution(
        transition_matrix,
        tolerance=tolerance,
    )

    arrivals_by_property: dict[str, float] = defaultdict(float)

    for (source, target), property_id in validated_mapping.items():
        i = state_space.index_of(source)
        j = state_space.index_of(target)

        transition_flow = stationary[i] * matrix[i, j]
        arrivals_by_property[property_id] += float(transition_flow)

    rates = tuple(
        PropertyArrivalRate(
            property_id=property_id,
            expected_arrivals_per_step=rate,
        )
        for property_id, rate in sorted(arrivals_by_property.items())
    )

    total_arrivals = sum(
        rate.expected_arrivals_per_step for rate in rates
    )

    if total_arrivals > 1.0 + tolerance:
        raise ValueError(
            "A taxa total de chegada excede 1 por passo. "
            "Verifique se as transições foram mapeadas corretamente."
        )

    return ArrivalRatesResult(
        rates=rates,
        total_mapped_arrivals_per_step=float(total_arrivals),
    )
