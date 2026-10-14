
from collections import defaultdict
from collections.abc import Hashable, Mapping
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from markov.transition_matrix import TransitionMatrix


FloatVector = NDArray[np.float64]

TransitionKey = tuple[Hashable, Hashable]
ArrivalCounts = Mapping[Hashable, float]


def calculate_expected_arrivals_per_step(
    transition_matrix: TransitionMatrix,
    stationary_distribution: FloatVector,
    arrival_counts_by_transition: Mapping[
        TransitionKey, ArrivalCounts
    ],
    tolerance: float = 1e-8,
) -> tuple[tuple[Hashable, float], ...]:
    """
    Calcula o número esperado de chegadas a cada posição por passo.

    arrival_counts_by_transition associa cada par
    (estado_origem, estado_destino) a um mapeamento:

        posição -> número esperado de chegadas à posição
                   condicionado à transição.

    Exemplos:
        {(state_a, state_b): {"property_01": 1.0}}
        {(state_a, state_b): {"property_01": 0.5}}

    Uma contagem condicional pode ser fracionária quando representa
    uma esperança sobre diferentes eventos possíveis.

    Transições não incluídas no mapeamento têm contagem zero.

    A função não interpreta regras do jogo. Os eventos e suas
    contagens devem ser definidos pelo adaptador das regras.
    """
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
    ):
        raise TypeError("tolerance deve ser numérica.")

    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva.")

    state_space = transition_matrix.state_space
    matrix = transition_matrix.matrix
    size = transition_matrix.size

    probabilities = np.asarray(stationary_distribution, dtype=np.float64)

    if probabilities.ndim != 1 or probabilities.size != size:
        raise ValueError(
            "A distribuição deve conter uma probabilidade para cada estado."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("A distribuição contém valores não finitos.")

    if np.any(probabilities < -tolerance):
        raise ValueError("A distribuição contém probabilidades negativas.")

    if not np.isclose(
        probabilities.sum(), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("As probabilidades devem somar 1.")

    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    if not np.allclose(
        probabilities @ matrix,
        probabilities,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "A distribuição fornecida não é estacionária para esta matriz."
        )

    validated_events: dict[
        tuple[int, int], tuple[tuple[Hashable, float], ...]
    ] = {}

    for (source, target), arrival_counts in arrival_counts_by_transition.items():
        source_index = state_space.index_of(source)
        target_index = state_space.index_of(target)

        if matrix[source_index, target_index] <= 0:
            raise ValueError(
                f"A transição ({source!r}, {target!r}) "
                "tem probabilidade zero."
            )

        if not isinstance(arrival_counts, Mapping):
            raise TypeError(
                "As contagens de chegada devem ser fornecidas em um mapeamento."
            )

        validated_counts: list[tuple[Hashable, float]] = []

        for position, count in arrival_counts.items():
            try:
                hash(position)
            except TypeError as exc:
                raise TypeError(
                    "Os identificadores de posição devem ser hashable."
                ) from exc

            if isinstance(count, (bool, np.bool_)) or not isinstance(
                count, (int, float, np.integer, np.floating)
            ):
                raise TypeError("As contagens devem ser numéricas.")

            count = float(count)

            if not isfinite(count) or count < 0:
                raise ValueError(
                    "As contagens devem ser finitas e não negativas."
                )

            validated_counts.append((position, count))

        validated_events[(source_index, target_index)] = tuple(
            validated_counts
        )

    expected_arrivals: dict[Hashable, float] = defaultdict(float)

    for source_index in range(size):
        source_probability = probabilities[source_index]

        for target_index in np.flatnonzero(matrix[source_index] > 0):
            transition_probability = matrix[source_index, target_index]
            event_counts = validated_events.get(
                (source_index, int(target_index)), ()
            )

            for position, conditional_count in event_counts:
                expected_arrivals[position] += (
                    source_probability
                    * transition_probability
                    * conditional_count
                )

    return tuple(expected_arrivals.items())
