
from collections import deque

import numpy as np
from numpy.typing import NDArray

from .transition_matrix import TransitionMatrix

FloatVector = NDArray[np.float64]


def calculate_expected_hitting_times(
    transition_matrix: TransitionMatrix,
    target: object,
    tolerance: float = 1e-10,
) -> FloatVector:
    """
    Calcula o tempo esperado de primeira passagem até um estado-alvo.

    Retorna um vetor alinhado à ordem dos estados da matriz:
    - 0.0 para o próprio estado-alvo;
    - valor finito para estados com tempo esperado finito;
    - np.inf para estados cujo tempo esperado é infinito.

    O tempo é medido em número de transições.

    Raises:
        ValueError: Se o estado-alvo não pertencer ao espaço de estados,
            a tolerância for inválida ou o cálculo numérico falhar.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva.")

    state_space = transition_matrix.state_space
    P = transition_matrix.matrix
    size = transition_matrix.size

    target_index = state_space.index_of(target)

    # Grafo reverso: identifica estados que conseguem alcançar o alvo.
    predecessors: list[list[int]] = [[] for _ in range(size)]

    for source_index in range(size):
        for destination_index in np.flatnonzero(P[source_index] > 0):
            predecessors[destination_index].append(source_index)

    can_reach_target = {target_index}
    queue = deque([target_index])

    while queue:
        current = queue.popleft()

        for predecessor in predecessors[current]:
            if predecessor not in can_reach_target:
                can_reach_target.add(predecessor)
                queue.append(predecessor)

    # Um estado que pode transitar para fora do conjunto alcançável
    # tem probabilidade positiva de nunca atingir o alvo.
    infinite_states: set[int] = set()
    queue = deque()

    for source_index in can_reach_target:
        for destination_index in np.flatnonzero(P[source_index] > 0):
            if destination_index not in can_reach_target:
                infinite_states.add(source_index)
                queue.append(source_index)
                break

    # Propaga a condição de tempo esperado infinito aos predecessores.
    while queue:
        current = queue.popleft()

        for predecessor in predecessors[current]:
            if (
                predecessor in can_reach_target
                and predecessor not in infinite_states
            ):
                infinite_states.add(predecessor)
                queue.append(predecessor)

    finite_states = can_reach_target - infinite_states - {target_index}
    finite_indices = sorted(finite_states)

    # Estados inalcançáveis ou com tempo esperado infinito começam em inf.
    expected_times = np.full(size, np.inf, dtype=np.float64)
    expected_times[target_index] = 0.0

    if finite_indices:
        # m = 1 + Qm  =>  (I - Q)m = 1
        Q = P[np.ix_(finite_indices, finite_indices)]
        system = np.eye(len(finite_indices), dtype=np.float64) - Q
        right_hand_side = np.ones(len(finite_indices), dtype=np.float64)

        try:
            solution = np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Não foi possível resolver o sistema de tempos de passagem."
            ) from exc

        if not np.all(np.isfinite(solution)):
            raise ValueError("A solução contém valores não finitos.")

        if np.any(solution < -tolerance):
            raise ValueError("A solução contém tempos negativos.")

        residual = system @ solution - right_hand_side
        if not np.allclose(residual, 0.0, rtol=0.0, atol=tolerance):
            raise ValueError(
                "A solução numérica não satisfaz o sistema de equações."
            )

        solution = np.maximum(solution, 0.0)
        expected_times[finite_indices] = solution

    expected_times.setflags(write=False)
    return expected_times
