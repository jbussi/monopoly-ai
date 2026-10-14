
from collections.abc import Hashable

import numpy as np

from markov.transition_matrix import TransitionMatrix


def simulate_markov_chain(
    transition_matrix: TransitionMatrix,
    initial_state: Hashable,
    n_steps: int,
    seed: int | None = None,
) -> tuple[Hashable, ...]:
    """
    Simula uma trajetória de uma cadeia de Markov.

    Args:
        transition_matrix: Matriz de transição da cadeia.
        initial_state: Estado inicial da trajetória.
        n_steps: Número de transições a executar.
        seed: Semente opcional para reproduzir a simulação.

    Returns:
        Tupla contendo o estado inicial e todos os estados visitados.
        Seu comprimento será n_steps + 1.

    Raises:
        TypeError: Se os argumentos inteiros tiverem tipos inválidos.
        ValueError: Se o estado inicial não existir ou os parâmetros
            numéricos forem inválidos.
    """
    if (
        not isinstance(n_steps, (int, np.integer))
        or isinstance(n_steps, (bool, np.bool_))
    ):
        raise TypeError("n_steps deve ser um número inteiro.")

    if n_steps < 0:
        raise ValueError("n_steps deve ser não negativo.")

    if seed is not None:
        if (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
        ):
            raise TypeError("seed deve ser um inteiro ou None.")
        seed = int(seed)

    state_space = transition_matrix.state_space
    current_index = state_space.index_of(initial_state)

    rng = np.random.default_rng(seed)
    trajectory: list[Hashable] = [initial_state]

    for _ in range(n_steps):
        probabilities = transition_matrix.matrix[current_index]

        current_index = int(
            rng.choice(transition_matrix.size, p=probabilities)
        )

        trajectory.append(state_space.state_at(current_index))

    return tuple(trajectory)
