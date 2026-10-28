
import numpy as np
from numpy.typing import NDArray

from .transition_matrix import TransitionMatrix


FloatVector = NDArray[np.float64]


def calculate_stationary_distribution(
    transition_matrix: TransitionMatrix,
    tolerance: float = 1e-10,
) -> FloatVector:
    """
    Calcula a distribuição estacionária única de uma cadeia finita.

    Resolve pi = pi @ P, sujeita a pi_i >= 0 e sum(pi) = 1.

    Raises:
        ValueError: Se a tolerância for inválida, se a distribuição
            estacionária não for única ou se a solução numérica falhar.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance deve ser finita e positiva.")

    P = transition_matrix.matrix
    size = transition_matrix.size

    # A distribuição é tratada como vetor coluna neste sistema linear:
    # (P.T - I) @ pi = 0.
    system = P.T - np.eye(size, dtype=np.float64)

    # Uma distribuição estacionária é única quando o núcleo de
    # (P.T - I) tem dimensão 1, isto é, quando seu posto é size - 1.
    rank = np.linalg.matrix_rank(system, tol=tolerance)

    if rank != size - 1:
        raise ValueError(
            "A cadeia não possui distribuição estacionária única. "
            "Verifique se existem múltiplas classes fechadas."
        )

    # Acrescenta a restrição sum(pi) = 1 ao sistema.
    augmented_system = np.vstack(
        [system, np.ones((1, size), dtype=np.float64)]
    )
    right_hand_side = np.append(
        np.zeros(size, dtype=np.float64),
        1.0,
    )

    probabilities, _, _, _ = np.linalg.lstsq(
        augmented_system,
        right_hand_side,
        rcond=None,
    )

    # Valida a solução antes de corrigir pequenos erros numéricos.
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("A solução numérica contém valores não finitos.")

    if np.any(probabilities < -tolerance):
        raise ValueError(
            "A solução contém probabilidades negativas além da tolerância."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("A solução não satisfaz a normalização.")

    # Corrige apenas resíduos negativos muito pequenos de ponto flutuante.
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    # Confirma que o resultado satisfaz a equação estacionária.
    residual = probabilities @ P - probabilities

    if not np.allclose(
        residual,
        0.0,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "A solução numérica não satisfaz a equação estacionária."
        )

    probabilities.setflags(write=False)
    return probabilities
