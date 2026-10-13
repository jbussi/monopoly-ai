import numpy as np
from numpy.typing import NDArray

from .transition_model import TransitionModel


FloatMatrix = NDArray[np.float64]


class TransitionMatrix:
    """Representa a matriz de transição de uma cadeia de Markov finita."""

    def __init__(
        self,
        model: TransitionModel,
        tolerance: float = 1e-9,
    ) -> None:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance deve ser finita e não negativa.")

        self._state_space = model.state_space
        size = self._state_space.size

        matrix = np.zeros((size, size), dtype=np.float64)

        for source in self._state_space:
            i = self._state_space.index_of(source)

            for target, probability in model.transitions_from(source):
                j = self._state_space.index_of(target)
                matrix[i, j] += probability

        self._validate(matrix, tolerance)

        matrix.setflags(write=False)
        self._matrix = matrix

    @property
    def state_space(self):
        """Retorna o espaço de estados associado à matriz."""
        return self._state_space

    @property
    def matrix(self) -> FloatMatrix:
        """Retorna uma visão somente para leitura da matriz."""
        return self._matrix

    @property
    def size(self) -> int:
        """Retorna a quantidade de estados."""
        return self._matrix.shape[0]

    def probability(self, source, target) -> float:
        """Retorna a probabilidade de transição entre dois estados."""
        i = self._state_space.index_of(source)
        j = self._state_space.index_of(target)
        return float(self._matrix[i, j])

    def _validate(
        self,
        matrix: FloatMatrix,
        tolerance: float,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A matriz de transição deve ser quadrada.")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("A matriz contém valores não finitos.")

        if np.any(matrix < 0):
            raise ValueError("A matriz contém probabilidades negativas.")

        row_sums = matrix.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "Cada linha da matriz deve somar 1."
            )
