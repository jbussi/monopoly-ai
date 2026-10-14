
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Experiment = Callable[[np.random.Generator], float]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Resumo estatístico de um experimento de Monte Carlo."""

    n_simulations: int
    seed: int | None
    mean: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float
    samples: tuple[float, ...]


def run_monte_carlo(
    experiment: Experiment,
    n_simulations: int,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Executa um experimento repetidamente e resume seus resultados.

    Args:
        experiment: Função que recebe um gerador aleatório e retorna
            um resultado numérico.
        n_simulations: Número de execuções independentes do experimento.
        seed: Semente opcional para reproduzir a sequência aleatória.

    Returns:
        MonteCarloResult com as amostras e estatísticas descritivas.

    Raises:
        TypeError: Se os argumentos ou resultados tiverem tipos inválidos.
        ValueError: Se a quantidade de simulações for inválida ou
            algum resultado for não finito.
    """
    if not callable(experiment):
        raise TypeError("experiment deve ser uma função chamável.")

    if (
        not isinstance(n_simulations, (int, np.integer))
        or isinstance(n_simulations, (bool, np.bool_))
    ):
        raise TypeError("n_simulations deve ser um número inteiro.")

    if n_simulations < 2:
        raise ValueError("São necessárias pelo menos duas simulações.")

    if seed is not None:
        if (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
        ):
            raise TypeError("seed deve ser um inteiro ou None.")
        seed = int(seed)

    rng = np.random.default_rng(seed)
    samples: list[float] = []

    for _ in range(n_simulations):
        result = experiment(rng)

        if isinstance(result, (bool, np.bool_)) or not isinstance(
            result, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                "O experimento deve retornar um resultado numérico escalar."
            )

        value = float(result)

        if not np.isfinite(value):
            raise ValueError(
                "O experimento retornou um resultado não finito."
            )

        samples.append(value)

    values = np.asarray(samples, dtype=np.float64)

    return MonteCarloResult(
        n_simulations=n_simulations,
        seed=seed,
        mean=float(np.mean(values)),
        variance=float(np.var(values, ddof=1)),
        standard_deviation=float(np.std(values, ddof=1)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        samples=tuple(samples),
    )
