
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Representa um intervalo de confiança."""

    lower: float
    upper: float
    confidence_level: float


def _validate_confidence_level(confidence_level: float) -> None:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
    ):
        raise TypeError("confidence_level deve ser numérico.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level deve estar entre 0 e 1.")


def mean_confidence_interval(
    samples: Sequence[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Calcula um intervalo normal aproximado para a média.

    Pressupõe observações independentes e utiliza o desvio-padrão
    amostral. A aproximação é mais adequada para amostras grandes.
    """
    _validate_confidence_level(confidence_level)

    values = np.asarray(samples, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("samples deve ser unidimensional.")

    if values.size < 2:
        raise ValueError("São necessárias pelo menos duas observações.")

    if not np.all(np.isfinite(values)):
        raise ValueError("As observações devem ser finitas.")

    n = values.size
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / sqrt(n))

    alpha = 1.0 - confidence_level
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    margin = z_critical * standard_error

    return ConfidenceInterval(
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence_level,
    )


def proportion_confidence_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Calcula um intervalo de confiança de Wilson para uma proporção.

    Args:
        successes: Número de sucessos observados.
        trials: Número total de ensaios independentes.
        confidence_level: Nível de confiança, entre 0 e 1.
    """
    _validate_confidence_level(confidence_level)

    for name, value in (("successes", successes), ("trials", trials)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
        ):
            raise TypeError(f"{name} deve ser um número inteiro.")

    if trials <= 0:
        raise ValueError("trials deve ser positivo.")

    if not 0 <= successes <= trials:
        raise ValueError("successes deve estar entre 0 e trials.")

    p_hat = successes / trials
    alpha = 1.0 - confidence_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_squared = z * z

    denominator = 1.0 + z_squared / trials
    center = (p_hat + z_squared / (2.0 * trials)) / denominator

    half_width = (
        z
        * sqrt(
            p_hat * (1.0 - p_hat) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )

    return ConfidenceInterval(
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
        confidence_level=confidence_level,
    )
