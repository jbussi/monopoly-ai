
from dataclasses import dataclass
from math import exp, isclose, isfinite, lgamma, log
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GoodnessOfFitResult:
    """Resultado de um teste qui-quadrado de aderência."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    observed_counts: tuple[int, ...]
    expected_counts: tuple[float, ...]
    minimum_expected_count: float
    approximation_reliable: bool


def _regularized_gamma_q(a: float, x: float) -> float:
    """
    Calcula Q(a, x), a função gama incompleta regularizada superior.

    É utilizada para calcular o valor-p da distribuição qui-quadrado.
    Implementação numérica baseada em série e fração contínua.
    """
    if not isfinite(a) or a <= 0:
        raise ValueError("a deve ser finito e positivo.")
    if np.isnan(x) or x < 0:
        raise ValueError("x deve ser não negativo.")
    if x == 0:
        return 1.0
    if np.isinf(x):
        return 0.0

    epsilon = 3e-14
    tiny = 1e-300
    max_iterations = 10_000
    log_factor = a * log(x) - x - lgamma(a)

    if x < a + 1.0:
        # Calcula P(a, x) por série e retorna Q(a, x) = 1 - P(a, x).
        term = 1.0 / a
        total = term
        denominator = a

        for _ in range(max_iterations):
            denominator += 1.0
            term *= x / denominator
            total += term

            if abs(term) <= abs(total) * epsilon:
                break
        else:
            raise ArithmeticError("A série da função gama não convergiu.")

        p_value = total * exp(log_factor)
        return min(1.0, max(0.0, 1.0 - p_value))

    # Calcula Q(a, x) diretamente por fração contínua.
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0:
        d = -d

    h = d

    for i in range(1, max_iterations + 1):
        an = -float(i) * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) <= epsilon:
            result = exp(log_factor) * h
            return min(1.0, max(0.0, result))

    raise ArithmeticError("A fração contínua da função gama não convergiu.")


def chi_square_goodness_of_fit(
    observed_counts: Sequence[int],
    expected_probabilities: Sequence[float],
    estimated_parameters: int = 0,
    tolerance: float = 1e-9,
) -> GoodnessOfFitResult:
    """
    Executa um teste qui-quadrado de aderência de Pearson.

    Args:
        observed_counts: Frequências observadas por categoria.
        expected_probabilities: Probabilidades teóricas correspondentes.
            Devem somar 1.
        estimated_parameters: Número de parâmetros estimados a partir
            dos próprios dados para definir as probabilidades.
        tolerance: Tolerância numérica para validar as probabilidades.

    Nota:
        A aproximação assintótica do teste pode ser inadequada quando
        as frequências esperadas são pequenas. O indicador
        approximation_reliable sinaliza a regra prática de que todas
        as frequências esperadas positivas sejam pelo menos 5.
    """
    if not isinstance(estimated_parameters, (int, np.integer)) or isinstance(
        estimated_parameters, (bool, np.bool_)
    ):
        raise TypeError("estimated_parameters deve ser um inteiro.")
    if estimated_parameters < 0:
        raise ValueError("estimated_parameters não pode ser negativo.")

    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
    ):
        raise TypeError("tolerance deve ser numérica.")
    if not isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance deve ser finita e não negativa.")

    if len(observed_counts) != len(expected_probabilities):
        raise ValueError(
            "As frequências observadas e as probabilidades devem ter "
            "o mesmo número de categorias."
        )
    if len(observed_counts) < 2:
        raise ValueError("São necessárias pelo menos duas categorias.")

    observed: list[int] = []
    for count in observed_counts:
        if not isinstance(count, (int, np.integer)) or isinstance(
            count, (bool, np.bool_)
        ):
            raise TypeError("As frequências observadas devem ser inteiros.")
        if count < 0:
            raise ValueError("As frequências observadas não podem ser negativas.")
        observed.append(int(count))

    probabilities: list[float] = []
    for probability in expected_probabilities:
        if (
            isinstance(probability, (bool, np.bool_))
            or not isinstance(
                probability, (int, float, np.integer, np.floating)
            )
        ):
            raise TypeError("As probabilidades devem ser numéricas.")

        probability = float(probability)
        if not isfinite(probability) or probability < 0:
            raise ValueError(
                "As probabilidades devem ser finitas e não negativas."
            )
        probabilities.append(probability)

    probability_sum = sum(probabilities)
    if not isclose(
        probability_sum, 1.0, rel_tol=0.0, abs_tol=float(tolerance)
    ):
        raise ValueError(
            f"As probabilidades devem somar 1; soma atual: {probability_sum}."
        )

    total_observations = sum(observed)
    if total_observations <= 0:
        raise ValueError("A amostra deve conter pelo menos uma observação.")

    expected = [
        total_observations * probability for probability in probabilities
    ]

    positive_categories = [
        i for i, probability in enumerate(probabilities) if probability > 0
    ]
    degrees_of_freedom = (
        len(positive_categories) - 1 - estimated_parameters
    )
    if degrees_of_freedom <= 0:
        raise ValueError(
            "Os graus de liberdade devem ser positivos. "
            "Verifique o número de categorias e parâmetros estimados."
        )

    # Uma observação em categoria teoricamente impossível contradiz o modelo.
    if any(
        probabilities[i] == 0 and observed[i] > 0
        for i in range(len(observed))
    ):
        return GoodnessOfFitResult(
            statistic=float("inf"),
            degrees_of_freedom=degrees_of_freedom,
            p_value=0.0,
            observed_counts=tuple(observed),
            expected_counts=tuple(expected),
            minimum_expected_count=min(expected[i] for i in positive_categories),
            approximation_reliable=False,
        )

    statistic = sum(
        (observed[i] - expected[i]) ** 2 / expected[i]
        for i in positive_categories
    )

    p_value = _regularized_gamma_q(
        degrees_of_freedom / 2.0,
        statistic / 2.0,
    )

    minimum_expected_count = min(
        expected[i] for i in positive_categories
    )

    return GoodnessOfFitResult(
        statistic=float(statistic),
        degrees_of_freedom=degrees_of_freedom,
        p_value=float(p_value),
        observed_counts=tuple(observed),
        expected_counts=tuple(expected),
        minimum_expected_count=float(minimum_expected_count),
        approximation_reliable=minimum_expected_count >= 5.0,
    )
