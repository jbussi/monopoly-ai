
from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class Outcome:
    """Um resultado possível e sua probabilidade."""

    value: float
    probability: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("value deve ser numérico.")
        if not isfinite(float(self.value)):
            raise ValueError("value deve ser finito.")

        if isinstance(self.probability, (bool, np.bool_)) or not isinstance(
            self.probability,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("probability deve ser numérica.")

        probability = float(self.probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability deve estar entre 0 e 1.")

        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True)
class ExpectedValueResult:
    """Medidas de uma distribuição discreta de resultados."""

    mean: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float


def _validate_distribution(
    outcomes: Sequence[Outcome],
    tolerance: float = 1e-9,
) -> tuple[Outcome, ...]:
    if not outcomes:
        raise ValueError("A distribuição não pode ser vazia.")

    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance deve ser finita e não negativa.")

    validated = tuple(outcomes)

    if any(not isinstance(outcome, Outcome) for outcome in validated):
        raise TypeError("Todos os elementos devem ser instâncias de Outcome.")

    total_probability = sum(
        outcome.probability for outcome in validated
    )
    if not isclose(
        total_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"As probabilidades devem somar 1; soma atual: "
            f"{total_probability}."
        )

    return validated


def calculate_expected_value(
    outcomes: Sequence[Outcome],
    tolerance: float = 1e-9,
) -> ExpectedValueResult:
    """
    Calcula valor esperado, variância populacional e desvio-padrão.

    A distribuição representa os resultados possíveis de uma variável
    aleatória discreta, e não uma amostra de observações.
    """
    distribution = _validate_distribution(outcomes, tolerance)

    mean = sum(
        outcome.value * outcome.probability
        for outcome in distribution
    )

    variance = sum(
        outcome.probability * (outcome.value - mean) ** 2
        for outcome in distribution
    )

    return ExpectedValueResult(
        mean=float(mean),
        variance=float(variance),
        standard_deviation=float(np.sqrt(variance)),
        minimum=min(outcome.value for outcome in distribution),
        maximum=max(outcome.value for outcome in distribution),
    )


def calculate_conditional_expected_value(
    outcomes: Sequence[Outcome],
    condition_probabilities: Sequence[float],
    tolerance: float = 1e-9,
) -> float:
    """
    Calcula E[X | A] a partir das probabilidades condicionais P(X=x | A).

    condition_probabilities[i] deve corresponder a outcomes[i].
    Essas probabilidades condicionais devem somar 1.
    """
    if len(outcomes) != len(condition_probabilities):
        raise ValueError(
            "Deve existir uma probabilidade condicional para cada resultado."
        )

    if not outcomes:
        raise ValueError("A distribuição não pode ser vazia.")

    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance deve ser finita e não negativa.")

    validated_outcomes = tuple(outcomes)
    if any(not isinstance(outcome, Outcome) for outcome in validated_outcomes):
        raise TypeError("Todos os elementos devem ser instâncias de Outcome.")

    probabilities: list[float] = []
    for probability in condition_probabilities:
        if isinstance(probability, (bool, np.bool_)) or not isinstance(
            probability, (int, float, np.integer, np.floating)
        ):
            raise TypeError("As probabilidades condicionais devem ser numéricas.")

        probability = float(probability)
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                "As probabilidades condicionais devem estar entre 0 e 1."
            )
        probabilities.append(probability)

    total_probability = sum(probabilities)
    if not isclose(
        total_probability, 1.0, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(
            "As probabilidades condicionais devem somar 1."
        )

    return float(sum(
        outcome.value * probability
        for outcome, probability in zip(
            validated_outcomes,
            probabilities,
            strict=True,
        )
    ))
