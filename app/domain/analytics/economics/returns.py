
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ReturnResult:
    """Resumo de retorno de um investimento."""

    initial_investment: float
    total_inflows: float
    total_outflows: float
    net_cash_flow: float
    net_profit: float
    roi: float | None


@dataclass(frozen=True, slots=True)
class IncrementalReturnResult:
    """Retorno adicional de um investimento incremental."""

    incremental_investment: float
    incremental_benefit: float
    net_incremental_profit: float
    roi: float | None


def _validate_amount(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico.")

    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} deve ser finito.")

    return value


def calculate_return(
    initial_investment: float,
    total_inflows: float,
    total_outflows: float,
) -> ReturnResult:
    """
    Calcula o retorno de um investimento.

    total_inflows e total_outflows são os fluxos operacionais acumulados,
    sem incluir novamente o investimento inicial.

    O ROI é calculado como lucro líquido / investimento inicial.
    Retorna None quando o investimento inicial é zero.
    """
    initial_investment = _validate_amount(
        "initial_investment", initial_investment
    )
    total_inflows = _validate_amount("total_inflows", total_inflows)
    total_outflows = _validate_amount("total_outflows", total_outflows)

    if initial_investment < 0:
        raise ValueError("initial_investment não pode ser negativo.")
    if total_inflows < 0 or total_outflows < 0:
        raise ValueError(
            "total_inflows e total_outflows devem ser não negativos."
        )

    net_cash_flow = total_inflows - total_outflows
    net_profit = net_cash_flow - initial_investment

    roi = (
        net_profit / initial_investment
        if initial_investment > 0
        else None
    )

    return ReturnResult(
        initial_investment=initial_investment,
        total_inflows=total_inflows,
        total_outflows=total_outflows,
        net_cash_flow=net_cash_flow,
        net_profit=net_profit,
        roi=roi,
    )


def calculate_incremental_return(
    incremental_investment: float,
    benefit_before: float,
    benefit_after: float,
) -> IncrementalReturnResult:
    """
    Calcula o retorno incremental de uma melhoria.

    benefit_before e benefit_after representam benefícios acumulados
    no mesmo horizonte e na mesma unidade monetária.
    """
    incremental_investment = _validate_amount(
        "incremental_investment", incremental_investment
    )
    benefit_before = _validate_amount("benefit_before", benefit_before)
    benefit_after = _validate_amount("benefit_after", benefit_after)

    if incremental_investment < 0:
        raise ValueError("incremental_investment não pode ser negativo.")

    incremental_benefit = benefit_after - benefit_before
    net_incremental_profit = (
        incremental_benefit - incremental_investment
    )

    roi = (
        net_incremental_profit / incremental_investment
        if incremental_investment > 0
        else None
    )

    return IncrementalReturnResult(
        incremental_investment=incremental_investment,
        incremental_benefit=incremental_benefit,
        net_incremental_profit=net_incremental_profit,
        roi=roi,
    )


def calculate_cumulative_return(
    cash_flows: Sequence[float],
    initial_investment: float = 0.0,
) -> float:
    """
    Calcula o resultado acumulado descontando o investimento inicial.

    cash_flows contém os fluxos posteriores ao investimento inicial.
    """
    initial_investment = _validate_amount(
        "initial_investment", initial_investment
    )
    if initial_investment < 0:
        raise ValueError("initial_investment não pode ser negativo.")

    validated_flows = [
        _validate_amount("cash_flow", cash_flow)
        for cash_flow in cash_flows
    ]

    return float(sum(validated_flows) - initial_investment)


def calculate_simple_periodic_return(
    initial_investment: float,
    cumulative_net_profit: float,
    n_periods: int,
) -> float | None:
    """
    Calcula o lucro médio por período como proporção do investimento inicial.

    Não representa uma taxa composta nem uma taxa interna de retorno.
    """
    initial_investment = _validate_amount(
        "initial_investment", initial_investment
    )
    cumulative_net_profit = _validate_amount(
        "cumulative_net_profit", cumulative_net_profit
    )

    if initial_investment < 0:
        raise ValueError("initial_investment não pode ser negativo.")
    if not isinstance(n_periods, int) or isinstance(n_periods, bool):
        raise TypeError("n_periods deve ser um inteiro.")
    if n_periods <= 0:
        raise ValueError("n_periods deve ser positivo.")

    if initial_investment == 0:
        return None

    return (
        cumulative_net_profit / initial_investment / n_periods
    )
