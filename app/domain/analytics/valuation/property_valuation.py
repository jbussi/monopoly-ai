
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PropertyValuation:
    """Resultado da avaliação financeira de uma propriedade."""

    property_id: str
    initial_investment: float
    operating_cash_flows: tuple[float, ...]
    terminal_value: float
    total_operating_cash_flow: float
    net_profit: float
    roi: float | None
    net_present_value: float
    discounted_operating_cash_flows: tuple[float, ...]
    discounted_terminal_value: float


def _validate_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico.")

    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{name} deve ser finito.")

    return value


def evaluate_property(
    property_id: str,
    purchase_price: float,
    additional_initial_investment: float,
    operating_cash_flows: Sequence[float],
    discount_rate_per_period: float,
    terminal_value: float = 0.0,
) -> PropertyValuation:
    """
    Avalia uma propriedade por fluxo de caixa descontado.

    Parâmetros:
    - property_id: identificador da propriedade.
    - purchase_price: preço de aquisição.
    - additional_initial_investment: investimentos iniciais adicionais,
      como construções realizadas no instante inicial.
    - operating_cash_flows: fluxos líquidos esperados de cada período,
      posteriores ao investimento inicial.
    - discount_rate_per_period: taxa de desconto por período.
    - terminal_value: valor residual recebido ao final do horizonte.

    Convenções:
    - O investimento inicial ocorre no instante zero.
    - O primeiro fluxo operacional ocorre no período 1.
    - Os fluxos podem ser positivos ou negativos.
    - O valor terminal é recebido ao final do último período e não deve
      estar incluído nos fluxos operacionais.
    - O ROI é o lucro líquido acumulado dividido pelo investimento inicial.
      Não é uma taxa anualizada nem uma taxa composta.
    - O VPL desconta todos os fluxos pela taxa fornecida.

    Esta avaliação considera os fluxos esperados como entradas conhecidas.
    Ela não calcula, por si só, o risco ou a distribuição de probabilidade
    desses fluxos.
    """
    if not isinstance(property_id, str) or not property_id.strip():
        raise ValueError("property_id deve ser uma string não vazia.")

    purchase_price = _validate_number("purchase_price", purchase_price)
    additional_initial_investment = _validate_number(
        "additional_initial_investment",
        additional_initial_investment,
    )
    discount_rate = _validate_number(
        "discount_rate_per_period",
        discount_rate_per_period,
    )
    terminal_value = _validate_number("terminal_value", terminal_value)

    if purchase_price < 0:
        raise ValueError("purchase_price não pode ser negativo.")

    if additional_initial_investment < 0:
        raise ValueError(
            "additional_initial_investment não pode ser negativo."
        )

    if discount_rate <= -1.0:
        raise ValueError("A taxa de desconto deve ser maior que -100%.")

    if terminal_value < 0:
        raise ValueError("terminal_value não pode ser negativo.")

    if not isinstance(operating_cash_flows, Sequence):
        raise TypeError("operating_cash_flows deve ser uma sequência.")

    if isinstance(operating_cash_flows, (str, bytes)):
        raise TypeError("operating_cash_flows deve conter valores numéricos.")

    if len(operating_cash_flows) == 0:
        raise ValueError("É necessário fornecer pelo menos um período.")

    cash_flows = tuple(
        _validate_number(f"operating_cash_flows[{index}]", value)
        for index, value in enumerate(operating_cash_flows)
    )

    initial_investment = (
        purchase_price + additional_initial_investment
    )

    total_operating_cash_flow = sum(cash_flows)

    net_profit = (
        total_operating_cash_flow
        + terminal_value
        - initial_investment
    )

    roi = (
        net_profit / initial_investment
        if initial_investment > 0
        else None
    )

    discounted_cash_flows = tuple(
        cash_flow / (1.0 + discount_rate) ** period
        for period, cash_flow in enumerate(cash_flows, start=1)
    )

    discounted_terminal_value = (
        terminal_value / (1.0 + discount_rate) ** len(cash_flows)
    )

    net_present_value = (
        -initial_investment
        + sum(discounted_cash_flows)
        + discounted_terminal_value
    )

    return PropertyValuation(
        property_id=property_id,
        initial_investment=initial_investment,
        operating_cash_flows=cash_flows,
        terminal_value=terminal_value,
        total_operating_cash_flow=total_operating_cash_flow,
        net_profit=net_profit,
        roi=roi,
        net_present_value=net_present_value,
        discounted_operating_cash_flows=discounted_cash_flows,
        discounted_terminal_value=discounted_terminal_value,
    )
