
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class PropertyValuation:
    """Estimativas econômicas de uma propriedade em um horizonte definido."""

    property_id: str
    horizon_turns: int
    initial_investment: float
    expected_arrivals_per_turn: float
    expected_net_cash_flow_per_arrival: float
    expected_net_cash_flow_per_turn: float
    expected_operating_cash_flow: float
    terminal_value: float
    expected_net_profit: float
    roi: float | None
    net_present_value: float | None


def _finite_number(name: str, value: float) -> float:
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
    expected_arrivals_per_turn: float,
    expected_net_cash_flow_per_arrival: float,
    horizon_turns: int,
    expected_other_costs_per_turn: float = 0.0,
    terminal_value: float = 0.0,
    discount_rate_per_turn: float | None = None,
) -> PropertyValuation:
    """
    Avalia uma propriedade em um horizonte finito.

    Convenções:
    - expected_arrivals_per_turn é o número esperado de chegadas
      de jogadores à propriedade por turno, não uma probabilidade.
    - expected_net_cash_flow_per_arrival é o fluxo líquido esperado
      para o proprietário por chegada, já considerando a probabilidade
      de recebimento e os efeitos variáveis associados à chegada.
    - expected_other_costs_per_turn são custos adicionais esperados
      por turno, não incluídos no fluxo por chegada.
    - terminal_value é o valor residual esperado no fim do horizonte.
    - discount_rate_per_turn, quando fornecida, deve ser uma taxa
      por turno superior a -100%.

    Supõe fluxo líquido esperado constante por turno. Não modela
    diretamente mudanças de aluguel, construção, propriedade ou
    comportamento dos adversários ao longo do horizonte.
    """
    if not isinstance(property_id, str) or not property_id.strip():
        raise ValueError("property_id deve ser uma string não vazia.")

    if (
        not isinstance(horizon_turns, int)
        or isinstance(horizon_turns, bool)
    ):
        raise TypeError("horizon_turns deve ser um inteiro.")
    if horizon_turns <= 0:
        raise ValueError("horizon_turns deve ser positivo.")

    purchase_price = _finite_number("purchase_price", purchase_price)
    additional_initial_investment = _finite_number(
        "additional_initial_investment",
        additional_initial_investment,
    )
    expected_arrivals_per_turn = _finite_number(
        "expected_arrivals_per_turn",
        expected_arrivals_per_turn,
    )
    expected_net_cash_flow_per_arrival = _finite_number(
        "expected_net_cash_flow_per_arrival",
        expected_net_cash_flow_per_arrival,
    )
    expected_other_costs_per_turn = _finite_number(
        "expected_other_costs_per_turn",
        expected_other_costs_per_turn,
    )
    terminal_value = _finite_number("terminal_value", terminal_value)

    if purchase_price < 0 or additional_initial_investment < 0:
        raise ValueError("Os investimentos iniciais não podem ser negativos.")
    if expected_arrivals_per_turn < 0:
        raise ValueError("expected_arrivals_per_turn não pode ser negativo.")
    if expected_other_costs_per_turn < 0:
        raise ValueError("expected_other_costs_per_turn não pode ser negativo.")
    if terminal_value < 0:
        raise ValueError("terminal_value não pode ser negativo.")

    discount_rate = None
    if discount_rate_per_turn is not None:
        discount_rate = _finite_number(
            "discount_rate_per_turn",
            discount_rate_per_turn,
        )
        if discount_rate <= -1.0:
            raise ValueError("discount_rate_per_turn deve ser maior que -1.")

    initial_investment = purchase_price + additional_initial_investment

    expected_net_cash_flow_per_turn = (
        expected_arrivals_per_turn
        * expected_net_cash_flow_per_arrival
        - expected_other_costs_per_turn
    )

    expected_operating_cash_flow = (
        horizon_turns * expected_net_cash_flow_per_turn
    )

    expected_net_profit = (
        expected_operating_cash_flow
        + terminal_value
        - initial_investment
    )

    roi = (
        expected_net_profit / initial_investment
        if initial_investment > 0
        else None
    )

    net_present_value = None
    if discount_rate is not None:
        discounted_operating_cash_flow = sum(
            expected_net_cash_flow_per_turn / (1.0 + discount_rate) ** turn
            for turn in range(1, horizon_turns + 1)
        )
        discounted_terminal_value = (
            terminal_value / (1.0 + discount_rate) ** horizon_turns
        )

        net_present_value = (
            -initial_investment
            + discounted_operating_cash_flow
            + discounted_terminal_value
        )

    return PropertyValuation(
        property_id=property_id,
        horizon_turns=horizon_turns,
        initial_investment=initial_investment,
        expected_arrivals_per_turn=expected_arrivals_per_turn,
        expected_net_cash_flow_per_arrival=(
            expected_net_cash_flow_per_arrival
        ),
        expected_net_cash_flow_per_turn=expected_net_cash_flow_per_turn,
        expected_operating_cash_flow=expected_operating_cash_flow,
        terminal_value=terminal_value,
        expected_net_profit=expected_net_profit,
        roi=roi,
        net_present_value=net_present_value,
    )
