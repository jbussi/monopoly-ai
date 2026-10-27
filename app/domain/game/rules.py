
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Rules(ABC):
    """Contrato para variantes das regras de uma partida."""

    @property
    @abstractmethod
    def starting_money(self) -> int:
        """Saldo inicial dos participantes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def go_salary(self) -> int:
        """Valor recebido ao passar pelo início."""
        raise NotImplementedError

    @property
    @abstractmethod
    def jail_fine(self) -> int:
        """Valor pago para sair da prisão."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_consecutive_doubles(self) -> int:
        """Limite de duplas consecutivas."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_jail_attempts(self) -> int:
        """Limite de tentativas para sair da prisão."""
        raise NotImplementedError

    @abstractmethod
    def is_game_over(self, active_players: int) -> bool:
        """Determina se a partida terminou."""
        raise NotImplementedError

    @abstractmethod
    def auction_after_declined_purchase(self) -> bool:
        """Determina se a propriedade recusada vai a leilão."""
        raise NotImplementedError

    @property
    def jail_position(self) -> int:
        """Posição da prisão no tabuleiro."""
        return 10

    @property
    def max_houses_in_bank(self) -> int:
        return 32

    @property
    def max_hotels_in_bank(self) -> int:
        return 12

    @property
    def auction_min_bid(self) -> int:
        return 1

    @property
    def income_tax_choice(self) -> bool:
        """Se True, o imposto percentual é aplicado quando for menor que o fixo."""
        return False

    @property
    def max_turns(self) -> int | None:
        """Limite de turnos antes de truncar a partida (None = sem limite)."""
        return None


class ClassicRules(Rules):
    """Implementa as regras tradicionais adotadas nesta versão."""

    @property
    def starting_money(self) -> int:
        return 1500

    @property
    def go_salary(self) -> int:
        return 200

    @property
    def jail_fine(self) -> int:
        return 50

    @property
    def max_consecutive_doubles(self) -> int:
        return 3

    @property
    def max_jail_attempts(self) -> int:
        return 3

    def is_game_over(self, active_players: int) -> bool:
        return active_players <= 1

    def auction_after_declined_purchase(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ConfiguredRules(Rules):
    """Regras parametrizadas (montadas a partir do settings.json da variante)."""

    starting_money_value: int = 1500
    go_salary_value: int = 200
    jail_fine_value: int = 50
    max_consecutive_doubles_value: int = 3
    max_jail_attempts_value: int = 3
    auction_enabled: bool = True
    jail_position_value: int = 10
    max_houses_value: int = 32
    max_hotels_value: int = 12
    auction_min_bid_value: int = 1
    income_tax_choice_value: bool = False
    max_turns_value: int | None = None

    @property
    def starting_money(self) -> int:
        return self.starting_money_value

    @property
    def go_salary(self) -> int:
        return self.go_salary_value

    @property
    def jail_fine(self) -> int:
        return self.jail_fine_value

    @property
    def max_consecutive_doubles(self) -> int:
        return self.max_consecutive_doubles_value

    @property
    def max_jail_attempts(self) -> int:
        return self.max_jail_attempts_value

    def is_game_over(self, active_players: int) -> bool:
        return active_players <= 1

    def auction_after_declined_purchase(self) -> bool:
        return self.auction_enabled

    @property
    def jail_position(self) -> int:
        return self.jail_position_value

    @property
    def max_houses_in_bank(self) -> int:
        return self.max_houses_value

    @property
    def max_hotels_in_bank(self) -> int:
        return self.max_hotels_value

    @property
    def auction_min_bid(self) -> int:
        return self.auction_min_bid_value

    @property
    def income_tax_choice(self) -> bool:
        return self.income_tax_choice_value

    @property
    def max_turns(self) -> int | None:
        return self.max_turns_value
