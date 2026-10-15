
from abc import ABC, abstractmethod



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
