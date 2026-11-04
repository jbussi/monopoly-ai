
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
    def go_to_jail_position(self) -> int:
        """Posição que envia o jogador diretamente à prisão."""
        return 30

    @property
    def max_houses_in_bank(self) -> int:
        """Return the total house pieces available in the bank."""
        return 32

    @property
    def max_hotels_in_bank(self) -> int:
        """Return the total hotel pieces available in the bank."""
        return 12

    @property
    def auction_min_bid(self) -> int:
        """Return the minimum opening bid for a property auction."""
        return 1

    @property
    def income_tax_choice(self) -> bool:
        """Se True, o imposto percentual é aplicado quando for menor que o fixo."""
        return False

    @property
    def max_turns(self) -> int | None:
        """Limite de turnos antes de truncar a partida (None = sem limite)."""
        return None

    @property
    def trade_enabled(self) -> bool:
        """Indicate whether player-to-player negotiations are enabled."""
        return False

    @property
    def trade_max_counteroffers(self) -> int:
        """Return the maximum counteroffers allowed after an initial offer."""
        return 3

    @property
    def debt_resolution_enabled(self) -> bool:
        """Indicate whether the extended debt-resolution flow is enabled."""
        return False

    @property
    def debt_negotiations_per_player(self) -> int:
        """Return how many distinct creditors a debtor may negotiate with."""
        return 3

    @property
    def bankruptcy_to_bank(self) -> bool:
        """Whether residual bankruptcy assets return to the bank for auction."""
        return False


class ClassicRules(Rules):
    """Implementa as regras tradicionais adotadas nesta versão."""

    @property
    def starting_money(self) -> int:
        """Return the starting cash granted to each player."""
        return 1500

    @property
    def go_salary(self) -> int:
        """Return the salary collected for passing or landing on Go."""
        return 200

    @property
    def jail_fine(self) -> int:
        """Return the fine for leaving jail by payment."""
        return 50

    @property
    def max_consecutive_doubles(self) -> int:
        """Return the doubles limit before a player is sent to jail."""
        return 3

    @property
    def max_jail_attempts(self) -> int:
        """Return how many turns a player may try doubles before paying."""
        return 3

    def is_game_over(self, active_players: int) -> bool:
        """Return whether the active-player count leaves at most one survivor."""
        return active_players <= 1

    def auction_after_declined_purchase(self) -> bool:
        """Return whether a declined property proceeds to auction."""
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
    go_to_jail_position_value: int = 30
    jail_position_value: int = 10
    max_houses_value: int = 32
    max_hotels_value: int = 12
    auction_min_bid_value: int = 1
    income_tax_choice_value: bool = False
    max_turns_value: int | None = None
    trade_enabled_value: bool = True
    trade_max_counteroffers_value: int = 3
    debt_resolution_enabled_value: bool = True
    debt_negotiations_per_player_value: int = 3
    bankruptcy_to_bank_value: bool = False

    @property
    def starting_money(self) -> int:
        """Return configured starting cash for each player."""
        return self.starting_money_value

    @property
    def go_salary(self) -> int:
        """Return configured salary for passing or landing on Go."""
        return self.go_salary_value

    @property
    def jail_fine(self) -> int:
        """Return configured fine for leaving jail by payment."""
        return self.jail_fine_value

    @property
    def max_consecutive_doubles(self) -> int:
        """Return the configured doubles limit before jail."""
        return self.max_consecutive_doubles_value

    @property
    def max_jail_attempts(self) -> int:
        """Return the configured number of jail doubles attempts."""
        return self.max_jail_attempts_value

    def is_game_over(self, active_players: int) -> bool:
        """Return whether at most one player remains active."""
        return active_players <= 1

    def auction_after_declined_purchase(self) -> bool:
        """Return the configured auction flag for declined properties."""
        return self.auction_enabled

    @property
    def jail_position(self) -> int:
        """Return the configured board position used for the jail space."""
        return self.jail_position_value

    @property
    def go_to_jail_position(self) -> int:
        """Return the configured board position that sends players to jail."""
        return self.go_to_jail_position_value

    @property
    def max_houses_in_bank(self) -> int:
        """Return the configured bank supply of house pieces."""
        return self.max_houses_value

    @property
    def max_hotels_in_bank(self) -> int:
        """Return the configured bank supply of hotel pieces."""
        return self.max_hotels_value

    @property
    def auction_min_bid(self) -> int:
        """Return the configured minimum opening auction bid."""
        return self.auction_min_bid_value

    @property
    def income_tax_choice(self) -> bool:
        """Return whether income tax uses the configured lower-payment choice."""
        return self.income_tax_choice_value

    @property
    def trade_enabled(self) -> bool:
        """Return whether player-to-player negotiations are enabled."""
        return self.trade_enabled_value

    @property
    def trade_max_counteroffers(self) -> int:
        """Return the configured limit on trade counteroffers."""
        return self.trade_max_counteroffers_value

    @property
    def debt_resolution_enabled(self) -> bool:
        """Return whether the extended debt-resolution flow is enabled."""
        return self.debt_resolution_enabled_value

    @property
    def debt_negotiations_per_player(self) -> int:
        """Return how many distinct creditors a debtor may negotiate with."""
        return self.debt_negotiations_per_player_value

    @property
    def bankruptcy_to_bank(self) -> bool:
        """Return whether residual bankruptcy assets are returned to the bank."""
        return self.bankruptcy_to_bank_value

    @property
    def max_turns(self) -> int | None:
        """Return the configured turn cap, or None when no cap is set."""
        return self.max_turns_value
