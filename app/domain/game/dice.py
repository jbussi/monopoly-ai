import random
from typing import Tuple
from typing import Dict, Any

class Dice:
    def __init__(self, sides: int = 6):
        self.sides = sides
    
    def roll(self) -> Tuple[int, int]:
        """
        Rola dois dados de 6 faces, atualiza o histórico e o contador de duplos.
        Retorna os valores dos dois dados.
        """
        die1 = random.randint(1, self.sides)
        die2 = random.randint(1, self.sides)
        
        self.last_roll = (die1, die2)

        if self.is_double():
            self.double_count += 1
        else:
            self.double_count = 0
    
    def is_double(self) -> bool:
        """
        Verifica se o último lançamento foi um duplo.
        Retorna True se for um duplo, caso contrário, False.
        """
        return self.last_roll[0] == self.last_roll[1]
    
    def reset_double_count(self) -> None:
            """Zera o contador de duplos (deve ser chamado ao mudar de turno)."""
            self.double_count = 0
    
    @property
    def total_value(self) -> int:
        """Retorna a soma dos valores dos dois dados do último lançamento."""
        return sum(self.last_roll)
    
    @property
    def speed_jail_condition(self) -> bool:
        """Retorna True se o jogador deve ir para a prisão por causa de três duplos consecutivos."""
        return self.double_count >= 3
    
    @property
    def doubles_remaining_for_jail(self) -> int:
        """
        Retorna quantos duplos faltam para o jogador ir para a prisão.
        Isso será um input crucial (Feature) para o estado da IA.
        """
        return max(0, 3 - self.double_count)

    def get_roll_data_for_log(self) -> Dict[str, Any]:
        """
        Retorna um dicionário com os dados brutos da rodada atual.
        Facilita o armazenamento do histórico do jogo.
        """
        return {
            "die1": self.last_roll[0],
            "die2": self.last_roll[1],
            "total": self.total_value,
            "is_double": self.is_double(),
            "accumulated_doubles": self.double_count,
            "doubles_remaining_for_jail": self.doubles_remaining_for_jail
        }