from typing import List

class Player:
    def __init__(self, player_id: int, name: str, starting_money: int = 1500):
        self.player_id: int = player_id
        self.name: str = name
        self.money: int = starting_money
        self.position: int = 0
        
        # Estados de Restrição
        self.is_in_jail: bool = False
        self.jail_turns: int = 0
        self.is_bankrupt: bool = False
        
        # Inventário
        self.properties: List[int] = []  # Armazena os tile_ids conquistados
        self.has_jail_card: int = 0      # Quantidade de cartas de "Saída Livre da Prisão"

    def pay_money(self, amount: int):
        self.money -= amount

    def receive_money(self, amount: int):
        self.money += amount

    def buy_property(self, tile_id: int, price: int):
        self.pay_money(price)
        self.properties.append(tile_id)

    def reset_assets_on_bankruptcy(self):
        """Limpa o inventário caso o jogador quebre."""
        self.money = 0
        self.properties.clear()
        self.has_jail_card = 0