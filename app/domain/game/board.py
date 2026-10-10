import json
import os
from typing import List, Optional

class Tile:
    def __init__(self, tile_id: int, name: str):
        self.tile_id: int = tile_id
        self.name: str = name

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self.tile_id}]: {self.name}>"
    
class SpecialTile(Tile):
    def __init__(self, tile_id: int, name: str, tile_type: str):
        super().__init__(tile_id, name)
        self.tile_type: str = tile_type  # 'GO', 'JAIL', 'FREE_PARKING', 'GO_TO_JAIL', 'CHANCE', 'COMMUNITY_CHEST'

class TaxTile(Tile):
    def __init__(self, tile_id: int, name: str, tax_amount: int):
        super().__init__(tile_id, name)
        self.tax_amount: int = tax_amount

class PropertyTile(Tile):
    def __init__(self, tile_id: int, name: str, price: int, rent_values: List[int], group: str, mortgage_value: int):
        super().__init__(tile_id, name)
        self.price: int = price
        self.rent_values: List[int] = rent_values  # [Base, 1 Casa, 2 Casas, 3 Casas, 4 Casas, Hotel]
        self.group: str = group                    
        self.mortgage_value: int = mortgage_value
        
        self.owner_id: Optional[int] = None        
        self.houses: int = 0                       # 0 a 4 = casas, 5 = hotel
        self.is_mortgaged: bool = False

    @property
    def current_rent(self) -> int:
        if self.is_mortgaged:
            return 0
        # Garante que não estoure o índice da lista de aluguéis por segurança
        idx = min(self.houses, len(self.rent_values) - 1)
        return self.rent_values[idx]


class Board:
    def __init__(self, json_path: str = "data/board_v1.json"):
        self.json_path: str = json_path
        self.tiles: List[Tile] = self._initialize_board()
    
    def _initialize_board(self) -> List[Tile]:
        setup: List[Tile] = []
        
        if not os.path.exists(self.json_path):
            # Fallback de segurança: Se o arquivo sumir, cria um tabuleiro mínimo de 2 casas
            return [SpecialTile(0, "GO", "GO"), SpecialTile(1, "Parada Livre", "FREE_PARKING")]

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Ordena os itens do JSON pelo tile_id para garantir a ordem física do tabuleiro
            data.sort(key=lambda x: x["tile_id"])
            
            for item in data:
                t_id = item["tile_id"]
                t_type = item["type"]
                
                if t_type == "PROPERTY":
                    setup.append(PropertyTile(
                        tile_id=t_id,
                        name=item["name"],
                        price=item["price"],
                        rent_values=item["rent_values"], # Passando a lista do JSON direto
                        group=item["group"],
                        mortgage_value=item["mortgage_value"]
                    ))
                elif t_type == "TAX":
                    setup.append(TaxTile(t_id, item["name"], item["tax_amount"]))
                elif t_type == "SPECIAL":
                    setup.append(SpecialTile(t_id, item["name"], item["tile_type"]))
                    
        return setup
    
    def get_tile(self, position: int) -> Tile:
        """Retorna o objeto da casa lidando com o circuito circular dinamicamente."""
        # O truque está aqui: usamos len(self.tiles) em vez de fixar o número 40!
        return self.tiles[position % len(self.tiles)]