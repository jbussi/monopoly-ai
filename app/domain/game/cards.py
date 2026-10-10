import json
import os
import random
from typing import List

class Card:
    """Representa uma carta de Sorte ou Revés (Chance / Community Chest)."""
    def __init__(self, card_id: int, text: str, effect_type: str, value: int):
        self.card_id: int = card_id
        self.text: str = text
        
        # Tipos de efeito esperados pelo GameLoop: 
        # "MONEY_GAIN" (Ganha dinheiro), "MONEY_LOSS" (Perde dinheiro), 
        # "MOVE_TO" (Vai para casa específica), "GET_JAIL_CARD" (Carta de sair da prisão)
        self.effect_type: str = effect_type
        self.value: int = value

    def __repr__(self):
        return f"<Card [{self.card_id}]: {self.effect_type} ({self.value})>"


class CardDeck:
    """Gerencia o baralho de cartas, cuidando do carregamento via JSON, descarte e embaralhamento."""
    def __init__(self, json_path: str = "data/cards_v1.json"):
        self.json_path: str = json_path
        self.cards: List[Card] = self._load_cards_from_json()
        self.discard_pile: List[Card] = []
        self.shuffle()

    def draw_card(self) -> Card:
        """Puxa uma carta do topo. Se o deck esvaziar, reembaralha o descarte."""
        if not self.cards:
            if not self.discard_pile:
                # Fallback de segurança extrema caso o descarte também suma
                self.cards = self._generate_fallback_cards()
            else:
                self.cards = list(self.discard_pile)
                self.discard_pile.clear()
            self.shuffle()
            
        card = self.cards.pop(0)
        
        # Cartas de "Sair Livre da Prisão" não vão para o descarte imediatamente, 
        # elas ficam retidas no inventário do jogador até serem usadas.
        if card.effect_type != "GET_JAIL_CARD":
            self.discard_pile.append(card)
            
        return card

    def return_jail_card_to_deck(self, card_id: int):
        """Devolve a carta de prisão para o monte de descarte quando o jogador a utiliza."""
        jail_card = Card(card_id, "Saída Livre da Prisão! Guarde esta carta para quando precisar.", "GET_JAIL_CARD", 0)
        self.discard_pile.append(jail_card)

    def shuffle(self):
        """Embaralha o deck de cartas atual."""
        random.shuffle(self.cards)

    def _load_cards_from_json(self) -> List[Card]:
        """Carrega e instancia dinamicamente as cartas a partir do arquivo JSON."""
        cards_list: List[Card] = []
        
        if not os.path.exists(self.json_path):
            return self._generate_fallback_cards()

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                cards_list.append(Card(
                    card_id=item["card_id"],
                    text=item["text"],
                    effect_type=item["effect_type"],
                    value=item["value"]
                ))
                
        return cards_list

    def _generate_fallback_cards(self) -> List[Card]:
        """Gera um deck mínimo de segurança caso o arquivo JSON não seja encontrado."""
        return [
            Card(1, "Sorte do iniciante! Receba $100.", "MONEY_GAIN", 100),
            Card(2, "Taxa de serviços hospitalares. Pague $50.", "MONEY_LOSS", 50)
        ]