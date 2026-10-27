"""Políticas de referência para avaliação e como oponentes fixos.

Uma política é qualquer callable `policy(loop) -> GameAction` chamado quando é a vez
dela (`loop.current_decision_player_id`). Pode ter `reset(seed)`, que os ambientes
chamam a cada episódio para oponentes estocásticos reprodutíveis.

Para usar um modelo treinado como oponente:

    def policy(loop):
        obs = encoder.encode(loop, loop.current_decision_player_id)
        index = model.act(obs, action_space.mask(loop))
        return action_space.to_game_action(index, loop)
"""

import random
from collections.abc import Callable

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop

Policy = Callable[[GameLoop], GameAction]


class RandomPolicy:
    """Escolhe uniformemente entre as ações legais."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng.seed(seed)

    def __call__(self, loop: GameLoop) -> GameAction:
        return self.rng.choice(loop.legal_actions())


class GreedyPolicy:
    """Heurística simples: compra mantendo uma reserva, dá lances até o preço de tabela,
    constrói com monopólio, sai da prisão pagando e, em dívida, hipoteca antes de vender."""

    def __init__(self, reserve: int = 150) -> None:
        self.reserve = reserve

    def __call__(self, loop: GameLoop) -> GameAction:
        legal = loop.legal_actions()
        by_kind: dict[str, list[GameAction]] = {}
        for action in legal:
            by_kind.setdefault(action.action_type, []).append(action)

        game = loop.game
        player_id = loop.current_decision_player_id
        state = game.player_states[player_id]
        phase = loop.turn_phase

        if phase == GameLoop.RESOLVE_DEBT:
            for kind in (ActionType.MORTGAGE, ActionType.SELL_HOUSE):
                if kind in by_kind:
                    return by_kind[kind][0]
            return GameAction(ActionType.DECLARE_BANKRUPTCY)

        if phase == GameLoop.AUCTION:
            tile = game.board.get_tile_by_id(loop.pending_tile_id)
            for bid in by_kind.get(ActionType.AUCTION_BID, []):
                amount = bid.payload["amount"]
                if amount <= tile.price and state.money - amount >= self.reserve:
                    return bid
            return GameAction(ActionType.AUCTION_WITHDRAW)

        if phase == GameLoop.AWAITING_PROPERTY_DECISION:
            tile = game.board.get_tile_by_id(loop.pending_tile_id)
            if ActionType.BUY_PROPERTY in by_kind and state.money - tile.price >= self.reserve:
                return GameAction(ActionType.BUY_PROPERTY)
            return GameAction(ActionType.DECLINE_PROPERTY)

        if phase == GameLoop.BEFORE_ROLL and state.is_in_jail:
            if ActionType.USE_JAIL_CARD in by_kind:
                return GameAction(ActionType.USE_JAIL_CARD)
            if ActionType.PAY_JAIL_FINE in by_kind and state.money - loop.rules.jail_fine >= self.reserve:
                return GameAction(ActionType.PAY_JAIL_FINE)

        for action in by_kind.get(ActionType.UNMORTGAGE, []):
            tile = game.board.get_tile_by_id(action.payload["tile_id"])
            if state.money - loop.unmortgage_cost(tile) >= 2 * self.reserve:
                return action
        for action in by_kind.get(ActionType.BUILD_HOUSE, []):
            tile = game.board.get_tile_by_id(action.payload["tile_id"])
            if state.money - tile.house_cost >= self.reserve:
                return action

        if phase == GameLoop.BEFORE_ROLL:
            return GameAction(ActionType.ROLL_DICE)
        return GameAction(ActionType.END_TURN)
