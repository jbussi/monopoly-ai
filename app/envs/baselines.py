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
from typing import TYPE_CHECKING

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop

if TYPE_CHECKING:
    from app.domain.agents.protocols import Agent
    Policy = Agent
else:
    # Preserve a lightweight runtime annotation and avoid importing analytics.
    from collections.abc import Callable

    Policy = Callable[[GameLoop], GameAction]


class RandomPolicy:
    """Escolhe uniformemente entre as ações legais."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Seed this policy's random choices for a new episode."""
        self.rng.seed(seed)

    def __call__(self, loop: GameLoop) -> GameAction:
        legal = loop.legal_actions()
        if loop.game.pending_trade is not None:
            reject = next((action for action in legal if action.action_type == ActionType.REJECT_TRADE), None)
            if reject is not None:
                return reject
        return self.rng.choice(legal)


class AggressivePolicy:
    """Prioriza expandir: compra, disputa leilões e desenvolve cedo."""

    def __call__(self, loop: GameLoop) -> GameAction:
        legal = loop.legal_actions()
        if loop.game.pending_trade is not None:
            accept = next((action for action in legal if action.action_type == ActionType.ACCEPT_TRADE), None)
            if accept is not None:
                return accept
        by_kind: dict[str, list[GameAction]] = {}
        for action in legal:
            by_kind.setdefault(action.action_type, []).append(action)
        phase = loop.turn_phase
        if phase == GameLoop.RESOLVE_DEBT:
            for kind in (ActionType.SELL_HOUSE, ActionType.MORTGAGE,
                         ActionType.END_DEBT_NEGOTIATIONS, ActionType.AUCTION_DEBT_PROPERTY):
                if by_kind.get(kind):
                    return by_kind[kind][0]
            return GameAction(ActionType.DECLARE_BANKRUPTCY)
        if phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER:
            if by_kind.get(ActionType.REDEEM_MORTGAGE):
                return by_kind[ActionType.REDEEM_MORTGAGE][0]
            return by_kind[ActionType.KEEP_MORTGAGE][0]
        if phase == GameLoop.AUCTION:
            bids = by_kind.get(ActionType.AUCTION_BID, [])
            if bids:
                return max(bids, key=lambda action: action.payload["amount"])
            return GameAction(ActionType.AUCTION_WITHDRAW)
        if phase == GameLoop.AWAITING_PROPERTY_DECISION:
            if by_kind.get(ActionType.BUY_PROPERTY):
                return GameAction(ActionType.BUY_PROPERTY)
            return GameAction(ActionType.DECLINE_PROPERTY)
        if phase == GameLoop.BEFORE_ROLL and loop.game.player_states[
            loop.current_decision_player_id
        ].is_in_jail:
            if by_kind.get(ActionType.USE_JAIL_CARD):
                return by_kind[ActionType.USE_JAIL_CARD][0]
            if by_kind.get(ActionType.PAY_JAIL_FINE):
                return by_kind[ActionType.PAY_JAIL_FINE][0]
        if by_kind.get(ActionType.BUILD_HOUSE):
            return by_kind[ActionType.BUILD_HOUSE][0]
        if by_kind.get(ActionType.UNMORTGAGE):
            return by_kind[ActionType.UNMORTGAGE][0]
        if by_kind.get(ActionType.ROLL_DICE):
            return GameAction(ActionType.ROLL_DICE)
        return GameAction(ActionType.END_TURN)


class ConservativePolicy:
    """Preserva uma reserva de caixa e só investe quando há folga."""

    def __init__(self, reserve: int = 400) -> None:
        self.reserve = reserve

    def __call__(self, loop: GameLoop) -> GameAction:
        legal = loop.legal_actions()
        if loop.game.pending_trade is not None:
            reject = next((action for action in legal if action.action_type == ActionType.REJECT_TRADE), None)
            if reject is not None:
                return reject
        by_kind: dict[str, list[GameAction]] = {}
        for action in legal:
            by_kind.setdefault(action.action_type, []).append(action)
        player = loop.game.player_states[loop.current_decision_player_id]
        phase = loop.turn_phase
        if phase == GameLoop.RESOLVE_DEBT:
            for kind in (ActionType.SELL_HOUSE, ActionType.MORTGAGE,
                         ActionType.END_DEBT_NEGOTIATIONS, ActionType.AUCTION_DEBT_PROPERTY):
                if by_kind.get(kind):
                    return by_kind[kind][0]
            return GameAction(ActionType.DECLARE_BANKRUPTCY)
        if phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER:
            for action in by_kind.get(ActionType.REDEEM_MORTGAGE, []):
                tile = loop.game.board.get_tile_by_id(action.payload["tile_id"])
                if player.money - loop.unmortgage_cost(tile) >= self.reserve:
                    return action
            return by_kind.get(ActionType.KEEP_MORTGAGE, [GameAction(ActionType.KEEP_MORTGAGE)])[0]
        if phase == GameLoop.AUCTION:
            tile = loop.game.board.get_tile_by_id(loop.pending_tile_id)
            affordable = [
                action for action in by_kind.get(ActionType.AUCTION_BID, [])
                if action.payload["amount"] <= tile.price // 2
                and player.money - action.payload["amount"] >= self.reserve
            ]
            return max(affordable, key=lambda action: action.payload["amount"]) if affordable else GameAction(ActionType.AUCTION_WITHDRAW)
        if phase == GameLoop.AWAITING_PROPERTY_DECISION:
            tile = loop.game.board.get_tile_by_id(loop.pending_tile_id)
            if by_kind.get(ActionType.BUY_PROPERTY) and player.money - tile.price >= self.reserve:
                return GameAction(ActionType.BUY_PROPERTY)
            return GameAction(ActionType.DECLINE_PROPERTY)
        if phase == GameLoop.BEFORE_ROLL and player.is_in_jail:
            if by_kind.get(ActionType.USE_JAIL_CARD):
                return by_kind[ActionType.USE_JAIL_CARD][0]
            if by_kind.get(ActionType.PAY_JAIL_FINE) and player.money - loop.rules.jail_fine >= self.reserve:
                return GameAction(ActionType.PAY_JAIL_FINE)
        for action in by_kind.get(ActionType.BUILD_HOUSE, []):
            tile = loop.game.board.get_tile_by_id(action.payload["tile_id"])
            if player.money - tile.house_cost >= self.reserve:
                return action
        for action in by_kind.get(ActionType.UNMORTGAGE, []):
            tile = loop.game.board.get_tile_by_id(action.payload["tile_id"])
            if player.money - loop.unmortgage_cost(tile) >= self.reserve:
                return action
        if by_kind.get(ActionType.ROLL_DICE):
            return GameAction(ActionType.ROLL_DICE)
        return GameAction(ActionType.END_TURN)


class GreedyPolicy:
    """Heurística simples: compra mantendo uma reserva, dá lances até o preço de tabela,
    constrói com monopólio, sai da prisão pagando e, em dívida, hipoteca antes de vender."""

    def __init__(self, reserve: int = 150) -> None:
        self.reserve = reserve

    def __call__(self, loop: GameLoop) -> GameAction:
        if loop.game.pending_trade is not None:
            return GameAction(ActionType.REJECT_TRADE)
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
            if ActionType.END_DEBT_NEGOTIATIONS in by_kind:
                return by_kind[ActionType.END_DEBT_NEGOTIATIONS][0]
            if ActionType.AUCTION_DEBT_PROPERTY in by_kind:
                return by_kind[ActionType.AUCTION_DEBT_PROPERTY][0]
            return GameAction(ActionType.DECLARE_BANKRUPTCY)

        if phase == GameLoop.RESOLVE_MORTGAGE_TRANSFER:
            for action in by_kind.get(ActionType.REDEEM_MORTGAGE, []):
                tile = game.board.get_tile_by_id(action.payload["tile_id"])
                if state.money - loop.unmortgage_cost(tile) >= self.reserve:
                    return action
            return by_kind[ActionType.KEEP_MORTGAGE][0]

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
