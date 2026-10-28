"""Structured action contract for the full Monopoly engine (monopoly_v2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop, MANAGEMENT_ACTIONS


ACTION_TYPES: tuple[str, ...] = (
    ActionType.ROLL_DICE,
    ActionType.BUY_PROPERTY,
    ActionType.DECLINE_PROPERTY,
    ActionType.END_TURN,
    ActionType.PAY_JAIL_FINE,
    ActionType.USE_JAIL_CARD,
    ActionType.DECLARE_BANKRUPTCY,
    ActionType.AUCTION_WITHDRAW,
    ActionType.AUCTION_BID,
    ActionType.BUILD_HOUSE,
    ActionType.SELL_HOUSE,
    ActionType.MORTGAGE,
    ActionType.UNMORTGAGE,
    ActionType.PROPOSE_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.COUNTER_TRADE,
    ActionType.END_DEBT_NEGOTIATIONS,
    ActionType.AUCTION_DEBT_PROPERTY,
    ActionType.KEEP_MORTGAGE,
    ActionType.REDEEM_MORTGAGE,
)
ACTION_TYPE_TO_INDEX = {kind: index for index, kind in enumerate(ACTION_TYPES)}
INDEX_TO_ACTION_TYPE = dict(enumerate(ACTION_TYPES))
PROPERTY_ACTION_TYPES = set(MANAGEMENT_ACTIONS) | {
    ActionType.AUCTION_DEBT_PROPERTY,
    ActionType.KEEP_MORTGAGE,
    ActionType.REDEEM_MORTGAGE,
}
INT32_MAX = np.iinfo(np.int32).max


class StructuredActionDict(spaces.Dict):
    """Dict action space whose sample uses the current engine's legal actions."""

    def __init__(self, spaces_by_key, sampler, seed=None):
        super().__init__(spaces_by_key, seed=seed)
        self._legal_sampler = sampler

    def sample(self, mask=None, probability=None):
        # PettingZoo's AEC checker calls action_space.sample(observation_mask).
        # The v2 mask is conditional and has numeric bounds, so it is not the same
        # shape as this Dict. Sampling a legal engine action is both safe and valid.
        return self._legal_sampler(self.np_random)


def _scalar(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Campos escalares da ação devem ter exatamente um valor.")
    return int(array.reshape(-1)[0])


class StructuredActionSpace:
    """Maps one fixed-shape structured action to a validated ``GameAction``.

    Numeric ranges are represented by conditional lower/upper bounds in the action
    mask, rather than by enumerating every possible dollar amount. Offer bundles
    use fixed-size bit vectors, so any valid combination of assets is representable.
    """

    def __init__(
        self,
        loop: GameLoop,
        max_players: int = 12,
        max_jail_cards: int = 2,
        loop_provider=None,
    ):
        self._loop_provider = loop_provider or (lambda: loop)
        self.tile_ids = tuple(tile.tile_id for tile in loop.game.board.property_tiles)
        self.property_index = {tile_id: index for index, tile_id in enumerate(self.tile_ids)}
        self.board_size = loop.game.board.size
        self.max_players = max_players
        self.jail_card_ids = self._get_jail_card_ids(loop, max_jail_cards)
        self.jail_card_index = {card_id: index for index, card_id in enumerate(self.jail_card_ids)}

        action_fields = {
            "action_type": spaces.Discrete(len(ACTION_TYPES)),
            "target_player": spaces.Discrete(max_players + 1),  # max_players is the unused sentinel
            "tile_id": spaces.Discrete(self.board_size + 1),    # board_size is the unused sentinel
            "amount": spaces.Box(0, INT32_MAX, shape=(1,), dtype=np.int32),
            "money_offered": spaces.Box(0, INT32_MAX, shape=(1,), dtype=np.int32),
            "money_requested": spaces.Box(0, INT32_MAX, shape=(1,), dtype=np.int32),
            "offered_properties": spaces.MultiBinary(len(self.tile_ids)),
            "requested_properties": spaces.MultiBinary(len(self.tile_ids)),
            "offered_jail_cards": spaces.MultiBinary(len(self.jail_card_ids)),
            "requested_jail_cards": spaces.MultiBinary(len(self.jail_card_ids)),
        }
        self.action_space = StructuredActionDict(action_fields, self._sample_legal_action)

        self.mask_space = spaces.Dict({
            "action_type": spaces.MultiBinary(len(ACTION_TYPES)),
            "target_player": spaces.MultiBinary(max_players + 1),
            "tile_id": spaces.MultiBinary(self.board_size + 1),
            "offered_properties": spaces.MultiBinary(len(self.tile_ids)),
            "requested_properties": spaces.MultiBinary(len(self.tile_ids)),
            "offered_jail_cards": spaces.MultiBinary(len(self.jail_card_ids)),
            "requested_jail_cards": spaces.MultiBinary(len(self.jail_card_ids)),
            # Rows: auction amount, cash offered, cash requested. Columns: inclusive min/max.
            "money_bounds": spaces.Box(0, INT32_MAX, shape=(3, 2), dtype=np.int32),
        })

    @staticmethod
    def _get_jail_card_ids(loop: GameLoop, minimum: int) -> tuple[str, ...]:
        card_ids = []
        for deck_name, deck in loop.game.decks.items():
            for card in (*deck._cards, *deck._discard_pile):
                if card.action == "get_out_of_jail":
                    card_ids.append(f"{deck_name}:{card.id}")
        # A card in a player's hand is absent from both deck piles.
        for player_state in loop.game.player_states.values():
            for deck_name, card in player_state.jail_cards:
                card_ids.append(f"{deck_name}:{card.id}")
        return tuple(sorted(set(card_ids))) or tuple(f"__jail_card_{i}" for i in range(minimum))

    def _ids_to_bits(self, identifiers: tuple[str, ...] | list[str], index: Mapping) -> np.ndarray:
        bits = np.zeros(len(index), dtype=np.int8)
        for identifier in identifiers:
            if identifier not in index:
                raise ValueError(f"Item não reconhecido no espaço v2: {identifier!r}.")
            bits[index[identifier]] = 1
        return bits

    def to_game_action(self, action: Mapping[str, Any], loop: GameLoop) -> GameAction:
        kind_index = _scalar(action["action_type"])
        try:
            kind = INDEX_TO_ACTION_TYPE[kind_index]
        except KeyError as exc:
            raise ValueError(f"Tipo de ação fora do espaço v2: {kind_index}.") from exc

        payload: dict[str, Any] = {}
        if kind in PROPERTY_ACTION_TYPES:
            tile_id = _scalar(action["tile_id"])
            if tile_id >= self.board_size:
                raise ValueError("Informe um tile_id válido para essa ação.")
            payload["tile_id"] = tile_id
        if kind == ActionType.AUCTION_BID:
            payload["amount"] = _scalar(action["amount"])
        if kind in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE):
            if kind == ActionType.PROPOSE_TRADE:
                target_index = _scalar(action["target_player"])
                if target_index >= len(loop.game.turn_order) or target_index == 0:
                    raise ValueError("Destinatário fora da partida.")
                actor_seat = loop.game.turn_order.index(loop.current_decision_player_id)
                payload["target_player_id"] = loop.game.turn_order[
                    (actor_seat + target_index) % len(loop.game.turn_order)
                ]
            payload.update({
                "money_offered": _scalar(action["money_offered"]),
                "money_requested": _scalar(action["money_requested"]),
                "properties_offered": [
                    self.tile_ids[i] for i, selected in enumerate(action["offered_properties"])
                    if selected
                ],
                "properties_requested": [
                    self.tile_ids[i] for i, selected in enumerate(action["requested_properties"])
                    if selected
                ],
                "jail_cards_offered": [
                    self.jail_card_ids[i] for i, selected in enumerate(action["offered_jail_cards"])
                    if selected
                ],
                "jail_cards_requested": [
                    self.jail_card_ids[i] for i, selected in enumerate(action["requested_jail_cards"])
                    if selected
                ],
            })
        return GameAction(kind, payload)

    def from_game_action(self, game_action: GameAction, loop: GameLoop) -> dict[str, np.ndarray | int]:
        action = self.empty_action()
        action["action_type"] = ACTION_TYPE_TO_INDEX[game_action.action_type]
        if "tile_id" in game_action.payload:
            action["tile_id"] = game_action.payload["tile_id"]
        if "amount" in game_action.payload:
            action["amount"][0] = game_action.payload["amount"]
        if "target_player_id" in game_action.payload:
            actor_seat = loop.game.turn_order.index(loop.current_decision_player_id)
            target_seat = loop.game.turn_order.index(game_action.payload["target_player_id"])
            action["target_player"] = (target_seat - actor_seat) % len(loop.game.turn_order)
        for key, index in (
            ("properties_offered", self.property_index),
            ("properties_requested", self.property_index),
            ("jail_cards_offered", self.jail_card_index),
            ("jail_cards_requested", self.jail_card_index),
        ):
            if key in game_action.payload:
                action[key] = self._ids_to_bits(game_action.payload[key], index)
        for key in ("money_offered", "money_requested"):
            if key in game_action.payload:
                action[key][0] = game_action.payload[key]
        return action

    def empty_action(self) -> dict[str, Any]:
        return {
            "action_type": 0,
            "target_player": self.max_players,
            "tile_id": self.board_size,
            "amount": np.zeros(1, dtype=np.int32),
            "money_offered": np.zeros(1, dtype=np.int32),
            "money_requested": np.zeros(1, dtype=np.int32),
            "offered_properties": np.zeros(len(self.tile_ids), dtype=np.int8),
            "requested_properties": np.zeros(len(self.tile_ids), dtype=np.int8),
            "offered_jail_cards": np.zeros(len(self.jail_card_ids), dtype=np.int8),
            "requested_jail_cards": np.zeros(len(self.jail_card_ids), dtype=np.int8),
        }

    def _sample_legal_action(self, rng) -> dict[str, Any]:
        loop = self._loop_provider()
        if loop.game.pending_trade is not None:
            # The counteroffer action requires terms; accept/reject are complete
            # valid actions and suffice for generic space samplers/checkers.
            candidates = [
                GameAction(ActionType.ACCEPT_TRADE),
                GameAction(ActionType.REJECT_TRADE),
            ]
        else:
            candidates = loop.legal_actions()
        if not candidates:
            raise RuntimeError(
                f"Nenhuma ação legal para amostrar na fase {loop.turn_phase}."
            )
        selected = candidates[int(rng.integers(len(candidates)))]
        return self.from_game_action(selected, loop)

    def action_mask(
        self,
        loop: GameLoop,
        *,
        action_type: str | int | None = None,
        target_player_index: int | None = None,
        money_offered: int | None = None,
        money_requested: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return legality masks and inclusive integer bounds for the selected branch.

        ``action_type`` and ``target_player_index`` let an autoregressive policy
        request masks after it has selected earlier fields of the action.
        """
        mask = {
            "action_type": np.zeros(len(ACTION_TYPES), dtype=np.int8),
            "target_player": np.zeros(self.max_players + 1, dtype=np.int8),
            "tile_id": np.zeros(self.board_size + 1, dtype=np.int8),
            "offered_properties": np.zeros(len(self.tile_ids), dtype=np.int8),
            "requested_properties": np.zeros(len(self.tile_ids), dtype=np.int8),
            "offered_jail_cards": np.zeros(len(self.jail_card_ids), dtype=np.int8),
            "requested_jail_cards": np.zeros(len(self.jail_card_ids), dtype=np.int8),
            "money_bounds": np.zeros((3, 2), dtype=np.int32),
        }
        if loop.game.game_over:
            return mask

        if isinstance(action_type, (int, np.integer)):
            try:
                action_type = INDEX_TO_ACTION_TYPE[int(action_type)]
            except KeyError as exc:
                raise ValueError(f"Tipo de ação fora do espaço v2: {action_type}.") from exc
        actor = loop.current_decision_player_id
        legal = loop.legal_actions()
        for game_action in legal:
            kind = game_action.action_type
            mask["action_type"][ACTION_TYPE_TO_INDEX[kind]] = 1
            if "tile_id" in game_action.payload and (action_type is None or kind == action_type):
                tile_id = game_action.payload["tile_id"]
                if tile_id in self.property_index:
                    mask["tile_id"][tile_id] = 1
            elif kind == ActionType.BUY_PROPERTY and (action_type is None or kind == action_type) and loop.pending_tile_id is not None:
                mask["tile_id"][loop.pending_tile_id] = 1
            elif kind == ActionType.DECLINE_PROPERTY and (action_type is None or kind == action_type) and loop.pending_tile_id is not None:
                mask["tile_id"][loop.pending_tile_id] = 1

        trade_actions = set(loop.available_trade_actions())
        for kind in trade_actions:
            mask["action_type"][ACTION_TYPE_TO_INDEX[kind]] = 1

        if action_type is None:
            return self._fill_union_parameter_masks(mask, loop, actor, trade_actions)
        if action_type not in ACTION_TYPE_TO_INDEX or not mask["action_type"][ACTION_TYPE_TO_INDEX[action_type]]:
            return mask

        if action_type == ActionType.AUCTION_BID and mask["action_type"][ACTION_TYPE_TO_INDEX[action_type]]:
            state = loop.game.player_states[actor]
            minimum = max(loop.rules.auction_min_bid, loop.auction_highest_bid + 1)
            maximum = state.money
            if maximum >= minimum:
                mask["money_bounds"][0] = (minimum, maximum)

        if action_type == ActionType.PROPOSE_TRADE:
            targets = loop.debt_negotiation_targets() if loop.game.debts else tuple(
                player_id for player_id in loop.game.active_player_ids if player_id != actor
            )
            self._mask_targets(mask, loop, actor, targets)
        elif action_type == ActionType.COUNTER_TRADE and loop.game.pending_trade is not None:
            target = loop.game.pending_trade.current_offer.proposer_id
            self._mask_targets(mask, loop, actor, (target,))

        target_id = self._target_id(loop, action_type, target_player_index)
        if action_type in (ActionType.PROPOSE_TRADE, ActionType.COUNTER_TRADE) and target_id is not None:
            actor_state = loop.game.player_states[actor]
            target_state = loop.game.player_states[target_id]
            cash_offer = int(actor_state.money)
            cash_request = int(target_state.money)
            if loop.game.debts:
                debtor = loop.game.debts[0].debtor_id
                if actor == debtor:
                    mask["money_bounds"][1] = (0, cash_offer)
                    minimum_request = max(0, (money_offered or 0) + 1)
                    if cash_request >= minimum_request:
                        mask["money_bounds"][2] = (minimum_request, cash_request)
                else:
                    minimum_offer = max(0, (money_requested or 0) + 1)
                    mask["money_bounds"][1] = (minimum_offer, cash_offer) if cash_offer >= minimum_offer else (0, 0)
                    mask["money_bounds"][2] = (0, cash_request)
            else:
                mask["money_bounds"][1] = (0, cash_offer)
                mask["money_bounds"][2] = (0, cash_request)
            self._mask_trade_assets(mask, loop, actor, target_id)

        return mask

    def _fill_union_parameter_masks(self, mask, loop, actor, trade_actions):
        if ActionType.AUCTION_BID in {
            kind for kind, index in ACTION_TYPE_TO_INDEX.items() if mask["action_type"][index]
        }:
            minimum = max(loop.rules.auction_min_bid, loop.auction_highest_bid + 1)
            maximum = loop.game.player_states[actor].money
            if maximum >= minimum:
                mask["money_bounds"][0] = (minimum, maximum)
        if ActionType.PROPOSE_TRADE in trade_actions:
            targets = loop.debt_negotiation_targets() if loop.game.debts else tuple(
                player_id for player_id in loop.game.active_player_ids if player_id != actor
            )
            self._mask_targets(mask, loop, actor, targets)
            if targets:
                actor_money = loop.game.player_states[actor].money
                target_money = max(loop.game.player_states[player_id].money for player_id in targets)
                if loop.game.debts:
                    mask["money_bounds"][1] = (0, actor_money)
                    mask["money_bounds"][2] = (1, target_money) if target_money else (0, 0)
                else:
                    mask["money_bounds"][1] = (0, actor_money)
                    mask["money_bounds"][2] = (0, target_money)
            for target_id in targets:
                self._mask_trade_assets(mask, loop, actor, target_id)
        elif ActionType.COUNTER_TRADE in trade_actions and loop.game.pending_trade is not None:
            target_id = loop.game.pending_trade.current_offer.proposer_id
            self._mask_targets(mask, loop, actor, (target_id,))
            actor_money = loop.game.player_states[actor].money
            target_money = loop.game.player_states[target_id].money
            if loop.game.debts and actor != loop.game.debts[0].debtor_id:
                mask["money_bounds"][1] = (1, actor_money) if actor_money else (0, 0)
                mask["money_bounds"][2] = (0, target_money)
            else:
                mask["money_bounds"][1] = (0, actor_money)
                mask["money_bounds"][2] = (1, target_money) if loop.game.debts and target_money else (0, target_money)
            self._mask_trade_assets(mask, loop, actor, target_id)
        return mask

    def _mask_targets(self, mask, loop, actor, player_ids) -> None:
        actor_seat = loop.game.turn_order.index(actor)
        for player_id in player_ids:
            if player_id in loop.game.turn_order:
                target_seat = loop.game.turn_order.index(player_id)
                relative_seat = (target_seat - actor_seat) % len(loop.game.turn_order)
                if relative_seat:
                    mask["target_player"][relative_seat] = 1

    def _target_id(self, loop, action_type, target_player_index):
        if action_type == ActionType.COUNTER_TRADE and loop.game.pending_trade is not None:
            return loop.game.pending_trade.current_offer.proposer_id
        if target_player_index is None or not 0 <= target_player_index < len(loop.game.turn_order):
            return None
        if target_player_index == 0:
            return None
        actor_seat = loop.game.turn_order.index(loop.current_decision_player_id)
        return loop.game.turn_order[
            (actor_seat + target_player_index) % len(loop.game.turn_order)
        ]

    def _mask_trade_assets(self, mask, loop, actor, target) -> None:
        for i, tile_id in enumerate(self.tile_ids):
            tile = loop.game.board.get_tile_by_id(tile_id)
            group = loop.game.board.group_members(tile.group)
            group_is_clear = not any(loop.game.property_states[item.tile_id].houses for item in group)
            owner = loop.game.property_states[tile_id].owner_id
            if group_is_clear and owner == actor:
                mask["offered_properties"][i] = 1
            if group_is_clear and owner == target:
                mask["requested_properties"][i] = 1

        for player_id, key in ((actor, "offered_jail_cards"), (target, "requested_jail_cards")):
            held = {
                f"{deck_name}:{card.id}"
                for deck_name, card in loop.game.player_states[player_id].jail_cards
            }
            for card_id, i in self.jail_card_index.items():
                if card_id in held:
                    mask[key][i] = 1
