"""Bounded, legal candidate generation for structured trade decisions."""

from __future__ import annotations

import random
from collections.abc import Iterable
from itertools import combinations
from typing import TypeAlias
from uuid import UUID

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop

Asset: TypeAlias = tuple[str, int | str]


class TradeCandidateProvider:
    """Generate a reproducible, bounded set of valid trade offers.

    The provider covers cash-only offers, single-asset and sampled multi-asset
    bundles on either side, and asset-for-asset exchanges. It deliberately
    samples the combinatorial bundle space instead of enumerating its powerset.
    Every returned action is checked by ``GameLoop.validate``.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 64,
        max_candidates_per_family: int = 12,
        bundle_samples_per_size: int = 6,
        max_bundle_size: int = 3,
        seed: int | None = None,
    ) -> None:
        for value, name in (
            (max_candidates, "max_candidates"),
            (max_candidates_per_family, "max_candidates_per_family"),
            (bundle_samples_per_size, "bundle_samples_per_size"),
            (max_bundle_size, "max_bundle_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} deve ser um inteiro positivo.")
        self.max_candidates = max_candidates
        self.max_candidates_per_family = max_candidates_per_family
        self.bundle_samples_per_size = bundle_samples_per_size
        self.max_bundle_size = max_bundle_size
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset bundle sampling so the next candidate set is reproducible."""
        self.rng.seed(seed)

    def __call__(self, loop: GameLoop, *, limit: int | None = None) -> tuple[GameAction, ...]:
        budget = self.max_candidates if limit is None else min(self.max_candidates, limit)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("limit deve ser um inteiro não negativo.")
        if budget == 0:
            return ()

        trade_actions = set(loop.available_trade_actions())
        if not trade_actions:
            return ()

        actor = loop.current_decision_player_id
        if loop.game.pending_trade is not None:
            offer = loop.game.pending_trade.current_offer
            targets = (offer.proposer_id,)
            action_kind = (
                ActionType.COUNTER_TRADE
                if ActionType.COUNTER_TRADE in trade_actions
                else None
            )
            include_responses = True
        elif ActionType.PROPOSE_TRADE in trade_actions:
            targets = (
                loop.debt_negotiation_targets()
                if loop.game.debts
                else tuple(
                    player_id for player_id in loop.game.active_player_ids
                    if player_id != actor
                )
            )
            action_kind = ActionType.PROPOSE_TRADE
            include_responses = False
        else:
            return ()

        if not targets or action_kind is None:
            if include_responses:
                response_actions = [
                    action for action in loop.trade_response_actions()
                    if action.action_type in trade_actions and loop.validate(action) is None
                ]
                return tuple(response_actions[:budget])
            return ()

        buckets: list[list[GameAction]] = []
        priority_responses: list[GameAction] = []
        if include_responses:
            priority_responses = [
                action for action in loop.trade_response_actions()
                if action.action_type in trade_actions and loop.validate(action) is None
            ]

        for target in targets:
            actor_assets = self._eligible_assets(loop, actor)
            target_assets = self._eligible_assets(loop, target)
            actor_bundles = self._sample_bundles(actor_assets)
            target_bundles = self._sample_bundles(target_assets)
            actor_cash = loop.game.player_states[actor].money
            target_cash = loop.game.player_states[target].money

            candidate_families = (
                self._cash_offers(action_kind, target, actor_cash, target_cash),
                self._actor_asset_offers(
                    action_kind, target, actor_bundles, target_cash
                ),
                self._target_asset_offers(
                    action_kind, target, target_bundles, actor_cash
                ),
                self._asset_exchanges(
                    action_kind, target, actor_bundles, target_bundles
                ),
            )
            buckets.extend(
                self._limit_family(family)
                for family in candidate_families
            )

        return self._sample_legal(
            loop,
            buckets,
            budget,
            priority_actions=priority_responses,
        )

    @staticmethod
    def _eligible_assets(loop: GameLoop, owner_id: UUID) -> tuple[Asset, ...]:
        assets: list[Asset] = []
        for tile in loop.game.owned_tiles(owner_id):
            group = loop.game.board.group_members(tile.group)
            if not any(loop.game.property_states[item.tile_id].houses for item in group):
                assets.append(("property", tile.tile_id))
        for deck_name, card in loop.game.player_states[owner_id].jail_cards:
            assets.append(("jail_card", f"{deck_name}:{card.id}"))
        return tuple(assets)

    def _sample_bundles(self, assets: tuple[Asset, ...]) -> tuple[tuple[Asset, ...], ...]:
        if not assets:
            return ()
        bundles: list[tuple[Asset, ...]] = [(asset,) for asset in assets]
        upper_size = min(self.max_bundle_size, len(assets))
        for size in range(2, upper_size + 1):
            total = _combination_count(len(assets), size)
            samples = min(self.bundle_samples_per_size, total)
            selected: set[tuple[int, ...]] = set()
            if total <= samples:
                selected.update(combinations(range(len(assets)), size))
            else:
                while len(selected) < samples:
                    selected.add(tuple(sorted(self.rng.sample(range(len(assets)), size))))
            bundles.extend(tuple(assets[index] for index in indices) for indices in sorted(selected))
        return tuple(bundles)

    @staticmethod
    def _cash_points(cash: int) -> tuple[int, ...]:
        return tuple(sorted({0, cash // 4, cash // 2, (3 * cash) // 4, cash}))

    def _cash_offers(
        self,
        kind: str,
        target: UUID,
        actor_cash: int,
        target_cash: int,
    ) -> list[GameAction]:
        return [
            self._offer(kind, target, money_offered=offered, money_requested=requested)
            for offered in self._cash_points(actor_cash)
            for requested in self._cash_points(target_cash)
            if offered or requested
        ]

    def _actor_asset_offers(
        self,
        kind: str,
        target: UUID,
        actor_bundles: tuple[tuple[Asset, ...], ...],
        target_cash: int,
    ) -> list[GameAction]:
        return [
            self._offer(
                kind,
                target,
                money_requested=requested,
                properties_offered=self._ids(bundle, "property"),
                jail_cards_offered=self._ids(bundle, "jail_card"),
            )
            for bundle in actor_bundles
            for requested in self._cash_points(target_cash)
        ]

    def _target_asset_offers(
        self,
        kind: str,
        target: UUID,
        target_bundles: tuple[tuple[Asset, ...], ...],
        actor_cash: int,
    ) -> list[GameAction]:
        return [
            self._offer(
                kind,
                target,
                money_offered=offered,
                properties_requested=self._ids(bundle, "property"),
                jail_cards_requested=self._ids(bundle, "jail_card"),
            )
            for bundle in target_bundles
            for offered in self._cash_points(actor_cash)
        ]

    def _asset_exchanges(
        self,
        kind: str,
        target: UUID,
        actor_bundles: tuple[tuple[Asset, ...], ...],
        target_bundles: tuple[tuple[Asset, ...], ...],
    ) -> list[GameAction]:
        pairs = [
            (actor_bundle, target_bundle)
            for actor_bundle in actor_bundles
            for target_bundle in target_bundles
        ]
        if len(pairs) > self.max_candidates_per_family:
            pairs = self.rng.sample(pairs, self.max_candidates_per_family)
        return [
            self._offer(
                kind,
                target,
                properties_offered=self._ids(actor_bundle, "property"),
                properties_requested=self._ids(target_bundle, "property"),
                jail_cards_offered=self._ids(actor_bundle, "jail_card"),
                jail_cards_requested=self._ids(target_bundle, "jail_card"),
            )
            for actor_bundle, target_bundle in pairs
        ]

    def _limit_family(self, candidates: list[GameAction]) -> list[GameAction]:
        if len(candidates) <= self.max_candidates_per_family:
            return candidates
        return self.rng.sample(candidates, self.max_candidates_per_family)

    @staticmethod
    def _ids(bundle: Iterable[Asset], kind: str) -> tuple[int | str, ...]:
        return tuple(sorted(value for asset_kind, value in bundle if asset_kind == kind))

    @staticmethod
    def _offer(
        kind: str,
        target: UUID,
        *,
        money_offered: int = 0,
        money_requested: int = 0,
        properties_offered: tuple[int | str, ...] = (),
        properties_requested: tuple[int | str, ...] = (),
        jail_cards_offered: tuple[int | str, ...] = (),
        jail_cards_requested: tuple[int | str, ...] = (),
    ) -> GameAction:
        payload = {
            "money_offered": money_offered,
            "money_requested": money_requested,
            "properties_offered": list(properties_offered),
            "properties_requested": list(properties_requested),
            "jail_cards_offered": list(jail_cards_offered),
            "jail_cards_requested": list(jail_cards_requested),
        }
        if kind == ActionType.PROPOSE_TRADE:
            payload["target_player_id"] = target
        return GameAction(kind, payload)

    def _sample_legal(
        self,
        loop: GameLoop,
        buckets: list[list[GameAction]],
        budget: int,
        *,
        priority_actions: list[GameAction] | None = None,
    ) -> tuple[GameAction, ...]:
        unique_buckets: list[list[GameAction]] = []
        seen: list[GameAction] = []
        selected = []
        for action in priority_actions or ():
            if len(selected) == budget:
                break
            if action not in seen and loop.validate(action) is None:
                seen.append(action)
                selected.append(action)
        for bucket in buckets:
            unique = []
            for action in bucket:
                if action in seen or loop.validate(action) is not None:
                    continue
                seen.append(action)
                unique.append(action)
            if unique:
                self.rng.shuffle(unique)
                unique_buckets.append(unique)

        while len(selected) < budget and unique_buckets:
            next_buckets = []
            for bucket in unique_buckets:
                if len(selected) == budget:
                    break
                selected.append(bucket.pop())
                if bucket:
                    next_buckets.append(bucket)
            unique_buckets = next_buckets
        return tuple(selected)


class LegalCandidateProvider:
    """Combine engine-enumerated actions with bounded structured trade offers."""

    def __init__(
        self,
        max_candidates: int = 128,
        seed: int | None = None,
        *,
        max_trade_candidates: int = 32,
    ) -> None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates deve ser um inteiro positivo.")
        if (
            isinstance(max_trade_candidates, bool)
            or not isinstance(max_trade_candidates, int)
            or max_trade_candidates < 0
        ):
            raise ValueError("max_trade_candidates deve ser um inteiro não negativo.")
        self.max_candidates = max_candidates
        self.max_trade_candidates = min(max_trade_candidates, max_candidates)
        self.trades = TradeCandidateProvider(
            # The provider itself requires a positive capacity; the wrapper
            # enforces a zero trade budget through ``limit=0`` when disabled.
            max_candidates=max(1, self.max_trade_candidates),
            seed=seed,
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset trade candidate sampling for a new episode."""
        self.trades.reset(seed)

    def __call__(self, loop: GameLoop) -> tuple[GameAction, ...]:
        candidates = list(loop.legal_actions())
        remaining = min(
            self.max_trade_candidates,
            max(0, self.max_candidates - len(candidates)),
        )
        candidates.extend(self.trades(loop, limit=remaining))
        return tuple(candidates)


def _combination_count(n: int, r: int) -> int:
    """Small integer binomial coefficient without constructing combinations."""
    result = 1
    for index in range(1, min(r, n - r) + 1):
        result = result * (n - index + 1) // index
    return result
