"""Policy adapter for selectors that consume encoded decision features."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.domain.analytics.decision_features import EncodedDecisionContext
from app.domain.game.actions import GameAction
from app.domain.game.game_loop import GameLoop


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    """One legal action paired with its full context and model features."""

    action: GameAction
    decision: EncodedDecisionContext


class CandidateEvaluator(Protocol):
    def __call__(self, loop: GameLoop, action: GameAction) -> EncodedDecisionContext: ...


class CandidateSelector(Protocol):
    def __call__(
        self,
        loop: GameLoop,
        candidates: Sequence[FeatureCandidate],
    ) -> GameAction: ...


class FeaturePolicy:
    """Evaluate candidate actions and delegate preferences to a selector.

    The evaluator owns candidate-specific analysis and encoding. The selector
    can be a trained model or a baseline scorer; this class does not prescribe
    a utility function. The selected action must be one of the evaluated legal
    candidates, preventing the model from bypassing the engine's action rules.
    """

    def __init__(
        self,
        evaluator: CandidateEvaluator,
        selector: CandidateSelector,
        *,
        candidate_provider: Callable[[GameLoop], Iterable[GameAction]] | None = None,
        max_candidates: int = 128,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator deve ser chamável.")
        if not callable(selector):
            raise TypeError("selector deve ser chamável.")
        if candidate_provider is not None and not callable(candidate_provider):
            raise TypeError("candidate_provider deve ser chamável ou None.")
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates deve ser um inteiro positivo.")
        self.evaluator = evaluator
        self.selector = selector
        self.candidate_provider = candidate_provider or self._legal_candidates
        self.max_candidates = max_candidates

    def __call__(self, loop: GameLoop) -> GameAction:
        candidates = tuple(self.candidate_provider(loop))
        if not candidates:
            raise RuntimeError("O provedor não retornou ações candidatas.")
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"O provedor retornou {len(candidates)} ações; limite configurado: "
                f"{self.max_candidates}."
            )
        if any(not isinstance(action, GameAction) for action in candidates):
            raise TypeError("candidate_provider deve retornar apenas GameAction.")
        if any(
            action == previous
            for index, action in enumerate(candidates)
            for previous in candidates[:index]
        ):
            raise ValueError("candidate_provider retornou ações duplicadas.")

        evaluated = []
        for action in candidates:
            reason = loop.validate(action)
            if reason is not None:
                raise ValueError(f"Ação candidata {action.action_type} ilegal: {reason}")
            decision = self.evaluator(loop, action)
            if not isinstance(decision, EncodedDecisionContext):
                raise TypeError("evaluator deve retornar EncodedDecisionContext.")
            evaluated.append(FeatureCandidate(action, decision))

        selected = self.selector(loop, tuple(evaluated))
        if not isinstance(selected, GameAction):
            raise TypeError("selector deve retornar GameAction.")
        if selected not in candidates:
            raise ValueError("selector escolheu uma ação que não estava entre as candidatas.")
        reason = loop.validate(selected)
        if reason is not None:
            raise ValueError(f"selector escolheu ação ilegal {selected.action_type}: {reason}")
        return selected

    def reset(self, seed: int | None = None) -> None:
        """Forward an optional episode seed to stateful evaluator/selector objects."""
        for component in (self.evaluator, self.selector, self.candidate_provider):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset(seed)

    @staticmethod
    def _legal_candidates(loop: GameLoop) -> tuple[GameAction, ...]:
        actions = list(loop.legal_actions())
        if loop.game.pending_trade is not None:
            actions.extend(
                action for action in loop.trade_response_actions()
                if loop.validate(action) is None
            )
        return tuple(actions)
