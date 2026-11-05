"""Customizable opponent-lineup policies for generational PPO training."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.envs.baselines import AggressivePolicy, ConservativePolicy, RandomPolicy


PolicyFactory = Callable[[], Any]


class TrainingLeague:
    """Select a reproducible, varied lineup for each training game.

    Subclass ``sample_lineup`` to implement a different schedule, or pass a
    mapping of custom heuristic names to zero-argument policy factories.
    The reserved label ``"archive"`` asks the trainer to select a frozen
    generation from its policy bank.
    """

    def __init__(
        self,
        heuristics: Mapping[str, PolicyFactory] | None = None,
        *,
        archive_opponent_rate: float = 0.5,
    ) -> None:
        self.heuristics = dict(heuristics or {
            "random": RandomPolicy,
            "aggressive": AggressivePolicy,
            "conservative": ConservativePolicy,
        })
        if not self.heuristics:
            raise ValueError("A liga precisa conter ao menos uma heurística.")
        if "archive" in self.heuristics:
            raise ValueError("'archive' é reservado para políticas históricas.")
        if not all(callable(factory) for factory in self.heuristics.values()):
            raise TypeError("Cada heurística deve ser uma fábrica chamável sem argumentos.")
        if not 0.0 <= archive_opponent_rate <= 1.0:
            raise ValueError("archive_opponent_rate deve estar entre 0 e 1.")
        self.archive_opponent_rate = float(archive_opponent_rate)

    def sample_lineup(
        self,
        opponent_count: int,
        *,
        archive_available: bool,
        rng: random.Random,
        episode_index: int,
    ) -> list[str]:
        """Return one heuristic/archive label per opponent seat.

        Heuristics rotate across episodes so two-player games also encounter
        all profiles. In larger games, the lineup cycles through every
        configured heuristic before shuffling seats. Historical policies may
        replace individual seats after the archive has at least one snapshot.
        """
        if opponent_count < 1:
            raise ValueError("Uma partida de treino precisa ter ao menos um oponente.")
        names = tuple(self.heuristics)
        offset = episode_index * opponent_count
        lineup = [names[(offset + seat) % len(names)] for seat in range(opponent_count)]
        rng.shuffle(lineup)
        if archive_available and self.archive_opponent_rate:
            for seat in range(opponent_count):
                if rng.random() < self.archive_opponent_rate:
                    lineup[seat] = "archive"
            if all(profile == "archive" for profile in lineup):
                lineup[rng.randrange(opponent_count)] = names[offset % len(names)]
        return lineup

    def create_heuristic(self, name: str) -> Any:
        """Instantiate a configured heuristic by its stable label."""
        try:
            return self.heuristics[name]()
        except KeyError as exc:
            raise ValueError(f"Heurística desconhecida na liga: {name!r}.") from exc


class CallableTrainingLeague(TrainingLeague):
    """Adapt a custom lineup function to the ``TrainingLeague`` interface."""

    def __init__(
        self,
        sampler: Callable[[int, bool, random.Random, int], Sequence[str]],
        heuristics: Mapping[str, PolicyFactory] | None = None,
        *,
        archive_opponent_rate: float = 0.5,
    ) -> None:
        super().__init__(heuristics, archive_opponent_rate=archive_opponent_rate)
        self.sampler = sampler

    def sample_lineup(
        self,
        opponent_count: int,
        *,
        archive_available: bool,
        rng: random.Random,
        episode_index: int,
    ) -> list[str]:
        lineup = list(self.sampler(opponent_count, archive_available, rng, episode_index))
        if len(lineup) != opponent_count:
            raise ValueError("A função da liga deve retornar exatamente um perfil por oponente.")
        valid_names = set(self.heuristics) | {"archive"}
        unknown = set(lineup) - valid_names
        if unknown:
            raise ValueError(f"Perfis desconhecidos retornados pela liga: {sorted(unknown)}.")
        if "archive" in lineup and not archive_available:
            raise ValueError("A liga solicitou política arquivada, mas o banco está vazio.")
        return lineup
