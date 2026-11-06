"""Rating-aware opponent pools, table scheduling, and promotion gate utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from app.envs.baselines import AggressivePolicy, ConservativePolicy, RandomPolicy


RATING_MODEL = "openskill.PlackettLuce"
REWARD_VERSION = "placement-geometric-q0.4-delta0.05-v1"
DEFAULTS_STATUS = "provisional"
ANCHOR_NAMES = ("random", "aggressive", "conservative")


@dataclass(frozen=True)
class LeagueConfig:
    """League, archive, table, rating, and gate settings (provisional defaults)."""

    # Snapshot and archive lifecycle.
    snapshot_games: int = 100
    max_archive_size: int = 32
    recent_snapshot_count: int = 8
    milestone_interval: int = 10
    # Admission uses candidate mu before snapshot uncertainty inflation.
    admission_mu_tolerance: float = 5.0
    # Paired promotion gate.
    promotion_margin: float = 0.02
    confidence_level: float = 0.95
    min_gate_games_per_n: int = 8
    max_gate_games_per_n: int = 40
    gate_regression_tolerance: float = 0.10
    # Target player-count distribution. Until dynamic spaces are supported,
    # the trainer samples only the configured player count.
    player_count_distribution: Mapping[int, float] = field(default_factory=lambda: {4: 1.0})
    stratified_player_counts: bool = True
    max_turns_per_player: int = 250
    # Table composition, sampled independently for each opponent seat.
    champion_probability: float = 0.20
    recent_probability: float = 0.25
    rating_archive_probability: float = 0.25
    anchor_probability: float = 0.30
    minimum_heuristics_per_table: int = 1
    maximum_neural_opponents_per_table: int = 2
    rating_temperature: float = 8.0
    rating_probability_floor: float = 0.10
    # OpenSkill rating scale and learner dynamics.
    rating_mu: float = 25.0
    rating_sigma: float = 8.3333333333
    learner_tau: float = 0.50
    snapshot_sigma_inflation: float = 1.50
    anchor_mu: float = 25.0
    anchor_sigma: float = 8.3333333333
    fixed_initial_champion: str | None = "aggressive"
    # Gate is run synchronously at a snapshot boundary; no overlapping gates.
    gate_seed: int = 880_000
    defaults_status: str = DEFAULTS_STATUS

    def __post_init__(self) -> None:
        if self.snapshot_games < 1 or self.max_archive_size < 1:
            raise ValueError("snapshot_games e max_archive_size devem ser positivos.")
        if self.recent_snapshot_count < 0 or self.milestone_interval < 1:
            raise ValueError("recent_snapshot_count não pode ser negativo e milestone_interval deve ser positivo.")
        if self.min_gate_games_per_n < 1 or self.max_gate_games_per_n < self.min_gate_games_per_n:
            raise ValueError("O teto do portão precisa ser maior ou igual ao mínimo positivo.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level deve estar entre 0 e 1.")
        if self.promotion_margin < 0 or self.gate_regression_tolerance < 0:
            raise ValueError("Margem de promoção e tolerância de regressão não podem ser negativas.")
        if self.max_turns_per_player < 1 or self.minimum_heuristics_per_table < 1:
            raise ValueError("Limite de turnos e mínimo de heurísticas devem ser positivos.")
        if self.maximum_neural_opponents_per_table < 0:
            raise ValueError("maximum_neural_opponents_per_table não pode ser negativo.")
        if self.rating_temperature <= 0 or not 0 <= self.rating_probability_floor <= 1:
            raise ValueError("Temperatura precisa ser positiva e o piso deve estar em [0, 1].")
        if self.admission_mu_tolerance < 0 or self.learner_tau < 0:
            raise ValueError("Tolerância de admissão e learner_tau não podem ser negativos.")
        if self.snapshot_sigma_inflation < 1:
            raise ValueError("snapshot_sigma_inflation precisa ser pelo menos 1.")
        if min(self.rating_mu, self.rating_sigma, self.anchor_mu, self.anchor_sigma) <= 0:
            raise ValueError("Os parâmetros iniciais de rating precisam ser positivos.")
        probabilities = (
            self.champion_probability,
            self.recent_probability,
            self.rating_archive_probability,
            self.anchor_probability,
        )
        if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0):
            raise ValueError("As probabilidades da mistura devem ser não negativas e somar 1.")
        if not self.player_count_distribution or any(
            not 2 <= int(players) <= 12 or weight < 0
            for players, weight in self.player_count_distribution.items()
        ) or not any(weight > 0 for weight in self.player_count_distribution.values()):
            raise ValueError("player_count_distribution deve mapear 2–12 jogadores a pesos positivos.")
        if any(int(players) - 1 < self.minimum_heuristics_per_table
               for players, weight in self.player_count_distribution.items() if weight > 0):
            raise ValueError("Cada N suportado precisa comportar minimum_heuristics_per_table.")
        if self.fixed_initial_champion is not None and not self.fixed_initial_champion:
            raise ValueError("fixed_initial_champion deve ser um rótulo não vazio ou None.")


@dataclass(frozen=True)
class PlayerRating:
    """Serializable OpenSkill mean and uncertainty for one league member."""

    mu: float
    sigma: float

    @property
    def conservative(self) -> float:
        """Conservative ordinal used for ordering, not candidate admission."""
        return self.mu - 3.0 * self.sigma


@dataclass(frozen=True)
class MatchTable:
    """Reproducible table assignment selected before a game starts."""

    num_players: int
    learner_seat: int
    opponent_ids: tuple[str, ...]
    seed: int
    max_turns: int


@dataclass
class PromotionGateState:
    """Checkpointable progress and paired outcomes for one candidate gate."""

    candidate_generation: int
    champion_id: str
    seed_start: int
    results_by_players: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    status: str = "running"
    decision: str | None = None
    reason: str | None = None


def sample_num_players(
    distribution: Mapping[int, float], *, rng: random.Random, stratified_index: int | None = None
) -> int:
    """Sample N from a weighted target distribution, optionally stratified by index."""
    choices = sorted((int(n), float(w)) for n, w in distribution.items() if w > 0)
    if not choices:
        raise ValueError("A distribuição de número de jogadores não tem peso positivo.")
    if stratified_index is not None:
        total = sum(weight for _, weight in choices)
        unit = ((stratified_index % 10_000) + 0.5) / 10_000
        cumulative = 0.0
        for players, weight in choices:
            cumulative += weight / total
            if unit < cumulative:
                return players
        return choices[-1][0]
    total = sum(weight for _, weight in choices)
    return rng.choices([n for n, _ in choices], weights=[w for _, w in choices], k=1)[0]


def schedule_table(
    config: LeagueConfig,
    *,
    rng: random.Random,
    episode_index: int,
    opponent_ids: Sequence[str] | None = None,
    opponent_sampler: Callable[[int, random.Random], Sequence[str]] | None = None,
    heuristic_names: Sequence[str] = ANCHOR_NAMES,
    fixed_num_players: int | None = None,
    fixed_seat: int | None = None,
    seed: int | None = None,
) -> MatchTable:
    """Choose N, learner seat, opponent identities, then seed in that order."""
    num_players = fixed_num_players if fixed_num_players is not None else sample_num_players(
        config.player_count_distribution,
        rng=rng,
        stratified_index=episode_index if config.stratified_player_counts else None,
    )
    if not 2 <= num_players <= 12:
        raise ValueError("N precisa estar entre 2 e 12.")
    seat = rng.randrange(num_players) if fixed_seat is None else fixed_seat
    if not 0 <= seat < num_players:
        raise ValueError("Assento fixo fora da mesa.")
    if opponent_sampler is not None:
        opponent_ids = opponent_sampler(num_players - 1, rng)
    if opponent_ids is None:
        raise ValueError("Informe opponent_ids ou opponent_sampler.")
    expected = num_players - 1
    if len(opponent_ids) != expected:
        raise ValueError(f"A mesa de {num_players} lugares precisa de {expected} oponentes.")
    selected = list(opponent_ids)
    neural = {name for name in selected if name.startswith("snapshot:") or name.startswith("champion:")}
    if len(neural) > config.maximum_neural_opponents_per_table:
        raise ValueError("A mesa excede maximum_neural_opponents_per_table.")
    heuristic_count = sum(name in heuristic_names or name.startswith("anchor:") for name in selected)
    if heuristic_count < config.minimum_heuristics_per_table:
        raise ValueError("A composição precisa preservar o mínimo configurado de heurísticas.")
    game_seed = rng.randrange(2**31) if seed is None else int(seed)
    return MatchTable(
        num_players=num_players,
        learner_seat=seat,
        opponent_ids=tuple(selected),
        seed=game_seed,
        max_turns=config.max_turns_per_player * num_players,
    )


def placement_ranks(num_players: int, elimination_order: Sequence[int], winner: int | None,
                    active_players: Sequence[int], truncated: bool) -> list[int]:
    """Convert a game outcome to competition ranks; truncated survivors tie."""
    ranks = [0] * num_players
    if truncated:
        active = set(active_players)
        shared_rank = 1
        for player in active:
            ranks[player] = shared_rank
        next_rank = len(active) + 1
        for player in reversed(elimination_order):
            ranks[player] = next_rank
            next_rank += 1
        return ranks
    if winner is not None:
        ranks[winner] = 1
        next_rank = 2
        for player in reversed(elimination_order):
            ranks[player] = next_rank
            next_rank += 1
        return ranks
    # Defensive fallback for a non-truncated game ended without a winner.
    for rank, player in enumerate(reversed(elimination_order), start=1):
        ranks[player] = rank
    return ranks


class TrainingLeague:
    """Rating-aware pool of immutable anchors and archived PPO snapshots."""

    def __init__(self, heuristics: Mapping[str, Callable[[], Any]] | None = None,
                 *, archive_opponent_rate: float = 0.5,
                 config: LeagueConfig | None = None) -> None:
        self.heuristics = dict(heuristics or {
            "random": RandomPolicy,
            "aggressive": AggressivePolicy,
            "conservative": ConservativePolicy,
        })
        self.config = config or LeagueConfig()
        if not self.heuristics:
            raise ValueError("A liga precisa conter ao menos uma heurística.")
        if "archive" in self.heuristics:
            raise ValueError("'archive' é reservado para políticas históricas.")
        if not all(callable(factory) for factory in self.heuristics.values()):
            raise TypeError("Cada heurística deve ser uma fábrica sem argumentos.")
        if (self.config.fixed_initial_champion is not None
                and self.config.fixed_initial_champion not in self.heuristics):
            raise ValueError("fixed_initial_champion precisa identificar uma heurística configurada.")
        if not 0 <= archive_opponent_rate <= 1:
            raise ValueError("archive_opponent_rate deve estar entre 0 e 1.")
        self.archive_opponent_rate = float(archive_opponent_rate)
        self.ratings = {
            name: PlayerRating(self.config.anchor_mu, self.config.anchor_sigma)
            for name in self.heuristics
        }
        self.ratings.setdefault("random", PlayerRating(self.config.anchor_mu, self.config.anchor_sigma))
        self.ratings["learner"] = PlayerRating(self.config.rating_mu, self.config.rating_sigma)
        self.ratings["champion"] = PlayerRating(self.config.rating_mu, self.config.rating_sigma)
        self.ratings_by_num_players: dict[str, dict[str, PlayerRating]] = {}
        self.snapshot_meta: dict[str, dict[str, Any]] = {}
        self.champion_id = (
            f"anchor:{self.config.fixed_initial_champion}"
            if self.config.fixed_initial_champion is not None
            else None
        )
        self.gate_state: PromotionGateState | None = None
        self._model = None

    def sample_lineup(self, opponent_count: int, *, archive_available: bool,
                      rng: random.Random, episode_index: int) -> list[str]:
        """Compatibility sampler that returns anchor labels and archive markers."""
        archive_ids = [name for name in self.snapshot_meta]
        selected = self.sample_opponents(opponent_count, rng=rng,
                                         episode_index=episode_index,
                                         archive_ids=archive_ids if archive_available else [])
        return [name if name in self.heuristics else "archive" for name in selected]

    def export_state(self) -> dict[str, Any]:
        """Return JSON/checkpoint-safe mutable league state."""
        return {
            "ratings": {name: {"mu": value.mu, "sigma": value.sigma}
                        for name, value in self.ratings.items()},
            "ratings_by_num_players": {
                num_players: {name: {"mu": value.mu, "sigma": value.sigma}
                              for name, value in ratings.items()}
                for num_players, ratings in self.ratings_by_num_players.items()
            },
            "snapshot_meta": {
                name: {**meta, "rating": {"mu": meta["rating"].mu, "sigma": meta["rating"].sigma}}
                for name, meta in self.snapshot_meta.items()
            },
            "champion_id": self.champion_id,
            "gate_state": None if self.gate_state is None else {
                "candidate_generation": self.gate_state.candidate_generation,
                "champion_id": self.gate_state.champion_id,
                "seed_start": self.gate_state.seed_start,
                "results_by_players": self.gate_state.results_by_players,
                "status": self.gate_state.status,
                "decision": self.gate_state.decision,
                "reason": self.gate_state.reason,
            },
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore learner, anchor, snapshot, champion, and in-progress gate ratings."""
        for name, value in state.get("ratings", {}).items():
            self.ratings[name] = PlayerRating(float(value["mu"]), float(value["sigma"]))
        self.ratings_by_num_players = {
            str(num_players): {
                name: PlayerRating(float(value["mu"]), float(value["sigma"]))
                for name, value in ratings.items()
            }
            for num_players, ratings in state.get("ratings_by_num_players", {}).items()
        }
        for name, meta in state.get("snapshot_meta", {}).items():
            restored = dict(meta)
            rating = restored.get("rating")
            if isinstance(rating, Mapping):
                restored["rating"] = PlayerRating(float(rating["mu"]), float(rating["sigma"]))
            self.snapshot_meta[name] = restored
        self.champion_id = state.get("champion_id", self.champion_id)
        gate = state.get("gate_state")
        self.gate_state = PromotionGateState(**gate) if gate else None

    def add_snapshot(self, generation: int, episode: int, rating: PlayerRating,
                     reward_version: str = REWARD_VERSION) -> str:
        """Add a snapshot candidate and return its stable pool identifier."""
        identifier = f"snapshot:{generation}"
        self.snapshot_meta[identifier] = {
            "generation": int(generation), "episode": int(episode),
            "rating": rating, "reward_version": reward_version,
            "mu_before_inflation": self.ratings["learner"].mu,
            "rating_by_num_players": {
                num_players: {
                    "mu": values.get("learner", self.ratings["learner"]).mu,
                    "sigma": values.get("learner", self.ratings["learner"]).sigma
                    * self.config.snapshot_sigma_inflation,
                }
                for num_players, values in self.ratings_by_num_players.items()
            },
        }
        self.ratings[identifier] = rating
        return identifier

    def prune(self, available_ids: Sequence[str] | None = None) -> set[str]:
        """Keep champion, top rated, recent, and configured milestone snapshots."""
        allowed = set(self.snapshot_meta) if available_ids is None else set(available_ids)
        entries = sorted(
            ((name, meta) for name, meta in self.snapshot_meta.items() if name in allowed),
            key=lambda item: item[1]["generation"],
        )
        if len(entries) <= self.config.max_archive_size:
            return allowed
        champion = self.champion_id if self.champion_id in allowed else None
        ranked = sorted(entries, key=lambda item: item[1]["rating"].conservative, reverse=True)
        recent = [name for name, _ in entries[-self.config.recent_snapshot_count:]]
        milestones = [name for name, meta in entries
                      if meta["generation"] % self.config.milestone_interval == 0]
        keep = {champion} if champion else set()
        priority_groups = (
            [name for name, _ in ranked[:max(1, self.config.max_archive_size // 2)]],
            list(reversed(recent)),
            list(reversed(milestones)),
            [name for name, _ in reversed(entries)],
        )
        for group in priority_groups:
            for name in group:
                if len(keep) >= self.config.max_archive_size:
                    break
                keep.add(name)
        return keep

    def _rating_model(self):
        if self._model is None:
            try:
                from openskill.models import PlackettLuce
            except ImportError as exc:
                raise RuntimeError(
                    "O rating da liga requer a dependência opcional openskill; instale o extra training."
                ) from exc
            self._model = PlackettLuce(mu=self.config.rating_mu, sigma=self.config.rating_sigma)
        return self._model

    def sample_opponents(self, opponent_count: int, *, rng: random.Random,
                         episode_index: int, archive_ids: Sequence[str]) -> list[str]:
        """Draw opponent sources, ratings, and anchors while enforcing constraints."""
        if opponent_count < self.config.minimum_heuristics_per_table:
            raise ValueError("N não comporta o mínimo de heurísticas configurado.")
        selected: list[str] = []
        recent = list(archive_ids[-self.config.recent_snapshot_count:]) if self.config.recent_snapshot_count else []
        for _ in range(opponent_count):
            source = rng.choices(
                ("champion", "recent", "rating", "anchor"),
                weights=(self.config.champion_probability, self.config.recent_probability,
                         self.config.rating_archive_probability, self.config.anchor_probability), k=1,
            )[0]
            if source == "champion" and self.champion_id:
                if self.champion_id.startswith("snapshot:"):
                    selected.append(self.champion_id)
                elif self.champion_id.startswith("anchor:"):
                    selected.append(self.champion_id.split(":", 1)[1])
                else:
                    selected.append(rng.choice(tuple(self.heuristics)))
            elif source == "recent" and recent:
                selected.append(rng.choice(recent))
            elif source == "rating" and archive_ids:
                selected.append(self._sample_rating_archive(archive_ids, rng))
            else:
                selected.append(rng.choice(tuple(self.heuristics)))
        while sum(name in self.heuristics for name in selected) < self.config.minimum_heuristics_per_table:
            neural_indices = [i for i, name in enumerate(selected) if name not in self.heuristics]
            if not neural_indices:
                break
            index = neural_indices[-1]
            selected[index] = rng.choice(tuple(self.heuristics))
        max_neural = self.config.maximum_neural_opponents_per_table
        neural_indices = [i for i, name in enumerate(selected) if name not in self.heuristics]
        for index in neural_indices[max_neural:]:
            selected[index] = rng.choice(tuple(self.heuristics))
        return selected

    def _sample_rating_archive(self, archive_ids: Sequence[str], rng: random.Random) -> str:
        learner_mu = self.ratings["learner"].mu
        scores = [
            math.exp(-abs(self.snapshot_meta.get(name, {}).get("rating", self.ratings["learner"]).mu - learner_mu)
                     / self.config.rating_temperature)
            for name in archive_ids
        ]
        total = sum(scores)
        probabilities = [score / total for score in scores]
        floor = self.config.rating_probability_floor / len(archive_ids)
        probabilities = [(1.0 - self.config.rating_probability_floor) * p + floor for p in probabilities]
        return rng.choices(list(archive_ids), weights=probabilities, k=1)[0]

    def create_heuristic(self, name: str) -> Any:
        try:
            return self.heuristics[name]()
        except KeyError as exc:
            raise ValueError(f"Heurística desconhecida na liga: {name!r}.") from exc

    def update_ratings(self, participant_ids: Sequence[str], ranks: Sequence[int],
                       num_players: int | None = None) -> None:
        """Update global and optional player-count-specific Plackett-Luce ratings."""
        if len(participant_ids) != len(ranks):
            raise ValueError("IDs e postos devem ter o mesmo tamanho.")
        model = self._rating_model()

        def update(target: dict[str, PlayerRating]) -> None:
            rating_objects = []
            bases = []
            for participant in participant_ids:
                base_participant = participant.split("#seat", 1)[0]
                current = target.get(
                    base_participant,
                    self.ratings.get(base_participant,
                                     PlayerRating(self.config.rating_mu, self.config.rating_sigma)),
                )
                sigma = (math.sqrt(current.sigma**2 + self.config.learner_tau**2)
                         if base_participant == "learner" else current.sigma)
                rating_objects.append(model.rating(mu=current.mu, sigma=sigma, name=participant))
                bases.append(base_participant)
            updated = model.rate([[rating] for rating in rating_objects], ranks=list(ranks), tau=0.0)
            grouped: dict[str, list[PlayerRating]] = {}
            for participant, group in zip(bases, updated):
                value = group[0]
                grouped.setdefault(participant, []).append(
                    PlayerRating(float(value.mu), float(value.sigma))
                )
            for participant, values in grouped.items():
                target[participant] = PlayerRating(
                    (self.config.anchor_mu if participant == "random"
                     else sum(value.mu for value in values) / len(values)),
                    math.sqrt(sum(value.sigma**2 for value in values) / len(values)),
                )

        if num_players is not None:
            player_count_ratings = self.ratings_by_num_players.setdefault(
                str(num_players),
                {name: value for name, value in self.ratings.items()},
            )
            update(self.ratings)
            update(player_count_ratings)
        else:
            update(self.ratings)
        for snapshot_id, metadata in self.snapshot_meta.items():
            if snapshot_id in self.ratings:
                metadata["rating"] = self.ratings[snapshot_id]
                if num_players is not None:
                    by_n = metadata.setdefault("rating_by_num_players", {})
                    rating_for_n = self.ratings_by_num_players[str(num_players)].get(snapshot_id)
                    if rating_for_n is not None:
                        by_n[str(num_players)] = {
                            "mu": rating_for_n.mu, "sigma": rating_for_n.sigma,
                        }

    def admission_allowed(self, candidate: PlayerRating, pool_ratings: Sequence[PlayerRating]) -> bool:
        """Permissive admission based on pre-inflation candidate mu, not ordinal."""
        if not pool_ratings:
            return True
        return candidate.mu >= max(rating.mu for rating in pool_ratings) - self.config.admission_mu_tolerance

    def snapshot_rating(self) -> PlayerRating:
        """Copy learner rating while increasing uncertainty for an immutable snapshot."""
        learner = self.ratings["learner"]
        return PlayerRating(learner.mu, learner.sigma * self.config.snapshot_sigma_inflation)


class CallableTrainingLeague(TrainingLeague):
    """Adapt a custom opponent-source sampler to the league interface."""

    def __init__(self, sampler: Callable[[int, bool, random.Random, int], Sequence[str]],
                 heuristics: Mapping[str, Callable[[], Any]] | None = None,
                 *, archive_opponent_rate: float = 0.5, config: LeagueConfig | None = None) -> None:
        super().__init__(heuristics, archive_opponent_rate=archive_opponent_rate, config=config)
        self.sampler = sampler

    def sample_lineup(self, opponent_count: int, *, archive_available: bool,
                      rng: random.Random, episode_index: int) -> list[str]:
        lineup = list(self.sampler(opponent_count, archive_available, rng, episode_index))
        if len(lineup) != opponent_count:
            raise ValueError("A função da liga deve retornar exatamente um rótulo por oponente.")
        valid_names = set(self.heuristics) | {"archive"}
        unknown = set(lineup) - valid_names
        if unknown:
            raise ValueError(f"Perfis desconhecidos retornados pela liga: {sorted(unknown)}.")
        return lineup


def paired_gate_summary(state: PromotionGateState, config: LeagueConfig) -> dict[str, Any]:
    """Compute seed-paired mean reward deltas and conservative confidence bounds."""
    per_n: dict[int, dict[str, float]] = {}
    weighted_mean = 0.0
    weighted_lower = 0.0
    weighted_bounds_available = True
    weights = {int(n): float(w) for n, w in config.player_count_distribution.items() if w > 0}
    weight_total = sum(weights.values())
    # Bonferroni-adjust the repeated sequential looks and N strata.
    looks = max(1, config.max_gate_games_per_n - config.min_gate_games_per_n + 1)
    comparisons = max(1, len(weights) * looks)
    tail = (1.0 - config.confidence_level) / (2.0 * comparisons)
    critical = NormalDist().inv_cdf(1.0 - tail)
    enough_everywhere = True
    regression = False
    for num_players, target_weight in weights.items():
        rows = state.results_by_players.get(str(num_players), [])
        differences = [float(row["candidate_reward"]) - float(row["champion_reward"]) for row in rows]
        count = len(differences)
        enough_everywhere &= count >= config.min_gate_games_per_n
        mean = sum(differences) / count if count else 0.0
        variance = sum((value - mean) ** 2 for value in differences) / (count - 1) if count > 1 else None
        se = math.sqrt(variance / count) if variance is not None else None
        lower = mean - critical * se if se is not None else None
        upper = mean + critical * se if se is not None else None
        regression |= (
            count >= config.min_gate_games_per_n and lower is not None
            and lower < -config.gate_regression_tolerance
        )
        normalized_weight = target_weight / weight_total
        weighted_mean += normalized_weight * mean
        if lower is None:
            weighted_bounds_available = False
        else:
            weighted_lower += normalized_weight * lower
        per_n[num_players] = {"games": float(count), "mean_delta": mean, "standard_error": se,
                              "ci_lower": lower, "ci_upper": upper}
    return {
        "per_num_players": per_n,
        "weighted_mean_delta": weighted_mean,
        "weighted_ci_lower": weighted_lower if weighted_bounds_available else None,
        "enough_games": enough_everywhere,
        "regression_block": regression,
        "promote": (enough_everywhere and not regression and weighted_bounds_available
                    and weighted_lower > config.promotion_margin),
        "max_games_reached": all(
            len(state.results_by_players.get(str(n), [])) >= config.max_gate_games_per_n for n in weights
        ),
    }
