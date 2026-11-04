"""Reproducible baseline evaluation for Monopoly policies.

Examples:
    python -m app.envs.simulate --games 200 --policies greedy random --seed 0
    python -m app.envs.simulate --games 120 --policies greedy greedy random --seed 20
    python -m app.envs.simulate --games 200 --policies greedy:0 greedy:150 --seed 20
    python -m app.envs.simulate --games 200 --policies greedy:0 greedy:150 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.application.game_setup import create_game, load_selected_game
from app.envs.baselines import GreedyPolicy, RandomPolicy

POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy}


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    reserve: int | None = None
    explicit_reserve: bool = False

    @property
    def label(self) -> str:
        """Return the CLI name, including an explicitly supplied reserve."""
        if self.name == "greedy" and self.explicit_reserve:
            return f"greedy:{self.reserve}"
        return self.name

    def create(self):
        """Instantiate the baseline described by this policy specification."""
        if self.name == "greedy":
            return GreedyPolicy(reserve=150 if self.reserve is None else self.reserve)
        return RandomPolicy()


def _new_metrics() -> dict[str, int | float]:
    return {
        "matches": 0,
        "decided_matches": 0,
        "match_wins": 0,
        "match_losses": 0,
        "match_draws": 0,
        "appearances": 0,
        "player_wins": 0,
        "bankruptcies": 0,
        "truncated_games": 0,
        "no_winner_games": 0,
        "cash": 0,
        "net_worth": 0,
        "properties": 0,
        "turns": 0,
    }


def _wilson_interval(wins: int, games: int) -> tuple[float, float]:
    """Return the 95% Wilson interval for a binomial win rate."""
    if games == 0:
        return 0.0, 0.0
    z = 1.96
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _rotate(names: list[PolicySpec], offset: int) -> list[PolicySpec]:
    offset %= len(names)
    return names[offset:] + names[:offset]


def _parse_policy_spec(value: str, parser: argparse.ArgumentParser) -> PolicySpec:
    name, separator, parameter = value.partition(":")
    if name not in POLICIES:
        parser.error(f"Política desconhecida: {value!r}. Use random ou greedy[:RESERVA].")
    if name == "random":
        if separator:
            parser.error("A política random não aceita parâmetros.")
        return PolicySpec(name="random")
    if not separator:
        return PolicySpec(name="greedy")
    try:
        reserve = int(parameter)
    except ValueError:
        parser.error(f"Reserva inválida em {value!r}; use um inteiro não negativo.")
    if reserve < 0:
        parser.error(f"Reserva inválida em {value!r}; use um inteiro não negativo.")
    return PolicySpec(name="greedy", reserve=reserve, explicit_reserve=True)


def main() -> None:
    """Run seeded baseline matchups and print aggregate and optional CSV data."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["greedy", "random"],
        metavar="POLICY",
        help="random, greedy ou greedy:RESERVA (por exemplo, greedy:150)",
    )
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument(
        "--no-auction", action="store_true", help="desliga o leilão de imóvel recusado"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv", metavar="PATH", help="salva uma linha por jogador e partida nesse arquivo"
    )
    args = parser.parse_args()
    policy_specs = [_parse_policy_spec(value, parser) for value in args.policies]

    if args.games < 1:
        parser.error("--games deve ser positivo.")
    if not 2 <= len(policy_specs) <= 12:
        parser.error("Informe entre 2 e 12 políticas, uma por assento.")
    if args.max_turns < 1:
        parser.error("--max-turns deve ser positivo.")

    config = load_selected_game()
    stats = defaultdict(_new_metrics)
    total_truncated = 0
    total_without_winner = 0
    total_actions = 0
    total_turns = 0
    game_records: list[dict[str, object]] = []
    started_at = time.perf_counter()

    for game_index in range(args.games):
        seed = args.seed + game_index
        seat_policies = _rotate(policy_specs, game_index)
        loop = create_game(
            [f"P{seat}" for seat in range(len(seat_policies))],
            config,
            seed=seed,
            max_turns=args.max_turns,
            auction_enabled=False if args.no_auction else None,
            record_events=False,
        )
        policies = {}
        for seat, (player_id, name) in enumerate(
            zip(loop.game.turn_order, seat_policies)
        ):
            policy = name.create()
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset(seed * 1_000_003 + seat)
            policies[player_id] = (name.label, policy)

        while not loop.game.game_over:
            name, policy = policies[loop.current_decision_player_id]
            result = loop.process_action(policy(loop))
            if result["status"] != "SUCCESS":
                raise RuntimeError(
                    f"Política {name!r} produziu ação inválida na partida {game_index}, "
                    f"seed {seed}: {result.get('reason')}"
                )
            total_actions += 1

        game = loop.game
        total_turns += game.turn_counter
        total_truncated += int(game.truncated)
        total_without_winner += int(game.winner_id is None and not game.truncated)
        winner_policy = policies[game.winner_id][0] if game.winner_id is not None else None
        participating_labels = {name for name, _ in policies.values()}
        for label in participating_labels:
            values = stats[label]
            values["matches"] += 1
            values["decided_matches"] += int(game.winner_id is not None)
            values["match_wins"] += int(winner_policy == label)
            values["match_losses"] += int(
                game.winner_id is not None and winner_policy != label
            )
            # A truncagem é um episódio inconclusivo, não um empate.
            values["match_draws"] += int(game.winner_id is None and not game.truncated)
            values["truncated_games"] += int(game.truncated)
            values["no_winner_games"] += int(game.winner_id is None and not game.truncated)

        for player_id, (name, _) in policies.items():
            player = game.player_states[player_id]
            values = stats[name]
            values["appearances"] += 1
            values["player_wins"] += int(game.winner_id == player_id)
            values["bankruptcies"] += int(player.is_bankrupt)
            values["cash"] += player.money
            values["net_worth"] += game.net_worth(player_id)
            values["properties"] += len(game.owned_tiles(player_id))
            values["turns"] += game.turn_counter
            game_records.append({
                "game_index": game_index,
                "seed": seed,
                "seat": loop.game.turn_order.index(player_id),
                "policy": name,
                "winner_policy": winner_policy or "",
                "won": game.winner_id == player_id,
                "match_decided": game.winner_id is not None,
                "truncated": game.truncated,
                "no_winner": game.winner_id is None and not game.truncated,
                "turns": game.turn_counter,
                "cash": player.money,
                "net_worth": game.net_worth(player_id),
                "bankrupt": player.is_bankrupt,
                "properties": len(game.owned_tiles(player_id)),
            })

    elapsed = time.perf_counter() - started_at
    print(
        f"{args.games} partidas | {len(policy_specs)} jogadores | "
        f"seeds {args.seed}–{args.seed + args.games - 1}"
    )
    print(
        f"{total_truncated / args.games:.1%} truncadas | "
        f"{total_without_winner / args.games:.1%} sem vencedor | "
        f"{total_turns / args.games:.1f} turnos médios | "
        f"{total_actions / elapsed:,.0f} ações/s | {elapsed:.1f}s"
    )

    distinct_labels = tuple(dict.fromkeys(spec.label for spec in policy_specs))
    seats_per_policy = {
        label: sum(spec.label == label for spec in policy_specs)
        for label in distinct_labels
    }
    for label in distinct_labels:
        values = stats[label]
        matches = int(values["matches"])
        match_wins = int(values["match_wins"])
        match_losses = int(values["match_losses"])
        match_draws = int(values["match_draws"])
        truncated_matches = int(values["truncated_games"])
        decisive_matches = int(values["decided_matches"])
        match_rate = match_wins / matches if matches else 0.0
        decisive_rate = match_wins / decisive_matches if decisive_matches else 0.0
        appearances = int(values["appearances"])
        seat_wins = int(values["player_wins"])
        low, high = _wilson_interval(match_wins, matches)
        if len(distinct_labels) > 1:
            matchup_name = (
                f"resultado do grupo de {seats_per_policy[label]} assentos"
                if seats_per_policy[label] > 1
                else "resultado por partida"
            )
            outcome_summary = (
                f"{matchup_name}: W–L–D {match_wins}–{match_losses}–{match_draws} "
                f"em {matches} partidas "
                f"({match_rate:.1%}; IC95% {low:.1%}–{high:.1%}; "
                f"{decisive_rate:.1%} nas decididas; "
                f"{truncated_matches} truncadas)"
            )
        else:
            outcome_summary = (
                f"vitórias por assento {seat_wins}/{appearances} "
                f"({seat_wins / appearances:.1%})"
            )
        print(
            f"{label} ({appearances} assentos) : {outcome_summary} | "
            f"vitórias por assento {seat_wins}/{appearances} "
            f"({seat_wins / appearances:.1%}) | "
            f"falências {int(values['bankruptcies']) / appearances:.1%} | "
            f"caixa médio {int(values['cash']) / appearances:.0f} | "
            f"patrimônio médio {int(values['net_worth']) / appearances:.0f} | "
            f"imóveis médios {int(values['properties']) / appearances:.1f}"
        )

    if args.csv:
        fields = (
            "game_index", "seed", "seat", "policy", "winner_policy", "won",
            "match_decided", "truncated", "no_winner", "turns", "cash",
            "net_worth", "bankrupt", "properties",
        )
        output_path = Path(args.csv)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(game_records)
        print(f"resultados por jogador/partida: {output_path}")


if __name__ == "__main__":
    main()
