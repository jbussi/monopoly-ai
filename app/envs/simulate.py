"""Simula partidas entre políticas de referência e mostra estatísticas.

    python -m app.envs.simulate --games 200 --policies greedy random random random
"""

import argparse
import time
from collections import Counter

from app.application.game_setup import create_game, load_selected_game
from app.envs.baselines import GreedyPolicy, RandomPolicy

POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--policies", nargs="+", default=["greedy", "random"], choices=sorted(POLICIES))
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--no-auction", action="store_true", help="desliga o leilão de imóvel recusado")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_selected_game()
    wins: Counter = Counter()
    truncated = actions = turns = 0
    start = time.perf_counter()

    for i in range(args.games):
        seed = args.seed + i
        loop = create_game(
            [f"P{k}" for k in range(len(args.policies))], config, seed=seed,
            max_turns=args.max_turns, auction_enabled=False if args.no_auction else None,
            record_events=False,
        )
        policies = {}
        for k, (player_id, name) in enumerate(zip(loop.game.turn_order, args.policies)):
            policy = POLICIES[name]()
            if hasattr(policy, "reset"):
                policy.reset(seed * 100 + k)
            policies[player_id] = policy

        while not loop.game.game_over:
            loop.process_action(policies[loop.current_decision_player_id](loop))
            actions += 1

        turns += loop.game.turn_counter
        if loop.game.truncated:
            truncated += 1
        else:
            wins[loop.game.turn_order.index(loop.game.winner_id)] += 1

    elapsed = time.perf_counter() - start
    print(f"{args.games} partidas, {len(args.policies)} jogadores, {actions / elapsed:,.0f} ações/s")
    for seat, name in enumerate(args.policies):
        print(f"  P{seat} ({name}): {wins[seat] / args.games:.1%} vitórias")
    print(f"  truncadas: {truncated / args.games:.1%} | turnos médios: {turns / args.games:.0f}")


if __name__ == "__main__":
    main()
