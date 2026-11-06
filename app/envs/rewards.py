"""Placement rewards shared by the Gymnasium and PettingZoo environments."""

from collections.abc import Callable
from uuid import UUID

from app.domain.game.game_loop import GameLoop

RewardFn = Callable[[GameLoop, UUID], float]
GEOMETRIC_REWARD_RATIO = 0.4
SURVIVAL_BONUS = 0.05


def placement_rewards(num_players: int) -> tuple[float, ...]:
    """Return the symmetric geometric reward assigned to each final place."""
    if num_players < 2:
        raise ValueError("Uma partida precisa ter pelo menos dois jogadores.")

    return tuple(
        _placement_reward(num_players, position)
        for position in range(1, num_players + 1)
    )


def _placement_reward(num_players: int, position: int) -> float:
    """Calculate the signed geometric reward for one one-based placement."""
    if not 1 <= position <= num_players:
        raise ValueError("A posição precisa estar entre 1 e o número de jogadores.")
    sign_value = num_players + 1 - 2 * position
    sign = (sign_value > 0) - (sign_value < 0)
    distance = min(position, num_players + 1 - position) - 1
    return sign * GEOMETRIC_REWARD_RATIO**distance


def _elimination_rank(loop: GameLoop, player_id: UUID) -> int | None:
    """Return the player's one-based final place from the recorded exit order."""
    order = loop.game.elimination_order
    try:
        return len(loop.game.turn_order) - order.index(player_id)
    except ValueError:
        return None


def win_loss_reward(loop: GameLoop, player_id: UUID) -> float:
    """Score final placement, including known placements at a turn-limit truncation."""
    game = loop.game
    player_count = len(game.turn_order)
    rewards = placement_rewards(player_count)

    if game.player_states[player_id].is_bankrupt:
        rank = _elimination_rank(loop, player_id)
        # Bankruptcy is always recorded by GameLoop; keep a defensive worst-place
        # fallback for external callers that manually mutate PlayerState.
        return rewards[rank - 1] if rank is not None else rewards[-1]

    if game.winner_id == player_id:
        return rewards[0]

    if game.truncated:
        survivor_count = len(game.active_player_ids)
        if survivor_count == 1:
            return rewards[0]
        last_survivor_placement_reward = rewards[survivor_count - 1]
        if last_survivor_placement_reward > 0.0:
            return (1.0 + SURVIVAL_BONUS) * last_survivor_placement_reward
        return GEOMETRIC_REWARD_RATIO**survivor_count

    return 0.0
