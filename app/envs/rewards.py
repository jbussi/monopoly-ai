"""Funções de recompensa.

Assinatura: `reward_fn(loop, player_id) -> float`. É chamada depois de cada ação,
para cada jogador ainda na partida. A recompensa padrão é esparsa e não impõe
estratégia; shaping (por patrimônio, por exemplo) fica com quem treina.
"""

from collections.abc import Callable
from uuid import UUID

from app.domain.game.game_loop import GameLoop

RewardFn = Callable[[GameLoop, UUID], float]


def win_loss_reward(loop: GameLoop, player_id: UUID) -> float:
    """+1 ao vencedor, -1 a quem vai à falência, 0 no resto (inclusive na truncagem)."""
    if loop.game.player_states[player_id].is_bankrupt:
        return -1.0
    if loop.game.winner_id == player_id:
        return 1.0
    return 0.0
