"""Representação textual do estado e das ações (render e agentes baseados em LLM)."""

from app.domain.game.actions import ActionType, GameAction
from app.domain.game.game_loop import GameLoop
from app.domain.game.game_state import HOTEL_LEVEL


def describe_action(loop: GameLoop, action: GameAction) -> str:
    board = loop.game.board
    if "tile_id" in action.payload:
        return f"{action.action_type} {board.get_tile_by_id(action.payload['tile_id']).name}"
    if action.action_type == ActionType.AUCTION_BID:
        return f"AUCTION_BID ${action.payload['amount']}"
    if action.action_type == ActionType.BUY_PROPERTY and loop.pending_tile_id is not None:
        tile = board.get_tile_by_id(loop.pending_tile_id)
        return f"BUY_PROPERTY {tile.name} (${tile.price})"
    return str(action.action_type)


def render_text(loop: GameLoop) -> str:
    game = loop.game
    seat = {player_id: i for i, player_id in enumerate(game.turn_order)}
    lines = [
        f"Turno {game.turn_counter} | fase {loop.turn_phase} | "
        f"decide: P{seat[loop.current_decision_player_id]}"
    ]

    for player_id in game.turn_order:
        state = game.player_states[player_id]
        name = f"P{seat[player_id]} ({game.players[player_id].name})"
        if state.is_bankrupt:
            lines.append(f"  {name}: FALIDO")
            continue
        props = []
        for tile in game.owned_tiles(player_id):
            prop = game.property_states[tile.tile_id]
            tag = "H" if prop.houses == HOTEL_LEVEL else (f"{prop.houses}c" if prop.houses else "")
            tag += "*hip" if prop.is_mortgaged else ""
            props.append(tile.name + (f"({tag})" if tag else ""))
        tile_name = game.board.get_tile(game.position_of(player_id)).name
        jail = f" [preso {state.jail_turns}]" if state.is_in_jail else ""
        lines.append(
            f"  {name}: ${state.money} em {tile_name}{jail} | "
            f"{', '.join(props) or 'sem imóveis'}"
        )

    if loop.pending_tile_id is not None:
        tile = game.board.get_tile_by_id(loop.pending_tile_id)
        extra = f" | maior lance ${loop.auction_highest_bid}" if loop.turn_phase == GameLoop.AUCTION else ""
        lines.append(f"  Em jogo: {tile.name} (${tile.price}){extra}")

    if game.game_over:
        winner = "nenhum (truncada)" if game.winner_id is None else f"P{seat[game.winner_id]}"
        lines.append(f"  FIM: vencedor = {winner}")
    else:
        lines.append("  Ações legais: " + "; ".join(describe_action(loop, a) for a in loop.legal_actions()))
    return "\n".join(lines)
