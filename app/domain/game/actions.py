from typing import Any, Dict, Optional

class GameAction:
    """Representa qualquer comando enviado de fora para a Engine."""
    def __init__(self, action_type: str, payload: Optional[Dict[str, Any]] = None):
        # Tipos: 'ROLL_DICE', 'BUY_PROPERTY', 'DECLINE_PROPERTY', 'USE_JAIL_CARD', 'BUILD_HOUSE', 'END_TURN'
        self.action_type: str = action_type  
        self.payload: Dict[str, Any] = payload or {}