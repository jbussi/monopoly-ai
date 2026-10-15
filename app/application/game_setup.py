
from app.core.config import get_app_settings
from app.infraestructure.game_config.loader import load_game_config
from app.infraestructure.game_config.models import GameConfig


def load_selected_game() -> GameConfig:
    settings = get_app_settings()

    return load_game_config(
        config_dir=settings.game_config_dir,
        variant=settings.game_variant,
    )
