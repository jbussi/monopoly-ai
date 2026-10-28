
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class AppSettings:
    project_root: Path
    configs_dir: Path
    game_variant: str

    @property
    def game_config_dir(self) -> Path:
        return self.configs_dir / "games" / self.game_variant


def get_app_settings() -> AppSettings:
    variant = os.getenv("MONOPOLY_GAME_VARIANT", "classic_us")

    if not variant or Path(variant).name != variant:
        raise ValueError("MONOPOLY_GAME_VARIANT inválida.")

    return AppSettings(
        project_root=PROJECT_ROOT,
        configs_dir=PROJECT_ROOT / "configs",
        game_variant=variant,
    )
