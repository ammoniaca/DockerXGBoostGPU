import json
from pathlib import Path
from src.dto.config_dto import ConfigDTO


def load_config(file_path: Path) -> ConfigDTO:
    """Load and validate a JSON configuration file."""
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return ConfigDTO.model_validate(data)