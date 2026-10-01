from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any

def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration.
    
    Args:
        path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(config: dict, path: str | Path) -> None:
    """Save configuration to YAML.
    
    Args:
        config: Configuration dictionary.
        path: Output path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
