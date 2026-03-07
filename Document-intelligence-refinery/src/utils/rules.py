"""Configuration rules loader."""

import yaml
from pathlib import Path


def load_rules(rules_path: str | Path) -> dict:
    """Load extraction rules from YAML file."""
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
