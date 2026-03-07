"""Extraction ledger utilities."""

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write JSON data to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(data: dict[str, Any], output_path: Path) -> None:
    """Append JSON line to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
