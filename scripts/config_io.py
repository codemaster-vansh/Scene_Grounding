"""
scripts/config_io.py
Helpers to read and write the repo-level config.json
"""

from pathlib import Path
import json
import os
from typing import Any, Dict

# ---------------------------------------------------------------------------
def _find_config() -> Path:
    """
    Resolve <repo_root>/config.json, no matter where the caller lives.

    Priority
    1. $CONFIG_PATH  environment variable (manual override)
    2. First parent that already contains config.json
    3. Assume this file sits in <repo_root>/scripts/, so repo root is parent.
    """
    # 1 ─ explicit override
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # 2 ─ climb parents until config.json found
    for parent in Path(__file__).resolve().parents:
        cand = parent / "config.json"
        if cand.exists():
            return cand

    # 3 ─ fallback: repo root is parent of scripts/
    return Path(__file__).resolve().parents[1] / "config.json"


CFG_PATH = _find_config()        # resolved once at import time
# ---------------------------------------------------------------------------


def save_to_config(new_items: Dict[str, Any], cfg_path: Path = CFG_PATH) -> None:
    """
    Merge `new_items` into config.json (creates the file if missing).
    Existing keys are overwritten only if present in `new_items`.
    """
    cfg_path = cfg_path.resolve()
    cfg = {}

    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    cfg.update(new_items)

    tmp = cfg_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=4)
    tmp.replace(cfg_path)        # atomic move

    print(f"Updated {cfg_path} with: {', '.join(new_items)}")


def load_config(cfg_path: Path = CFG_PATH) -> Dict[str, Any]:
    """Return the entire config dict (raises if file is missing)."""
    with open(cfg_path) as f:
        return json.load(f)


def get_config_value(key: str, default: Any = None, cfg_path: Path = CFG_PATH) -> Any:
    """
    Fetch a single key from config.json.
    Returns `default` if the key (or the file) is absent.
    """
    try:
        return load_config(cfg_path).get(key, default)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
