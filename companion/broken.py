"""Lineups the player has marked as not working (map changes can break a bounce), never picked again."""

import json

from .paths import DATA_DIR

BROKEN_FILE = DATA_DIR / "broken_lineups.json"


def _load():
    try:
        return json.loads(BROKEN_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ids():
    return {int(lineup_id) for lineup_id in _load()}


def mark(lineup):
    broken = _load()
    broken[str(lineup["id"])] = lineup["title"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BROKEN_FILE.write_text(json.dumps(broken, indent=1), encoding="utf-8")


def clear():
    """Forget every mark; returns how many there were."""
    count = len(_load())
    BROKEN_FILE.unlink(missing_ok=True)
    return count
