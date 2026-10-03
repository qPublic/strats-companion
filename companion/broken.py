"""Marks the player has put on lineups: ones that do not work (map changes can break a bounce), which are
never picked again, and ones whose Strats.gg picture is of something else, which are shown without it."""

import json

from .paths import DATA_DIR

BROKEN_FILE = DATA_DIR / "broken_lineups.json"
PICTURE_FILE = DATA_DIR / "wrong_pictures.json"


def _load(file):
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _mark(file, lineup):
    marks = _load(file)
    marks[str(lineup["id"])] = lineup["title"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(marks, indent=1), encoding="utf-8")


def ids():
    return {int(lineup_id) for lineup_id in _load(BROKEN_FILE)}


def mark(lineup):
    _mark(BROKEN_FILE, lineup)


def picture_ids():
    """Lineups whose aim picture is not to be shown or matched against."""
    return {int(lineup_id) for lineup_id in _load(PICTURE_FILE)}


def mark_picture(lineup):
    _mark(PICTURE_FILE, lineup)


def count():
    return len(_load(BROKEN_FILE)) + len(_load(PICTURE_FILE))


def clear():
    """Forget every mark of both kinds; returns how many there were."""
    marks = count()
    BROKEN_FILE.unlink(missing_ok=True)
    PICTURE_FILE.unlink(missing_ok=True)
    return marks
