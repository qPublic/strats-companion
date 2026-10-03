"""Read-only client for the public Strats.gg lineup data.

These are the same queries the Strats.gg app makes when you open its lineup
tool. Results are cached on disk so each map/agent pair is fetched at most
once per CACHE_TTL.
"""

import json
import re
import time

import cv2
import numpy as np
import requests

from . import devalue
from .paths import CACHE_DIR

API_URL = "https://strats.gg/api/trpc"
CDN_URL = "https://cdn.strats.gg"
CACHE_TTL = 6 * 60 * 60


class Unavailable(RuntimeError):
    """The lineup data could not be fetched and no earlier copy is on disk."""


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _query(procedure, payload=None):
    params = {"batch": "1", "input": json.dumps({"0": devalue.stringify(payload)})}
    response = requests.get(f"{API_URL}/{procedure}", params=params, timeout=15)
    response.raise_for_status()
    body = response.json()[0]
    if "error" in body:
        raise RuntimeError(f"Strats.gg API error for {procedure}: {body['error']}")
    return devalue.parse(body["result"]["data"])


def _cached(name, fetch):
    path = CACHE_DIR / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        data = fetch()
    except (requests.RequestException, ValueError) as error:
        # Strats.gg sometimes puts a browser check in front of its API. The Strats.gg app may
        # have loaded the same data itself; failing that, an old copy still works.
        from . import app_cache

        app_cache.harvest()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        status = getattr(getattr(error, "response", None), "status_code", None)
        reason = f"HTTP {status}" if status else type(error).__name__
        raise Unavailable(f"Strats.gg is refusing data requests right now ({reason}) and no saved copy exists") from error
    save(path, data)
    return data


def save(path, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def maps():
    return _cached("maps", lambda: _query("valorant.lineups.maps"))


def agents():
    return _cached("agents", lambda: _query("valorant.lineups.agents"))


def lineups(map_id, agent_id, side):
    payload = {"agentId": agent_id, "mapId": map_id, "side": side, "tag": "all", "alignment": "all"}
    return _cached(lineups_path(map_id, agent_id, side).stem, lambda: _query("valorant.lineups.all", payload))


def detail_path(lineup_id):
    return CACHE_DIR / f"lineup_{lineup_id}.json"


def lineup_detail(lineup_id):
    """Everything about one lineup, including `imageKey`, its aim screenshot."""
    return _cached(detail_path(lineup_id).stem, lambda: _query("valorant.lineups.byId", {"id": lineup_id}))


AIM_DIR = CACHE_DIR / "aim"
PREVIEW_WIDTH = 640


def aim_stored(lineup_id):
    """Whether the aim data for a lineup is on disk (or it is known to have none)."""
    return (AIM_DIR / f"{lineup_id}.npz").exists() or (AIM_DIR / f"{lineup_id}.none").exists()


def aim_data(lineup_id, strict=False):
    """(AimGuide, preview picture) for a lineup's aim screenshot, or (None, None).

    Stored on disk after the first time: the screenshot's features and a small
    preview, about 100 KB, instead of the full picture. The screenshot comes
    from cdn.strats.gg, or from the Strats.gg app's cache once the lineup has
    been opened there. With `strict`, a refused download raises Unavailable
    instead of giving (None, None), so a background download can wait and retry.
    """
    from .aim import AimGuide

    stored, preview_path = AIM_DIR / f"{lineup_id}.npz", AIM_DIR / f"{lineup_id}.jpg"
    if stored.exists():
        preview = cv2.imread(str(preview_path)) if preview_path.exists() else None
        return AimGuide.load(stored), preview
    if (AIM_DIR / f"{lineup_id}.none").exists():
        return None, None
    earlier = CACHE_DIR / "images" / f"{lineup_id}.img"      # full screenshots saved by version 0.18-0.21
    if earlier.exists():
        picture = cv2.imdecode(np.frombuffer(earlier.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if picture is not None:
            return _store_aim(lineup_id, picture)
    try:
        key = lineup_detail(lineup_id).get("imageKey")
    except (Unavailable, requests.RequestException):
        if strict:
            raise
        return None, None
    AIM_DIR.mkdir(parents=True, exist_ok=True)
    if not key:
        (AIM_DIR / f"{lineup_id}.none").touch()
        return None, None
    try:
        response = requests.get(f"{CDN_URL}/{key}", timeout=20)
        response.raise_for_status()
        content = response.content
    except requests.RequestException as error:
        from . import app_cache

        content = app_cache.file(key)
        if not content:
            if strict:
                raise Unavailable(f"Strats.gg is refusing the aim screenshot for lineup {lineup_id}") from error
            return None, None
    picture = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        return None, None
    return _store_aim(lineup_id, picture)


def _store_aim(lineup_id, picture):
    from .aim import AimGuide

    AIM_DIR.mkdir(parents=True, exist_ok=True)
    guide = AimGuide.from_picture(picture)
    guide.save(AIM_DIR / f"{lineup_id}.npz")
    preview = cv2.resize(picture, (PREVIEW_WIDTH, int(PREVIEW_WIDTH * picture.shape[0] / picture.shape[1])),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(AIM_DIR / f"{lineup_id}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (CACHE_DIR / "images" / f"{lineup_id}.img").unlink(missing_ok=True)
    return guide, preview


def lineups_path(map_id, agent_id, side):
    """Where the saved copy of one map/agent/side's lineups lives."""
    return CACHE_DIR / f"lineups_{map_id}_{agent_id}_{side}.json"


def find_by_name(items, name):
    wanted = slugify(name)
    return next((item for item in items if slugify(item["name"]) == wanted), None)
