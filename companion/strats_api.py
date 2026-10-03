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


def aim_image(lineup_id):
    """The lineup's aim screenshot (BGR), or None when it has none or it cannot be had.

    Fetched from cdn.strats.gg, or read from the Strats.gg app's cache: opening
    the lineup in the app loads the full-size picture.
    """
    path = CACHE_DIR / "images" / f"{lineup_id}.img"
    if not path.exists():
        try:
            key = lineup_detail(lineup_id).get("imageKey")
        except (Unavailable, requests.RequestException):
            return None
        if not key:
            return None
        try:
            response = requests.get(f"{CDN_URL}/{key}", timeout=20)
            response.raise_for_status()
            content = response.content
        except requests.RequestException:
            from . import app_cache

            content = app_cache.file(key)
        if not content:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    return image


def lineups_path(map_id, agent_id, side):
    """Where the saved copy of one map/agent/side's lineups lives."""
    return CACHE_DIR / f"lineups_{map_id}_{agent_id}_{side}.json"


def find_by_name(items, name):
    wanted = slugify(name)
    return next((item for item in items if slugify(item["name"]) == wanted), None)
