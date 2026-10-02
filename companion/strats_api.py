"""Read-only client for the public Strats.gg lineup data.

These are the same queries the Strats.gg app makes when you open its lineup
tool. Results are cached on disk so each map/agent pair is fetched at most
once per CACHE_TTL.
"""

import json
import re
import time

import requests

from . import devalue
from .paths import CACHE_DIR

API_URL = "https://strats.gg/api/trpc"
CACHE_TTL = 6 * 60 * 60


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
    data = fetch()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def maps():
    return _cached("maps", lambda: _query("valorant.lineups.maps"))


def agents():
    return _cached("agents", lambda: _query("valorant.lineups.agents"))


def lineups(map_id, agent_id, side):
    payload = {"agentId": agent_id, "mapId": map_id, "side": side, "tag": "all", "alignment": "all"}
    return _cached(f"lineups_{map_id}_{agent_id}_{side}", lambda: _query("valorant.lineups.all", payload))


def find_by_name(items, name):
    wanted = slugify(name)
    return next((item for item in items if slugify(item["name"]) == wanted), None)
