"""Save the lineup data for every agent on every map ahead of time, then the aim data for their mollies.

Strats.gg sometimes turns away data requests. Anything saved here keeps
working while it does, so this runs in the background and retries until
every combination is on disk.
"""

import json

import requests

from . import app_cache, map_shape, selector, spot, strats_api

FIRST = ("Brimstone", "Viper", "Killjoy")   # saved before the other agents
SIDES = ("attack", "defense")
RETRY_SECONDS = 30
PAUSE_SECONDS = 1.0     # between downloads, to stay gentle on Strats.gg


def missing():
    """Downloads still needed, as (description, function) pairs."""
    maps = [item for item in strats_api.maps() if item.get("showInLineups", True)]
    agents = [item for item in strats_api.agents() if item.get("showInLineups", True)]
    agents.sort(key=lambda item: item["name"] not in FIRST)
    jobs = []
    for map_item in maps:
        for side in SIDES:
            if not map_shape.svg_path(map_item, side).exists():
                jobs.append((f"{map_item['name']} {side} map", lambda m=map_item, s=side: map_shape.silhouette(m, s)))
    for agent in agents:
        for map_item in maps:
            for side in SIDES:
                if not strats_api.lineups_path(map_item["id"], agent["id"], side).exists():
                    jobs.append((
                        f"{map_item['name']} / {agent['name']} / {side}",
                        lambda m=map_item, a=agent, s=side: strats_api.lineups(m["id"], a["id"], s),
                    ))
    # Then the aim screenshots of every lineup the companion would pick (the agent's mollies), stored
    # as matching features plus a small preview, so the in-game guide needs nothing at plant time.
    for agent in agents:
        mollies = selector.post_plant_ability_ids(agent)
        if not mollies:
            continue
        for map_item in maps:
            for side in SIDES:
                path = strats_api.lineups_path(map_item["id"], agent["id"], side)
                if not path.exists():
                    continue
                for lineup in json.loads(path.read_text(encoding="utf-8")):
                    if lineup["status"] == "approved" and lineup["abilityId"] in mollies \
                            and not strats_api.aim_stored(lineup["id"]):
                        jobs.append((
                            f"aim for {lineup['title']}",
                            lambda lineup_id=lineup["id"]: strats_api.aim_data(lineup_id, strict=True),
                        ))
    # Last, the true standing spots, read from the minimap in each stored aim screenshot.
    for agent in agents:
        mollies = selector.post_plant_ability_ids(agent)
        if not mollies:
            continue
        for map_item in maps:
            for side in SIDES:
                path = strats_api.lineups_path(map_item["id"], agent["id"], side)
                if not path.exists():
                    continue
                for lineup in json.loads(path.read_text(encoding="utf-8")):
                    if lineup["status"] == "approved" and lineup["abilityId"] in mollies \
                            and spot.minimap_path(lineup["id"]).exists() and not spot.known(lineup["id"]):
                        jobs.append((
                            f"spot for {lineup['title']}",
                            lambda item={**lineup, "side": side}, m=map_item: spot.true_spot(item, m),
                        ))
    return jobs


def run(stop, on_log=print):
    """Download whatever is missing, checking again every RETRY_SECONDS until nothing is, then stop."""
    downloaded, reported = 0, None
    while not stop.is_set():
        jobs, index = None, 0
        try:
            app_cache.harvest()      # whatever the Strats.gg app has loaded since the last pass
            jobs = missing()
            for index, (description, fetch) in enumerate(jobs):
                if stop.is_set():
                    return
                fetch()
                if downloaded == 0:
                    on_log(f"Saving lineup and aim data for every agent on every map, {', '.join(FIRST)} first ({len(jobs)} downloads)...")
                downloaded += 1
                stop.wait(PAUSE_SECONDS)
        except (strats_api.Unavailable, requests.RequestException):
            left = None if jobs is None else len(jobs) - index
            if left != reported:
                reported = left
                count = "the map list" if left is None else f"{left} downloads"
                on_log(f"Strats.gg is refusing downloads; {count} still to save. Checking again every {RETRY_SECONDS} seconds.")
            stop.wait(RETRY_SECONDS)
            continue
        if downloaded:
            on_log("Lineup and aim data for every agent is saved for every map.")
        return
