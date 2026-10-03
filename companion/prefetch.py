"""Save the lineup data for every agent on every map ahead of time.

Strats.gg sometimes turns away data requests. Anything saved here keeps
working while it does, so this runs in the background and retries until
every combination is on disk.
"""

import requests

from . import app_cache, map_shape, strats_api

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
                    on_log(f"Saving lineup data for every agent on every map, {', '.join(FIRST)} first ({len(jobs)} downloads)...")
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
            on_log("Lineup data for every agent is saved for every map.")
        return
