"""Lineup geometry in Strats.gg map coordinates (percent of the square map image)."""

import math

GROUP_GRID_SIZE = 2.5


def landing_point(lineup):
    points = lineup.get("lineupPoints") or []
    if not points:
        return None
    last = points[-1]
    if not last.get("left") or not last.get("top"):
        return None
    return last["left"], last["top"]


def group_lineups(lineups):
    """Mirror of Strats.gg's own marker grouping, so group markers land where the app draws them."""
    buckets = {}
    for lineup in lineups:
        landing = landing_point(lineup)
        if landing is None:
            continue
        left, top = landing
        key = (math.floor(left / GROUP_GRID_SIZE), math.floor(top / GROUP_GRID_SIZE), lineup["abilityId"])
        bucket = buckets.setdefault(key, {"lineups": [], "coordinates": []})
        bucket["lineups"].append(lineup)
        bucket["coordinates"].append((left, top))
    groups = []
    for bucket in buckets.values():
        coordinates = bucket["coordinates"]
        left_avg = sum(c[0] for c in coordinates) / len(coordinates)
        top_avg = sum(c[1] for c in coordinates) / len(coordinates)
        centroid = min(coordinates, key=lambda c: math.hypot(c[0] - left_avg, c[1] - top_avg))
        groups.append({"index": len(groups), "point": centroid, "lineups": bucket["lineups"]})
    return groups


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def metres(map_item, a, b):
    """In-game distance between two map points.

    Strats.gg's map art has the same framing as Riot's minimap image, so Riot's
    xMultiplier (image fraction per game unit, a game unit being 1 cm) applies.
    """
    return distance(a, b) / 100 / abs(map_item["xMultiplier"]) / 100


def map_distance(map_item, metres):
    """A distance in metres as map coordinates (percent of the map image) on this map."""
    return metres * 100 * abs(map_item["xMultiplier"]) * 100


def standing_spot(lineup):
    return lineup["left"], lineup["top"]
