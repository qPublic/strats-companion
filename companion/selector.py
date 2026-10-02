"""Picks the lineup to show for a spike position and a player position."""

from . import geometry

# Map coordinates are percent of the map image; 1% is roughly 1.4 m in game.
SPIKE_RADIUS = 3.5
# How far from the player a standing spot may be and still count as "not too far".
MAX_WALK = 25.0

# Abilities that stop or punish a defuse. Lineups using one of these win over
# anything else that happens to land near the spike (smokes, recon, traps).
POST_PLANT_ABILITIES = {
    "Aftershock", "Incendiary", "Orbital Strike", "Mosh Pit", "FRAG/ment", "Nanoswarm",
    "Hot Hands", "Paint Shells", "Shock Bolt", "Double Shock Bolt", "Hunter's Fury",
    "Snake Bite", "Guided Salvo", "Armageddon",
}


def post_plant_ability_ids(agent):
    return {ability["id"] for ability in agent["abilities"] if ability["name"] in POST_PLANT_ABILITIES}


def choose(lineups, spike, player=None, preferred_abilities=(), spawn=None, radius=SPIKE_RADIUS):
    """The best lineup landing within `radius` of the spike, or None when nothing lands there.

    Lineups using a preferred ability are considered first. Among those, the
    one thrown from closest to the team's `spawn` wins, as long as its standing
    spot is within MAX_WALK of the player; if none is that close, the nearest
    standing spot wins. Without a player position, closeness to spawn decides,
    and without a spawn either, the landing closest to the spike.
    """
    candidates = []
    for lineup in lineups:
        landing = geometry.landing_point(lineup)
        if landing is None:
            continue
        miss = geometry.distance(landing, spike)
        if miss <= radius:
            candidates.append((lineup, miss))
    preferred = [item for item in candidates if item[0]["abilityId"] in preferred_abilities]
    candidates = preferred or candidates
    if not candidates:
        return None

    def standing(item):
        return item[0]["left"], item[0]["top"]

    if player is not None:
        reachable = [item for item in candidates if geometry.distance(standing(item), player) <= MAX_WALK]
        if not reachable or spawn is None:
            return min(candidates, key=lambda item: geometry.distance(standing(item), player))[0]
        candidates = reachable
    if spawn is not None:
        return min(candidates, key=lambda item: geometry.distance(standing(item), spawn))[0]
    return min(candidates, key=lambda item: item[1])[0]


def group_of(groups, lineup):
    return next(group for group in groups if any(item["id"] == lineup["id"] for item in group["lineups"]))
