"""Picks the lineup to show for a spike position and a player position."""

from . import geometry

# Map coordinates are percent of the map image; 1% is roughly 1.4 m in game.
SPIKE_RADIUS = 3.5

# Abilities that stop or punish a defuse. Lineups using one of these win over
# anything else that happens to land near the spike (smokes, recon, traps).
POST_PLANT_ABILITIES = {
    "Aftershock", "Incendiary", "Orbital Strike", "Mosh Pit", "FRAG/ment", "Nanoswarm",
    "Hot Hands", "Paint Shells", "Shock Bolt", "Double Shock Bolt", "Hunter's Fury",
    "Snake Bite", "Guided Salvo", "Armageddon",
}


def post_plant_ability_ids(agent):
    return {ability["id"] for ability in agent["abilities"] if ability["name"] in POST_PLANT_ABILITIES}


def choose(lineups, spike, player=None, preferred_abilities=(), radius=SPIKE_RADIUS):
    """The lineup landing within `radius` of the spike whose standing spot is nearest the player.

    Lineups using a preferred ability are considered first. Without a player
    position, the lineup landing closest to the spike wins. Returns None when
    nothing lands on the spike.
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
    if player is None:
        return min(candidates, key=lambda item: item[1])[0]
    return min(candidates, key=lambda item: geometry.distance((item[0]["left"], item[0]["top"]), player))[0]


def group_of(groups, lineup):
    return next(group for group in groups if any(item["id"] == lineup["id"] for item in group["lineups"]))
