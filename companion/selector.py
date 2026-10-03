"""Picks the lineup to show for a spike position and a player position."""

from . import geometry

# Distances in metres. Maps differ in size, so each is converted with that map's
# scale (see geometry.metres) before comparing with map coordinates.
# A lineup counts when it lands this close to the spike (about a molly's radius).
SPIKE_RADIUS = 4.5
# How far from the player a standing spot may be and still count as "not too far".
MAX_WALK = 35.0
# Standing spots this close to a recently spotted enemy or a teammate's death are avoided.
DANGER_RADIUS = 14.0

# Mollies and other thrown abilities that damage a defuser. For an agent with one
# of these, only these are used: never a smoke, recon or trap, and not ultimates.
POST_PLANT_ABILITIES = {
    "Aftershock", "Incendiary", "Mosh Pit", "FRAG/ment", "Nanoswarm", "Hot Hands",
    "Paint Shells", "Shock Bolt", "Double Shock Bolt", "Snake Bite", "Guided Salvo",
}


def post_plant_ability_ids(agent):
    return {ability["id"] for ability in agent["abilities"] if ability["name"] in POST_PLANT_ABILITIES}


def in_cone(spot, spike):
    """Whether `spot` lies in the right-angle cone opening south (down the map image) from the spike.

    Strats.gg draws each side's map with that side's spawn at the bottom, as a
    fixed minimap does, so south is back towards your own team's side.
    """
    across, down = spot[0] - spike[0], spot[1] - spike[1]
    return down > 0 and abs(across) <= down


def choose(map_item, lineups, spike, player=None, preferred_abilities=(), threats=()):
    """The best lineup landing within SPIKE_RADIUS of the spike, or None when nothing lands there.

    With `preferred_abilities` (the agent's mollies), only lineups using one of
    them count; an agent without any can use whatever lands there. Standing spots
    within DANGER_RADIUS of a threat (a spotted enemy or a teammate's death) are
    dropped; if every spot is that close, the one farthest from the threats
    wins. Otherwise the standing spot closest to the spike inside the
    south-facing cone wins, preferring spots within MAX_WALK of the player.
    With nothing in the cone, the standing spot nearest the player wins (or,
    without a player position, the one closest to the spike).
    """
    candidates = []
    for lineup in lineups:
        landing = geometry.landing_point(lineup)
        if landing is None:
            continue
        if geometry.distance(landing, spike) <= geometry.map_distance(map_item, SPIKE_RADIUS):
            candidates.append(lineup)
    if preferred_abilities:
        candidates = [lineup for lineup in candidates if lineup["abilityId"] in preferred_abilities]
    if not candidates:
        return None

    def danger(lineup):
        """Distance from the standing spot to the nearest threat."""
        spot = geometry.standing_spot(lineup)
        return min((geometry.distance(spot, threat) for threat in threats), default=float("inf"))

    safe = [lineup for lineup in candidates if danger(lineup) > geometry.map_distance(map_item, DANGER_RADIUS)]
    if not safe:
        return max(candidates, key=danger)

    def from_spike(lineup):
        return geometry.distance(geometry.standing_spot(lineup), spike)

    cone = [lineup for lineup in safe if in_cone(geometry.standing_spot(lineup), spike)]
    if cone:
        if player is not None:
            reachable = [lineup for lineup in cone if geometry.distance(geometry.standing_spot(lineup), player) <= geometry.map_distance(map_item, MAX_WALK)]
            cone = reachable or cone
        return min(cone, key=from_spike)
    if player is not None:
        return min(safe, key=lambda lineup: geometry.distance(geometry.standing_spot(lineup), player))
    return min(safe, key=from_spike)


def group_of(groups, lineup):
    return next(group for group in groups if any(item["id"] == lineup["id"] for item in group["lineups"]))
