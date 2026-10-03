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
# After a lineup has been used this many times for the same plant, others get a turn.
REPEAT_LIMIT = 2
# Lineups landing more than this much further from the spike than the closest one are for another plant.
LANDING_SLACK = 1.0

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


def choose(map_item, lineups, spike, player=None, preferred_abilities=(), threats=(), used=None, exclude=()):
    """The best lineup landing within SPIKE_RADIUS of the spike, or None when nothing lands there."""
    best = ranked(map_item, lineups, spike, player, preferred_abilities, threats, used, exclude)
    return best[0] if best else None


def ranked(map_item, lineups, spike, player=None, preferred_abilities=(), threats=(), used=None, exclude=()):
    """Every lineup landing within SPIKE_RADIUS of the spike, best first.

    With `preferred_abilities` (the agent's mollies), only lineups using one of
    them count; an agent without any can use whatever lands there. Lineups whose
    id is in `exclude` (marked broken, or skipped while testing) are left out.
    In order of importance:
    - Lineups landing within LANDING_SLACK of the closest landing come first:
      with two plant spots close together, the other spot's lineups come later.
    - `used` counts how often each lineup was used for this plant before; ones
      used fewer than REPEAT_LIMIT times come first, then the least used.
    - Standing spots more than DANGER_RADIUS from every threat (a spotted enemy
      or a teammate's death) come first; the others follow, farthest first.
    - Standing spots inside the south-facing cone and within MAX_WALK of the
      player, then the rest of the cone, closest to the spike first; then spots
      outside the cone, nearest the player first (or nearest the spike, without
      a player position).
    """
    candidates = []
    for lineup in lineups:
        landing = geometry.landing_point(lineup)
        if landing is None or lineup["id"] in exclude:
            continue
        if geometry.distance(landing, spike) <= geometry.map_distance(map_item, SPIKE_RADIUS):
            candidates.append(lineup)
    if preferred_abilities:
        candidates = [lineup for lineup in candidates if lineup["abilityId"] in preferred_abilities]
    used = used or {}
    safe_distance = geometry.map_distance(map_item, DANGER_RADIUS)
    reach = geometry.map_distance(map_item, MAX_WALK)

    # Plants close together: a lineup made for the other one also lands within SPIKE_RADIUS. Those
    # landing clearly further from this spike than the best one come after all the close ones.
    misses = {lineup["id"]: geometry.metres(map_item, geometry.landing_point(lineup), spike) for lineup in candidates}
    closest = min(misses.values(), default=0.0)

    def order(lineup):
        spot = geometry.standing_spot(lineup)
        other_plant = misses[lineup["id"]] > closest + LANDING_SLACK
        uses = used.get(lineup["id"], 0)
        # Stay unpredictable: a lineup used REPEAT_LIMIT times for this plant gives way to the others.
        use_rank = 0 if uses < REPEAT_LIMIT else uses
        danger = min((geometry.distance(spot, threat) for threat in threats), default=float("inf"))
        danger_rank = (0, 0.0) if danger > safe_distance else (1, -danger)
        from_spike = geometry.distance(spot, spike)
        if in_cone(spot, spike):
            near_player = player is None or geometry.distance(spot, player) <= reach
            place_rank = (0 if near_player else 1, from_spike)
        else:
            place_rank = (2, from_spike if player is None else geometry.distance(spot, player))
        return other_plant, use_rank, danger_rank, place_rank

    return sorted(candidates, key=order)


def group_of(groups, lineup):
    return next(group for group in groups if any(item["id"] == lineup["id"] for item in group["lineups"]))
