"""The watch loop: read the minimap, pick a lineup when the spike is planted, open it in Strats.gg.

Shared by the console `run` command and the window in ui.py. Progress is
reported through three callbacks so each front end can show it its own way.
"""

import json
import time
from collections import deque

import cv2
import numpy as np
import requests

from . import app_cache, driver, geometry, map_shape, selector, strats_api, win
from .driver import DriverError, StratsWindow
from .minimap import MinimapReader
from .paths import CALIBRATION_FILE
from .riot_local import ATTACKERS_FIRST, DEFAULT_HALF_LENGTH, NotAvailable, RiotClient, sides_swapped

MATCH_POLL_SECONDS = 5
THOROUGH_CALIBRATION_SECONDS = 45
DATA_RETRY_SECONDS = 60
APP_LOAD_WAIT = 2.0         # seconds between looks at the app's cache after opening a map in it
APP_LOAD_ATTEMPTS = 4
STEADY_TOLERANCE = 2.0
SPIKE_GONE_READS = 3     # reads in a row without the planted indicator before the lineup is closed
LOCK_METRES = 25        # once this close to the chosen lineup's standing spot, keep it for the round
PREVIEW_SIZE = 360


ORIENTATION_READS = 6          # consecutive upside-down minimap reads before the side is flipped
ORIENTATION_TOLERANCE = 3.0
TELEPORT_METRES = 25           # further than anyone moves between two reads: only a round-start respawn does it
TELEPORT_SECONDS = 3
SPAWN_METRES = 20              # how close to a team's spawn label a respawn has to land
SPAWN_READS = 3                # reads in a row at that spawn afterwards (the buy phase keeps you there)
FOLLOWS_SIDE_KEY = "minimap_follows_side"


def _minimap_follows_side():
    """Whether this player's fixed minimap turns with their side, as learnt in an earlier match (None if unknown)."""
    try:
        return json.loads(CALIBRATION_FILE.read_text()).get(FOLLOWS_SIDE_KEY)
    except (OSError, ValueError):
        return None


def _remember_minimap_follows_side(follows):
    try:
        stored = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
        stored[FOLLOWS_SIDE_KEY] = follows
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(stored, indent=1))
    except (OSError, ValueError):
        pass


class SideTracker:
    """Which side the player is on, corrected from the screen and carried across side swaps."""

    def __init__(self):
        self.starts_attacking = None
        self.team = None
        self.rounds = 0
        self.half_length = DEFAULT_HALF_LENGTH
        self.angles = deque(maxlen=ORIENTATION_READS)
        self.rotating = False
        # A fixed minimap is either drawn from your side ("Based on Side") or always the same way
        # round; only the first says anything about the side. Learnt from a respawn.
        self.follows_side = _minimap_follows_side()
        self.last_position = None
        self.spawn_candidate = None

    def update_match(self, team, rounds, half_length):
        if self.starts_attacking is None:
            self.starts_attacking = team == ATTACKERS_FIRST
        elif self.team is not None and team != self.team:
            # The teams were swapped (custom games allow it at any time): the side swaps with them.
            self.starts_attacking = not self.starts_attacking
            self.angles.clear()
        self.team = team
        self.half_length = half_length
        if rounds != self.rounds:
            self.rounds = rounds
            self.angles.clear()

    def side(self):
        starts_attacking = True if self.starts_attacking is None else self.starts_attacking
        return "attack" if starts_attacking != sides_swapped(self.rounds, self.half_length) else "defense"

    def set_side(self, side):
        self.starts_attacking = (side == "attack") != sides_swapped(self.rounds, self.half_length)
        self.angles.clear()
        # Positions are about to be read on the other side's map art.
        self.last_position = self.spawn_candidate = None

    def observe(self, reading, assumed):
        """The side the minimap says the player is on, when that differs from `assumed`; else None.

        A fixed minimap set to follow your side is drawn with your own spawn at
        the bottom, like the map art for your side; if it keeps matching upside
        down, the art is for the wrong side. A minimap that is always drawn the
        same way, or a rotating one, says nothing about the side.
        """
        angle = reading.registration.angle
        if abs((angle + 45) % 90 - 45) > ORIENTATION_TOLERANCE:
            self.rotating = True
        if self.rotating or self.follows_side is False:
            self.angles.clear()
            return None
        self.angles.append(angle)
        if len(self.angles) == ORIENTATION_READS and self.upside_down():
            return "defense" if assumed == "attack" else "attack"
        return None

    def upside_down(self):
        """Whether every recent read of a fixed minimap matched the map art turned half way round."""
        return self.follows_side is not False and bool(self.angles) and all(
            abs((angle - 180 + 180) % 360 - 180) <= ORIENTATION_TOLERANCE for angle in self.angles
        )

    def observe_spawn(self, reading, now, map_item, assumed):
        """The side whose spawn the player has just been respawned in at a round start, else None.

        Only the jump into a spawn counts, never where the player stands later
        in the round: a move of TELEPORT_METRES between two reads can only be the
        respawn, and the buy phase then keeps the player at that spawn.
        """
        player = reading.player
        if player is None:
            return None
        previous, self.last_position = self.last_position, (player, now)
        spawns = {side: map_shape.spawn(map_item, assumed, of=side) for side in ("attack", "defense")}
        if self.spawn_candidate is not None:
            side, count = self.spawn_candidate
            if spawns[side] is None or geometry.metres(map_item, player, spawns[side]) > SPAWN_METRES:
                self.spawn_candidate = None
            elif count + 1 >= SPAWN_READS:
                self.spawn_candidate = None
                self._learn_minimap(reading, side, assumed)
                return side
            else:
                self.spawn_candidate = (side, count + 1)
                return None
        if previous is None or now - previous[1] > TELEPORT_SECONDS:
            return None
        if geometry.metres(map_item, previous[0], player) < TELEPORT_METRES:
            return None
        nearest = min((side for side in spawns if spawns[side] is not None),
                      key=lambda side: geometry.metres(map_item, player, spawns[side]), default=None)
        if nearest is not None and geometry.metres(map_item, player, spawns[nearest]) <= SPAWN_METRES:
            self.spawn_candidate = (nearest, 1)
        return None

    def _learn_minimap(self, reading, side, assumed):
        """Compare a known side with the minimap's orientation to see whether the minimap follows the side."""
        if self.rotating:
            return
        upside_down = abs((reading.registration.angle - 180 + 180) % 360 - 180) <= ORIENTATION_TOLERANCE
        follows = upside_down == (side != assumed)
        if follows != self.follows_side:
            self.follows_side = follows
            _remember_minimap_follows_side(follows)
        self.angles.clear()


THREAT_SECONDS = 10     # an enemy or a teammate's death this old says nothing about where enemies are now
SAME_PLACE = 3.0        # map percent within which two sightings are the same marker


class Threats:
    """Recently spotted enemies and recent teammate deaths, in map coordinates."""

    def __init__(self):
        self.enemies = []   # [position, last seen]
        self.deaths = []    # [position, first seen]; every death marker seen this round, so none counts twice

    def clear(self):
        self.enemies.clear()
        self.deaths.clear()

    @staticmethod
    def _near(entries, position):
        return next((entry for entry in entries if geometry.distance(entry[0], position) <= SAME_PLACE), None)

    def update(self, reading, now):
        """Record this read's markers; returns descriptions of the new ones."""
        news = []
        self.enemies = [entry for entry in self.enemies if now - entry[1] <= THREAT_SECONDS]
        for enemy in reading.enemies:
            known = self._near(self.enemies, enemy)
            if known is None:
                self.enemies.append([enemy, now])
                news.append(f"Enemy spotted at {enemy[0]:.0f},{enemy[1]:.0f}.")
            else:
                known[:] = [enemy, now]
        for death in reading.deaths:
            if self._near(self.deaths, death) is None:
                self.deaths.append([death, now])
                news.append(f"Teammate died at {death[0]:.0f},{death[1]:.0f}.")
        return news

    def active(self, now):
        return [position for position, seen in self.enemies + self.deaths if now - seen <= THREAT_SECONDS]


def frames(video, interval, stop):
    """Frames to analyse: the live primary screen, or a video file sampled every `interval` seconds."""
    if video is None:
        while not stop.is_set():
            started = time.time()
            yield win.capture_screen()
            stop.wait(max(0.0, interval - (time.time() - started)))
        return
    source = cv2.VideoCapture(video)
    position = 0.0
    while not stop.is_set():
        source.set(cv2.CAP_PROP_POS_MSEC, position * 1000)
        ok, frame = source.read()
        if not ok:
            return
        yield frame
        position += interval


class Watcher:
    def __init__(self, map_name=None, agent_name=None, side=None, drive=True, interval=1.0, video=None,
                 on_log=print, on_state=None, on_preview=None, fullscreen=False, hide=False):
        self.map_name, self.agent_name, self.side = map_name, agent_name, side
        self.drive, self.interval, self.video = drive, interval, video
        # Show the lineup video full size, and keep Strats.gg minimised while no lineup is open.
        self.fullscreen, self.hide = fullscreen, hide
        self.on_log = on_log
        self.on_state = on_state or (lambda **changes: None)
        self.on_preview = on_preview
        self.recalibrate = False

    def _preview(self, reader, frame, reading, silhouette_contours):
        roi = reader._roi(frame).copy()
        registration = reading.registration
        if registration is not None:
            for contour in silhouette_contours:
                points = np.array([registration.to_screen(point[0] / map_shape.VIEW_SIZE * 100) for point in contour])
                cv2.polylines(roi, [points.astype(np.int32)], True, (0, 0, 255), 1)
            for position, color in ((reading.spike, (0, 255, 255)), (reading.player, (255, 0, 255))):
                if position is not None:
                    x, y = registration.to_screen(position)
                    cv2.circle(roi, (int(x), int(y)), 14, color, 2)
            # Show the part of the region the map occupies rather than the whole corner of the screen.
            half = int(map_shape.VIEW_SIZE * registration.scale * 0.75)
            x, y = int(registration.center[0]), int(registration.center[1])
            roi = roi[max(0, y - half):y + half, max(0, x - half):x + half]
        scale = PREVIEW_SIZE / max(roi.shape[:2])
        self.on_preview(cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))

    def _park(self, window, maps, agents, map_item, agent_item, side, groups):
        """Leave Strats.gg on the lineup map, so only the lineup click is left when the spike goes down."""
        try:
            window.show_map(maps, agents, map_item, agent_item, side, groups)
            if self.hide:
                window.minimize()
        except DriverError as error:
            self.on_log(f"Strats.gg: {error}")

    def _load_through_app(self, window, maps, agents, map_item, agent_item, side):
        """Open this map, agent and side in Strats.gg, which loads its lineups, and read them from the app's cache."""
        self.on_log(f"No saved lineups for {map_item['name']} / {agent_item['name']} / {side}; loading them in Strats.gg.")
        try:
            window.show_map(maps, agents, map_item, agent_item, side, None)
        except DriverError as error:
            self.on_log(f"Strats.gg: {error}")
        for _ in range(APP_LOAD_ATTEMPTS):
            time.sleep(APP_LOAD_WAIT)
            app_cache.harvest()
            if strats_api.lineups_path(map_item["id"], agent_item["id"], side).exists():
                return strats_api.lineups(map_item["id"], agent_item["id"], side)
        raise strats_api.Unavailable("Strats.gg did not load them either")

    def run(self, stop):
        maps, agents = strats_api.maps(), strats_api.agents()
        manual = bool(self.map_name and self.agent_name)
        riot = None if manual else RiotClient()
        if self.drive:
            driver.launch(restart=True)
        window = StratsWindow()
        target = reader = lineups = groups = map_item = agent_item = contours = None
        threats = Threats()
        threat_round = None
        was_planted = False
        sides = SideTracker()
        last_match_check = last_thorough = 0.0
        last_spike = last_player = opened = pending = None
        locked = False
        opened_round = unplanted = 0
        waiting_reason = refused = None
        retry_at = 0.0
        self.on_log("Watching the minimap.")
        self.on_state(match="waiting for a match" if not manual else "", minimap="", spike="", player="", lineup="")

        for index, frame in enumerate(frames(self.video, self.interval, stop)):
            now = time.time() if self.video is None else index * self.interval
            if manual:
                wanted = (self.map_name, self.agent_name, self.side or sides.side())
            elif time.time() - last_match_check >= MATCH_POLL_SECONDS:
                last_match_check = time.time()
                try:
                    match = riot.current_match()
                except NotAvailable as reason:
                    if str(reason) != waiting_reason:
                        waiting_reason = str(reason)
                        self.on_log(f"Waiting: {reason}")
                        self.on_state(match=f"waiting: {reason}")
                    if target is None:
                        continue
                    wanted = (target[0], target[1], self.side or sides.side())
                else:
                    waiting_reason = None
                    found_map = next((item for item in maps if item["assetPath"].lower() == match["map_path"].lower()), None)
                    found_agent = next((item for item in agents if item["id"].lower() == match["agent_id"]), None)
                    if found_map is None or found_agent is None:
                        continue
                    sides.update_match(match["team"], match["rounds_played"], match["half_length"])
                    wanted = (found_map["name"], found_agent["name"], self.side or sides.side())
            elif target is None:
                continue
            else:
                wanted = (target[0], target[1], self.side or sides.side())

            if wanted != target:
                map_item = strats_api.find_by_name(maps, wanted[0])
                agent_item = strats_api.find_by_name(agents, wanted[1])
                if map_item is None or agent_item is None:
                    self.on_log(f"Unknown map or agent: {wanted[0]} / {wanted[1]}")
                    return
                if wanted == refused and time.time() < retry_at:
                    continue
                try:
                    try:
                        found = strats_api.lineups(map_item["id"], agent_item["id"], wanted[2])
                    except strats_api.Unavailable:
                        if not self.drive or wanted == refused:
                            raise
                        found = self._load_through_app(window, maps, agents, map_item, agent_item, wanted[2])
                    silhouette = map_shape.silhouette(map_item, wanted[2])
                except (strats_api.Unavailable, requests.RequestException) as error:
                    if wanted != refused:
                        self.on_log(f"No lineup data for {wanted[0]} / {wanted[1]} / {wanted[2]}: {error}")
                        self.on_state(match=f"{wanted[0]} / {wanted[1]}: no lineup data")
                    refused, retry_at = wanted, time.time() + DATA_RETRY_SECONDS
                    continue
                refused = None
                target = wanted
                lineups = [item for item in found if item["status"] == "approved"]
                groups = geometry.group_lineups(lineups)
                contours, _ = cv2.findContours(silhouette, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                reader = MinimapReader(silhouette, frame.shape[0], MinimapReader.stored_scale(frame.shape))
                opened = pending = last_spike = None
                locked = False
                description = f"{map_item['name']} / {agent_item['name']} / {target[2]}"
                self.on_log(f"Match: {description} ({len(lineups)} lineups)")
                self.on_state(match=description, lineup="")
                if self.drive:
                    # Set Strats.gg up now, so only the lineup click is left when the spike goes down.
                    self._park(window, maps, agents, map_item, agent_item, target[2], groups)

            if self.recalibrate:
                self.recalibrate = False
                reader.scale = None
                last_thorough = 0.0
            if reader.scale is None:
                thorough = time.time() - last_thorough >= THOROUGH_CALIBRATION_SECONDS
                self.on_state(minimap="calibrating (full search)..." if thorough else "looking for the minimap...")
                if thorough:
                    last_thorough = time.time()
                score = reader.calibrate(frame, thorough=thorough)
                if reader.scale is None:
                    self.on_state(minimap=f"not found (best match {score:.2f})")
                    continue
                self.on_log(f"Minimap zoom calibrated ({reader.scale}, match score {score:.2f}).")

            reading = reader.read(frame)
            if self.on_preview is not None:
                self._preview(reader, frame, reading, contours)
            # Leave the lineup once the spike is defused or explodes, or the next round starts, so
            # Strats.gg is back on the map and the next lineup is a single click away.
            if opened is not None:
                unplanted = 0 if reading.planted else unplanted + 1
                over = ("a new round started" if sides.rounds != opened_round
                        else "the spike is gone" if unplanted >= SPIKE_GONE_READS else None)
                if over:
                    opened = pending = last_spike = None
                    locked = False
                    self.on_log(f"Closing the lineup: {over}.")
                    self.on_state(lineup="")
                    if self.drive:
                        self._park(window, maps, agents, map_item, agent_item, target[2], groups)
            if reading.registration is None:
                self.on_state(minimap="not visible", spike="planted" if reading.planted else "not planted")
                continue
            if self.side is None:
                respawned = sides.observe_spawn(reading, now, map_item, target[2])
                if respawned is not None and respawned != target[2]:
                    sides.set_side(respawned)
                    last_player = last_spike = None
                    self.on_log(f"Side corrected to {respawned}: you respawned in the {respawned} spawn.")
                    continue
                correction = sides.observe(reading, target[2])
                if correction is not None:
                    sides.set_side(correction)
                    last_player = last_spike = None
                    self.on_log(f"Side corrected to {correction}: the minimap is drawn from that side.")
                    continue
                if sides.upside_down():
                    # Probably the wrong side's map art; wait for the check above before picking a lineup.
                    continue
            if reading.player is not None:
                last_player = reading.player
            # Markers from an earlier round mean nothing: start again when the round count moves or the spike goes.
            if sides.rounds != threat_round or (was_planted and not reading.planted):
                threats.clear()
                threat_round = sides.rounds
            was_planted = reading.planted
            for news in threats.update(reading, now):
                self.on_log(news)
            self.on_state(
                minimap=f"found (match {reading.registration.score:.2f})",
                player="not seen" if last_player is None else f"{last_player[0]:.0f}, {last_player[1]:.0f}",
                spike=("not planted" if not reading.planted else "planted, icon hidden" if reading.spike is None
                       else f"planted at {reading.spike[0]:.0f}, {reading.spike[1]:.0f}"),
            )
            if not reading.planted:
                last_spike = None
                continue
            if opened is not None and not locked and last_player is not None:
                away = geometry.metres(map_item, last_player, geometry.standing_spot(opened))
                if away <= LOCK_METRES:
                    locked = True
                    self.on_log(f"Locked in {opened['title']}: you are {away:.0f} m from where it is thrown.")
                    self.on_state(lineup=f"{opened['title']} (locked)")
            if locked:
                continue
            if reading.spike is None:
                continue
            # Act only on a spike position seen in two reads in a row.
            steady = last_spike is not None and geometry.distance(reading.spike, last_spike) <= STEADY_TOLERANCE
            last_spike = reading.spike
            if not steady:
                continue
            lineup = selector.choose(
                map_item, lineups, reading.spike, last_player, selector.post_plant_ability_ids(agent_item), threats.active(now)
            )
            if lineup is None:
                self.on_state(lineup="none lands on the spike")
                continue
            if opened is not None and lineup["id"] == opened["id"]:
                continue
            if lineup["id"] != pending:
                pending = lineup["id"]
                continue
            opened, opened_round, unplanted = lineup, sides.rounds, 0
            where = f"spike {reading.spike[0]:.0f},{reading.spike[1]:.0f}"
            if last_player is not None:
                where += f" / you {last_player[0]:.0f},{last_player[1]:.0f}"
            self.on_log(f"Lineup: {lineup['title']} (#{lineup['id']}) for {where}")
            self.on_state(lineup=lineup["title"])
            if not self.drive:
                continue
            try:
                window.show_map(maps, agents, map_item, agent_item, target[2], groups)
                if window.open_lineup(groups, selector.group_of(groups, lineup), lineup) and self.fullscreen:
                    window.fullscreen_video()
            except DriverError as error:
                self.on_log(f"Strats.gg: {error}")
