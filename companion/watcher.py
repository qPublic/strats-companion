"""The watch loop: read the minimap, pick a lineup when the spike is planted, open it in Strats.gg.

Shared by the console `run` command and the window in ui.py. Progress is
reported through three callbacks so each front end can show it its own way.
"""

import json
import queue
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import requests

from . import app_cache, broken, driver, geometry, map_shape, selector, spot, strats_api, win
from .aim import AimGuide, Tracker
from .driver import LINEUP_PAGE, MAP_VIEW, OTHER, WINDOW_TITLE, DriverError, StratsWindow
from .minimap import ICON_RADIUS, MIN_CALIBRATION_SCORE, MinimapReader
from .paths import CALIBRATION_FILE
from .riot_local import ATTACKERS_FIRST, DEFAULT_HALF_LENGTH, NotAvailable, RiotClient, sides_swapped

MATCH_POLL_SECONDS = 5
THOROUGH_CALIBRATION_SECONDS = 45
DATA_RETRY_SECONDS = 60
APP_LOAD_WAIT = 2.0         # seconds between looks at the app's cache after opening a map in it
APP_LOAD_ATTEMPTS = 4
STEADY_TOLERANCE = 2.0
SPIKE_GONE_READS = 3     # reads in a row without the planted indicator before the lineup is closed
SAME_PLANT_METRES = 3    # spikes this close together count as the same plant
LOCK_METRES = 25        # once this close to the chosen lineup's standing spot, keep it for the round
PREVIEW_SIZE = 360
PLANTED_INTERVAL = 0.3   # seconds between reads once the spike is down and no lineup is open yet
IN_POSITION_METRES = 2.0  # this close to the standing spot, the aim guide starts matching the screen
# Test mode: the minimap draws a spike badge beside the carrier's icon; a spike this many icon radii
# from the player's icon is that badge (or was only just dropped), not a spike lying on the ground.
CARRIED_ICON_RADII = 3.0
AIM_IMAGE_RETRY = 5.0
HEALTH_SECONDS = 10         # how often everything is checked and put right
STALL_SECONDS = 4.0         # a tracker that has drawn nothing for this long is restarted
LOST_MINIMAP_SECONDS = 20   # minimap not found for this long: look for it again from scratch
SPOT_ATTEMPTS = 12          # tries at reading a lineup's standing spot from its screenshot...
SPOT_RETRY_SECONDS = 3.0    # ...this far apart, while Strats.gg loads it


ORIENTATION_READS = 6          # consecutive upside-down minimap reads before the side is flipped
ORIENTATION_TOLERANCE = 3.0
TELEPORT_METRES = 25           # further than anyone moves between two reads: only a round-start respawn does it
TELEPORT_SECONDS = 3           # without the round count, a jump only counts between reads this close together
RESPAWN_WINDOW_SECONDS = 60    # after the round count goes up, look this long for the respawn, however far apart the reads
SPAWN_METRES = 20              # how close to a team's spawn label a respawn has to land
SPAWN_READS = 3                # reads at that spawn, the jump included (the buy phase keeps you there)
# (An earlier key, "minimap_follows_side", could be learnt wrongly on attack rounds and is no longer read.)
FOLLOWS_SIDE_KEY = "minimap_turns_with_side"


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
        self.respawn_due = None      # (round, time the round count went up) while a respawn is expected
        self.respawn_news = []

    def update_match(self, team, rounds, half_length):
        seen_before = self.team is not None
        if self.starts_attacking is None:
            self.starts_attacking = team == ATTACKERS_FIRST
        elif self.team is not None and team != self.team:
            # The teams were swapped (custom games allow it at any time): the side swaps with them.
            self.starts_attacking = not self.starts_attacking
            self.angles.clear()
        self.team = team
        self.half_length = half_length
        if rounds != self.rounds:
            if seen_before and rounds > self.rounds:
                self.respawn_due = (rounds + 1, time.time())
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
        same way, or a rotating one, says nothing about the side. Until a defence
        respawn has shown which kind a fixed one is, it is taken to follow the
        side, which is the game's default.
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
        respawn, and the buy phase then keeps the player at that spawn. With the
        round count known, the jump may span any gap between reads in the minute
        after the count goes up (the reads pause while Strats.gg is being driven);
        without it, only reads TELEPORT_SECONDS apart count.
        """
        due = self.respawn_due
        if due is not None and time.time() - due[1] > RESPAWN_WINDOW_SECONDS:
            self.respawn_due = None
            self.respawn_news.append(f"Round {due[0]}: no respawn seen on the minimap.")
        player = reading.player
        if player is None:
            return None
        previous, self.last_position = self.last_position, (player, now)
        spawns = {side: map_shape.spawn(map_item, assumed, of=side) for side in ("attack", "defense")}

        def at_spawn(side):
            return spawns[side] is not None and geometry.metres(map_item, player, spawns[side]) <= SPAWN_METRES

        if self.spawn_candidate is not None:
            side, count = self.spawn_candidate
            if not at_spawn(side):
                self.spawn_candidate = None
            elif count + 1 >= SPAWN_READS:
                self.spawn_candidate = None
                if self.respawn_due is not None:
                    self.respawn_news.append(f"Round {self.respawn_due[0]}: you respawned in the {side} spawn.")
                    self.respawn_due = None
                self._learn_minimap(reading, side, assumed)
                return side
            else:
                self.spawn_candidate = (side, count + 1)
                return None
        if previous is None:
            return None
        if self.respawn_due is None and now - previous[1] > TELEPORT_SECONDS:
            return None
        if geometry.metres(map_item, previous[0], player) < TELEPORT_METRES:
            return None
        landed = [side for side in spawns if at_spawn(side)]
        if landed:
            self.spawn_candidate = (min(landed, key=lambda side: geometry.metres(map_item, player, spawns[side])), 1)
        return None

    def _learn_minimap(self, reading, side, assumed):
        """Compare a known side with the minimap's orientation to see whether the minimap follows the side.

        Only a defence respawn tells the two kinds apart: on attack both are
        drawn with the attackers' spawn at the bottom.
        """
        self.angles.clear()
        if self.rotating or side != "defense":
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
    """Frames to analyse: the live primary screen, or a video file sampled every `interval` seconds.

    `interval` may be a function, asked again before every frame.
    """
    if video is None:
        while not stop.is_set():
            started = time.time()
            yield win.capture_screen()
            wait = interval() if callable(interval) else interval
            stop.wait(max(0.0, wait - (time.time() - started)))
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
                 on_log=print, on_state=None, on_preview=None, fullscreen=False, hide=False, on_guide=None,
                 test_mode=False):
        self.map_name, self.agent_name, self.side = map_name, agent_name, side
        self.drive, self.interval, self.video = drive, interval, video
        # Show the lineup video full size, and keep Strats.gg minimised while no lineup is open.
        self.fullscreen, self.hide = fullscreen, hide
        # Draws where to stand and aim over the game (overlay.py); None turns the guide off.
        self.on_guide = on_guide
        # Practice without planting: a spike dropped on the ground is treated as planted there.
        self.test_mode = test_mode
        # Set from the window: "broken" (never pick the open lineup again), "next" (try the next one)
        # or "picture" (the open lineup's Strats.gg picture is of something else).
        self.request = None
        self.aim_guides = {}        # lineup id -> (AimGuide, picture) or (None, time of the last try)
        self.guide_shown = False
        self.guide = None           # the latest minimap part of the guide, which the aim tracker adds to
        self.last_aim = None
        self.tracker = None
        self.true_spots = {}        # lineup id -> standing spot read from its aim screenshot (or None)
        self.wrong_pictures = broken.picture_ids()
        self.spot_reads = set()     # lineups whose screenshot is being read for the spot right now
        self.reader = None
        self.stop_event = None
        self.hurry = False
        self.strats_jobs = queue.Queue()
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

    def _aim_guide(self, lineup):
        """The aim matcher and screenshot for a lineup, or (None, None) while the screenshot is not to be had."""
        if lineup["id"] in self.wrong_pictures:
            return None, None
        known = self.aim_guides.get(lineup["id"])
        if known is not None and known[0] is not None:
            return known
        if known is not None and time.time() - known[1] < AIM_IMAGE_RETRY:
            return None, None
        matcher, preview = strats_api.aim_data(lineup["id"])
        if matcher is None:
            self.aim_guides[lineup["id"]] = (None, time.time())
            return None, None
        self.aim_guides[lineup["id"]] = (matcher, preview)
        return self.aim_guides[lineup["id"]]

    def _preload_aims(self, lineups, agent_item):
        """Load the stored aim data of every lineup this match could use, so none is fetched at plant time."""
        wanted = selector.post_plant_ability_ids(agent_item)
        loaded = 0
        for lineup in lineups:
            if (not wanted or lineup["abilityId"] in wanted) and strats_api.aim_stored(lineup["id"]):
                matcher, preview = strats_api.aim_data(lineup["id"])
                if matcher is not None:
                    self.aim_guides[lineup["id"]] = (matcher, preview)
                    loaded += 1
        return loaded

    def _standing_spot(self, lineup, map_item):
        """The lineup's standing spot: as read from its aim screenshot once that is done, else Strats.gg's dot."""
        if lineup["id"] not in self.true_spots:
            self.true_spots[lineup["id"]] = None
            self.spot_reads.add(lineup["id"])

            def read():
                found = None
                # The screenshot may only arrive once Strats.gg has opened the lineup: keep trying a while.
                for _ in range(SPOT_ATTEMPTS):
                    try:
                        found = spot.true_spot(lineup, map_item)
                    except Exception:  # noqa: BLE001 - the dot still works
                        found = None
                    if found is not None or spot.known(lineup["id"]) or self.stop_event.is_set():
                        break
                    self.stop_event.wait(SPOT_RETRY_SECONDS)
                self.true_spots[lineup["id"]] = found
                self.spot_reads.discard(lineup["id"])
                if found is not None:
                    moved = geometry.metres(map_item, found, geometry.standing_spot(lineup))
                    self.on_log(f"Standing spot for {lineup['title']} taken from its screenshot "
                                f"({moved:.1f} m from the Strats.gg dot).")
            threading.Thread(target=read, daemon=True).start()
        if lineup["id"] in self.wrong_pictures:
            return geometry.standing_spot(lineup)       # a picture of somewhere else says nothing of the spot
        return self.true_spots[lineup["id"]] or geometry.standing_spot(lineup)

    def _show_guide(self, frame, reading, lineup, map_item, player):
        """Tell the overlay where the standing spot is on the minimap and, once there, where to aim."""
        registration = reading.registration
        spot = self._standing_spot(lineup, map_item)
        guide = {"title": lineup["title"], "metres": None, "in_position": False}
        if registration is not None:
            guide["stand"] = tuple(float(v) for v in registration.to_screen(spot))
            landing = geometry.landing_point(lineup)
            if landing is not None:
                guide["land"] = tuple(float(v) for v in registration.to_screen(landing))
        if player is not None:
            guide["metres"] = geometry.metres(map_item, player, spot)
            guide["in_position"] = guide["metres"] <= IN_POSITION_METRES
        # The aim picture shows the whole time the lineup is open; the reticle only once on the spot.
        matcher, guide["picture"] = self._aim_guide(lineup)
        if not guide["in_position"]:
            matcher = None
        if matcher is None:
            self._stop_tracker()
        elif self.video is not None:
            self.last_aim = matcher.locate(frame)
        elif self.tracker is None or self.tracker.lineup_id != lineup["id"]:
            self._stop_tracker()
            self.tracker = Tracker(matcher, self._publish_aim, also_stop=self.stop_event)
            self.tracker.lineup_id = lineup["id"]
            self.tracker.start()
        guide["aim"] = self.last_aim
        self.guide = guide
        self.on_guide(guide)
        self.guide_shown = True

    def _publish_aim(self, aim):
        """From the aim tracker's thread: redraw with the newest aim point."""
        self.last_aim = aim
        if self.guide is not None:
            self.on_guide({**self.guide, "aim": aim})

    def _stop_tracker(self):
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
        self.last_aim = None

    def _hide_guide(self):
        self._stop_tracker()
        self.guide = None
        if self.guide_shown:
            self.on_guide(None)
            self.guide_shown = False

    def _health(self, window, maps, agents, map_item, agent_item, side, groups, opened):
        """Every HEALTH_SECONDS: check that Strats.gg and the in-game guide are as they should be, and fix them."""
        if self.on_guide is not None:
            if opened is None:
                if self.guide_shown or self.tracker is not None:
                    self._hide_guide()
                    self.on_log("Check: hid a guide that was left up with no lineup open.")
            else:
                if self.tracker is not None and self.tracker.stalled(STALL_SECONDS):
                    self._stop_tracker()
                    self.on_log("Check: the aim reticle had stopped; restarting it.")
                known = self.aim_guides.get(opened["id"])
                if known is not None and known[0] is None:
                    self.aim_guides.pop(opened["id"])          # try the aim picture again on the next read
                if (self.true_spots.get(opened["id"], 0) is None and opened["id"] not in self.spot_reads
                        and not spot.known(opened["id"])):
                    self.true_spots.pop(opened["id"])          # and the standing spot from its screenshot
        if self.drive:
            self._drive(lambda: self._check_strats(window, maps, agents, map_item, agent_item, side, groups, opened))

    def _check_strats(self, window, maps, agents, map_item, agent_item, side, groups, opened):
        """On the driving thread: Strats.gg open, and showing the open lineup, or the map (or minimised) otherwise."""
        found = win.find_windows(WINDOW_TITLE)
        if not found:
            self.on_log("Check: Strats.gg was closed; starting it again.")
            driver.launch(restart=True)     # running without a window, it does not open one when asked again
            window.shown = None
            self._park(window, maps, agents, map_item, agent_item, side, groups)
            return
        minimised = bool(win.user32.IsIconic(found[0][0]))
        if opened is None:
            if minimised:
                return                                  # set aside between lineups, as asked
            window.attach()
            state = window.state(window.capture())
            if self.hide:
                window.leave_fullscreen()
                window.minimize()
                self.on_log("Check: minimised Strats.gg, which was left open between lineups.")
            elif state != MAP_VIEW:
                window.show_map(maps, agents, map_item, agent_item, side, groups)
                self.on_log("Check: put Strats.gg back on the lineup map.")
            return
        window.attach()
        state = window.state(window.capture())
        showing = state == LINEUP_PAGE or (state == OTHER and window.fullscreen)
        if not showing:
            self.on_log(f"Check: Strats.gg was not showing {opened['title']}; opening it again.")
            window.show_map(maps, agents, map_item, agent_item, side, groups)
            if window.open_lineup(groups, selector.group_of(groups, opened), opened) and self.fullscreen:
                window.fullscreen_video()
        elif state == LINEUP_PAGE and self.fullscreen:
            window.fullscreen_video()
            self.on_log("Check: put the lineup video back to full screen.")

    def _drive(self, job):
        """Run a Strats.gg job on the driving thread, after any already waiting."""
        self.strats_jobs.put(job)

    def _driving(self, stop):
        while not stop.is_set():
            try:
                job = self.strats_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                job()
            except DriverError as error:
                self.on_log(f"Strats.gg: {error}")
            finally:
                self.strats_jobs.task_done()

    def _park(self, window, maps, agents, map_item, agent_item, side, groups):
        """Leave Strats.gg on the lineup map, so only the lineup click is left when the spike goes down."""
        def park():
            window.show_map(maps, agents, map_item, agent_item, side, groups)
            if self.hide:
                window.minimize()
        self._drive(park)

    def _open(self, window, maps, agents, map_item, agent_item, side, groups, lineup):
        def open_lineup():
            window.show_map(maps, agents, map_item, agent_item, side, groups)
            if window.open_lineup(groups, selector.group_of(groups, lineup), lineup) and self.fullscreen:
                window.fullscreen_video()
        self._drive(open_lineup)

    def _load_through_app(self, window, maps, agents, map_item, agent_item, side):
        """Open this map, agent and side in Strats.gg, which loads its lineups, and read them from the app's cache."""
        self.on_log(f"No saved lineups for {map_item['name']} / {agent_item['name']} / {side}; loading them in Strats.gg.")
        self.strats_jobs.join()          # the driving thread must be idle before Strats.gg is used here
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
        opened_spike = None
        skipped = set()        # lineups passed over with "Next lineup" for the current plant
        immediate = False      # open the next pick at once, without waiting for it to repeat
        no_lineup_note = None
        used_lineups = []      # (spike, lineup id) for each lineup used this match
        waiting_reason = refused = None
        retry_at = 0.0
        self.on_log("Watching the minimap.")
        self.on_state(match="waiting for a match" if not manual else "", minimap="", spike="", player="", lineup="")

        self.stop_event = stop
        if self.drive:
            threading.Thread(target=self._driving, args=(stop,), daemon=True).start()
        pace = self.interval if self.video is not None else (lambda: PLANTED_INTERVAL if self.hurry else self.interval)
        last_health = time.time()
        minimap_seen = time.time()
        for index, frame in enumerate(frames(self.video, pace, stop)):
            now = time.time() if self.video is None else index * self.interval
            if self.video is None and target is not None and time.time() - last_health >= HEALTH_SECONDS:
                last_health = time.time()
                self._health(window, maps, agents, map_item, agent_item, target[2], groups, opened)
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
                self.reader = reader
                opened = pending = last_spike = None
                locked = False
                used_lineups = []
                description = f"{map_item['name']} / {agent_item['name']} / {target[2]}"
                self.on_log(f"Match: {description} ({len(lineups)} lineups)")
                if self.on_guide is not None:
                    ready = self._preload_aims(lineups, agent_item)
                    if ready:
                        self.on_log(f"Aim guide ready for {ready} of them.")
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
            if self.test_mode and not reading.planted and reading.registration is not None:
                dropped = reader.find_dropped_spike(frame, reading.registration)
                carrier = reading.player or last_player
                if dropped is not None and carrier is not None:
                    gap = np.subtract(reading.registration.to_screen(dropped), reading.registration.to_screen(carrier))
                    if np.hypot(*gap) <= CARRIED_ICON_RADII * ICON_RADIUS * reading.registration.scale:
                        dropped = None
                if dropped is not None:
                    reading.planted, reading.spike = True, dropped
            if self.on_preview is not None:
                self._preview(reader, frame, reading, contours)
            # Leave the lineup once the spike is defused or explodes, or the next round starts, so
            # Strats.gg is back on the map and the next lineup is a single click away.
            if opened is not None:
                unplanted = 0 if reading.planted else unplanted + 1
                moved = (self.test_mode and reading.spike is not None
                         and geometry.metres(map_item, reading.spike, opened_spike) > SAME_PLANT_METRES)
                over = ("a new round started" if sides.rounds != opened_round
                        else "the spike is gone" if unplanted >= SPIKE_GONE_READS
                        else "the spike was moved" if moved else None)
                if over:
                    if self.on_guide is not None:
                        self._hide_guide()
                    used_lineups.append((opened_spike, opened["id"]))
                    skipped.clear()
                    opened = pending = last_spike = None
                    locked = False
                    self.on_log(f"Closing the lineup: {over}.")
                    self.on_state(lineup="")
                    if self.drive:
                        self._park(window, maps, agents, map_item, agent_item, target[2], groups)
            if reading.registration is None:
                self.on_state(minimap="not visible", spike="planted" if reading.planted else "not planted")
                if self.video is None and time.time() - minimap_seen > LOST_MINIMAP_SECONDS:
                    minimap_seen = time.time()
                    if reader.calibrate(frame, thorough=False) >= MIN_CALIBRATION_SCORE:
                        self.on_log(f"Check: found the minimap again (zoom {reader.scale}).")
                continue
            minimap_seen = time.time()
            if self.side is None:
                respawned = sides.observe_spawn(reading, now, map_item, target[2])
                for news in sides.respawn_news:
                    self.on_log(news)
                sides.respawn_news.clear()
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
            if self.on_guide is not None:
                if opened is not None:
                    self._show_guide(frame, reading, opened, map_item, reading.player or last_player)
                else:
                    self._hide_guide()
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
            self.hurry = reading.planted and opened is None
            if not reading.planted:
                last_spike = None
                continue
            if self.request is not None:
                request, self.request = self.request, None
                if opened is None:
                    self.on_log("No lineup is open.")
                elif request == "picture":
                    broken.mark_picture(opened)
                    self.wrong_pictures.add(opened["id"])
                    self._stop_tracker()
                    self.on_log(f"Marked the picture of {opened['title']} as wrong; it and its reticle will not be shown.")
                else:
                    if request == "broken":
                        broken.mark(opened)
                        self.on_log(f"Marked {opened['title']} as broken; it will not be picked again.")
                    else:
                        skipped.add(opened["id"])
                    if self.on_guide is not None:
                        self._hide_guide()
                    opened = pending = None
                    locked = False
                    immediate = True
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
            # Same plant as an earlier round: count how often each lineup was used there.
            used = Counter(
                lineup_id for spike, lineup_id in used_lineups
                if geometry.metres(map_item, spike, reading.spike) <= SAME_PLANT_METRES
            )
            choose = lambda exclude: selector.choose(  # noqa: E731
                map_item, lineups, reading.spike, last_player, selector.post_plant_ability_ids(agent_item),
                threats.active(now), used, exclude,
            )
            lineup = choose(broken.ids() | skipped)
            if lineup is None and skipped:
                skipped.clear()
                lineup = choose(broken.ids())
                if lineup is not None:
                    self.on_log("That was the last lineup for this plant; back to the first.")
            if lineup is None:
                mollies = selector.post_plant_ability_ids(agent_item)
                nearest = min((geometry.metres(map_item, geometry.landing_point(item), reading.spike) for item in lineups
                               if geometry.landing_point(item) and (not mollies or item["abilityId"] in mollies)),
                              default=None)
                note = "none lands on the spike" if nearest is None else f"none lands on the spike (nearest {nearest:.0f} m away)"
                if note != no_lineup_note:
                    no_lineup_note = note
                    self.on_log("No lineup lands on this plant" + ("." if nearest is None else
                                f"; the nearest lands {nearest:.0f} m away."))
                self.on_state(lineup=note)
                continue
            no_lineup_note = None
            if opened is not None and lineup["id"] == opened["id"]:
                continue
            if lineup["id"] != pending and not immediate:
                pending = lineup["id"]
                continue
            immediate = False
            opened, opened_round, unplanted, opened_spike = lineup, sides.rounds, 0, reading.spike
            if used:
                usual = selector.choose(
                    map_item, lineups, reading.spike, last_player, selector.post_plant_ability_ids(agent_item),
                    threats.active(now), exclude=broken.ids() | skipped,
                )
                if usual is not None and usual["id"] != lineup["id"] and used[usual["id"]] >= selector.REPEAT_LIMIT:
                    self.on_log(f"Not using {usual['title']} again: used {used[usual['id']]} times for this plant.")
            where = f"spike {reading.spike[0]:.0f},{reading.spike[1]:.0f}"
            if last_player is not None:
                where += f" / you {last_player[0]:.0f},{last_player[1]:.0f}"
            self.on_log(f"Lineup: {lineup['title']} (#{lineup['id']}) for {where}")
            self.on_state(lineup=lineup["title"])
            self.hurry = False
            if self.on_guide is not None:
                # The in-game guide goes up first; Strats.gg follows on its own thread.
                self._show_guide(frame, reading, opened, map_item, reading.player or last_player)
            if self.drive:
                self._open(window, maps, agents, map_item, agent_item, target[2], groups, lineup)
