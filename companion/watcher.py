"""The watch loop: read the minimap, pick a lineup when the spike is planted, open it in Strats.gg.

Shared by the console `run` command and the window in ui.py. Progress is
reported through three callbacks so each front end can show it its own way.
"""

import time

import cv2
import numpy as np

from . import driver, geometry, map_shape, selector, strats_api, win
from .driver import DriverError, StratsWindow
from .minimap import MinimapReader
from .riot_local import NotAvailable, RiotClient

MATCH_POLL_SECONDS = 5
THOROUGH_CALIBRATION_SECONDS = 45
STEADY_TOLERANCE = 2.0
PREVIEW_SIZE = 360


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
                 on_log=print, on_state=None, on_preview=None):
        self.map_name, self.agent_name, self.side = map_name, agent_name, side
        self.drive, self.interval, self.video = drive, interval, video
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

    def run(self, stop):
        maps, agents = strats_api.maps(), strats_api.agents()
        manual = bool(self.map_name and self.agent_name)
        riot = None if manual else RiotClient()
        if self.drive:
            driver.launch(restart=True)
        window = StratsWindow()
        target = reader = lineups = groups = map_item = agent_item = contours = None
        last_match_check = last_thorough = 0.0
        last_spike = last_player = opened = pending = None
        waiting_reason = None
        self.on_log("Watching the minimap.")
        self.on_state(match="waiting for a match" if not manual else "", minimap="", spike="", player="", lineup="")

        for frame in frames(self.video, self.interval, stop):
            if manual:
                wanted = (self.map_name, self.agent_name, self.side or "attack")
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
                    wanted = target
                else:
                    waiting_reason = None
                    found_map = next((item for item in maps if item["assetPath"].lower() == match["map_path"].lower()), None)
                    found_agent = next((item for item in agents if item["id"].lower() == match["agent_id"]), None)
                    if found_map is None or found_agent is None:
                        continue
                    wanted = (found_map["name"], found_agent["name"], self.side or match["side"])
            elif target is None:
                continue
            else:
                wanted = target

            if wanted != target:
                map_item = strats_api.find_by_name(maps, wanted[0])
                agent_item = strats_api.find_by_name(agents, wanted[1])
                if map_item is None or agent_item is None:
                    self.on_log(f"Unknown map or agent: {wanted[0]} / {wanted[1]}")
                    return
                target = wanted
                lineups = [item for item in strats_api.lineups(map_item["id"], agent_item["id"], target[2]) if item["status"] == "approved"]
                groups = geometry.group_lineups(lineups)
                silhouette = map_shape.silhouette(map_item, target[2])
                contours, _ = cv2.findContours(silhouette, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                reader = MinimapReader(silhouette, frame.shape[0], MinimapReader.stored_scale(frame.shape))
                opened = pending = last_spike = None
                description = f"{map_item['name']} / {agent_item['name']} / {target[2]}"
                self.on_log(f"Match: {description} ({len(lineups)} lineups)")
                self.on_state(match=description, lineup="")

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
            if reading.registration is None:
                self.on_state(minimap="not visible", spike="planted" if reading.planted else "not planted")
                continue
            if reading.player is not None:
                last_player = reading.player
            self.on_state(
                minimap=f"found (match {reading.registration.score:.2f})",
                player="not seen" if last_player is None else f"{last_player[0]:.0f}, {last_player[1]:.0f}",
                spike=("not planted" if not reading.planted else "planted, icon hidden" if reading.spike is None
                       else f"planted at {reading.spike[0]:.0f}, {reading.spike[1]:.0f}"),
            )
            if not reading.planted:
                last_spike = None
                if opened is not None:
                    opened = pending = None
                    self.on_log("Spike no longer planted.")
                    self.on_state(lineup="")
                continue
            if reading.spike is None:
                continue
            # Act only on a spike position seen in two reads in a row.
            steady = last_spike is not None and geometry.distance(reading.spike, last_spike) <= STEADY_TOLERANCE
            last_spike = reading.spike
            if not steady:
                continue
            lineup = selector.choose(lineups, reading.spike, last_player, selector.post_plant_ability_ids(agent_item))
            if lineup is None:
                self.on_state(lineup="none lands on the spike")
                continue
            if lineup["id"] == opened:
                continue
            if lineup["id"] != pending:
                pending = lineup["id"]
                continue
            opened = lineup["id"]
            where = f"spike {reading.spike[0]:.0f},{reading.spike[1]:.0f}"
            if last_player is not None:
                where += f" / you {last_player[0]:.0f},{last_player[1]:.0f}"
            self.on_log(f"Lineup: {lineup['title']} (#{lineup['id']}) for {where}")
            self.on_state(lineup=lineup["title"])
            if not self.drive:
                continue
            try:
                window.show_map(maps, agents, map_item, agent_item, target[2], groups)
                window.open_lineup(groups, selector.group_of(groups, lineup), lineup)
            except DriverError as error:
                self.on_log(f"Strats.gg: {error}")
