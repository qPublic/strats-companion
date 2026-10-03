"""A marker on the ground at the lineup's standing spot, drawn where it is in the game world.

Placing a point of the world on screen needs where the camera is and how it is
turned. All of it comes from the screen:

- position: the player's icon on the minimap;
- heading: the pointer on that icon, which shows the way the player faces;
- pitch (looking up or down): vertical edges of walls and doors lean towards a
  common point that moves with the pitch;
- between those readings, the shift of the picture from one frame to the next,
  which follows every small turn of the mouse.

The absolute readings are noisy and the frame-to-frame shifts drift, so the
heading and pitch follow the shifts and are pulled gently towards the absolute
readings (a complementary filter). The ground is taken to be level with the
player's feet and the eye EYE_HEIGHT above it.
"""

import threading
import time

import cv2
import numpy as np

from . import geometry
from .aim import HORIZONTAL_FOV, WEAPON, _OneEuro, mss_grabber, scenery_region
from .minimap import ROI_FRACTION

EYE_HEIGHT = 1.55               # metres from the floor to the camera
RING_RADIUS = 0.55              # metres
PIN_HEIGHT = 2.2                # metres: a post standing in the ring, visible over low cover
RING_POINTS = 36
PERIOD = 1 / 40
WORK_WIDTH = 960                # the scene is read at this width
SHIFT_REGION = (0.30, 0.16, 0.70, 0.52)   # middle of the screen, above the weapon: used for frame-to-frame shifts
HEADING_PULL = 0.2              # how strongly each minimap reading corrects the heading
PITCH_PULL = 0.15               # how strongly each vertical-edge reading corrects the pitch
PITCH_PERIOD = 0.25             # seconds between vertical-edge readings (a separate thread)
SHIFT_SCALE = 0.5               # the middle of the screen is shrunk this much for frame-to-frame shifts
MIN_EDGES = 6
NEAREST = 0.4                   # metres in front of the camera; anything nearer is not drawn


def _focal(width):
    return (width / 2) / np.tan(np.radians(HORIZONTAL_FOV / 2))


def estimate_pitch(scene, origin, scale, screen):
    """Degrees the camera looks up (negative: down) from vertical edges, or None with too few of them.

    `scene` is a grey picture of the screen part at `origin`, shrunk by `scale`.
    Vertical world lines all point at one vanishing point straight above or
    below the middle of the screen, f / tan(pitch) away from it.
    """
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(scene)[0]
    if found is None:
        return None
    width, height = screen
    focal, cx, cy = _focal(width), width / 2, height / 2
    values, weights = [], []
    for x1, y1, x2, y2 in found.reshape(-1, 4) / scale + np.array([origin[0], origin[1]] * 2):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 40 or abs(dx) > abs(dy) * 0.36:          # within 20 degrees of upright
            continue
        mx, my = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        if WEAPON[0] < mx and WEAPON[1] < my:
            continue
        side = (x1 + x2) / 2 - cx
        if abs(side) < width * 0.09:                         # lines near the middle say little
            continue
        denominator = dx * (cy - y1) - dy * (cx - x1)
        if abs(denominator) < 1e-6:
            continue
        tangent = dx * focal / denominator
        if abs(tangent) < 1.5:
            values.append(tangent)
            weights.append(length * abs(side))
    if len(values) < MIN_EDGES:
        return None
    order = np.argsort(values)
    values, weights = np.array(values)[order], np.array(weights)[order]
    middle = values[min(len(values) - 1, int(np.searchsorted(np.cumsum(weights), weights.sum() / 2)))]
    return float(np.degrees(np.arctan(middle)))


def project(points, heading, pitch, screen):
    """Screen positions of world points given relative to the player.

    `points` are (east, south, up) in metres on the map art's axes, from the
    player's feet; heading is the map-art direction faced, in degrees (0 east,
    90 south). Points behind or too close to the camera come back as None.
    """
    width, height = screen
    focal = _focal(width)
    h, p = np.radians(heading), np.radians(pitch)
    forward, right = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    out = []
    for east, south, up in points:
        flat = np.array([east, south])
        x, z_level, y_level = flat @ right, flat @ forward, EYE_HEIGHT - up
        y = y_level * np.cos(p) + z_level * np.sin(p)
        z = z_level * np.cos(p) - y_level * np.sin(p)
        out.append(None if z < NEAREST else (width / 2 + focal * x / z, height / 2 + focal * y / z))
    return out


class GroundTracker:
    """Keeps the standing-spot marker in place in the world while the player walks to it."""

    def __init__(self, reader, map_item, spot, publish, also_stop=None, grabber=mss_grabber):
        self.reader, self.map_item, self.spot, self.publish = reader, map_item, spot, publish
        self.grabber = grabber
        self.registration = None          # kept up to date by the watcher's minimap reads
        self.stopped = threading.Event()
        self.also_stop = also_stop or threading.Event()
        self.heading = self.pitch = None
        self.position = _OneEuro(min_cutoff=1.0, beta=0.05)
        self.icon = None
        self.measured_pitch = None      # (degrees, when) from the vertical-edge thread

    def start(self):
        for loop in (self._loop, self._pitch_loop):
            threading.Thread(target=self._guarded, args=(loop,), daemon=True).start()

    def _guarded(self, loop):
        """Run a loop; if it fails (a capture error, say), start it again after a pause rather than dying."""
        while not self._done():
            try:
                loop()
            except Exception:  # noqa: BLE001 - the guide must not take the companion down
                self.stopped.wait(1.0)

    def stop(self):
        self.stopped.set()

    def _done(self):
        return self.stopped.is_set() or self.also_stop.is_set()

    def _loop(self):
        grab, screen = self.grabber()
        width, height = screen
        side = int(height * ROI_FRACTION)
        left, top, right, bottom = SHIFT_REGION
        shift_area = (int(left * width), int(top * height), int((right - left) * width), int((bottom - top) * height))
        focal = _focal(width) * SHIFT_SCALE
        previous = window = None
        used_pitch = 0.0
        while not self._done():
            started = time.perf_counter()
            registration = self.registration
            if registration is None:
                self.stopped.wait(PERIOD)
                continue
            # The player's icon: only the small area around where it was, unless it was lost.
            place = facing = icon = None
            if self.icon is not None:
                reach = self.reader.icon_reach(registration) + 4
                box_left, box_top = max(0, int(self.icon[0]) - reach), max(0, int(self.icon[1]) - reach)
                box = grab(box_left, box_top, min(2 * reach, side - box_left), min(2 * reach, side - box_top))
                place, facing, icon = self.reader.player_and_facing(box, registration, self.icon, (box_left, box_top))
            if place is None:
                place, facing, icon = self.reader.player_and_facing(grab(0, 0, side, side), registration)
            self.icon = icon

            # Frame-to-frame turn of the camera, from how far the middle of the picture shifted.
            middle = cv2.cvtColor(cv2.resize(grab(*shift_area), None, fx=SHIFT_SCALE, fy=SHIFT_SCALE,
                                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if window is None or window.shape != middle.shape:
                window = cv2.createHanningWindow(middle.shape[::-1], cv2.CV_32F)
            if previous is not None and self.heading is not None:
                (dx, dy), response = cv2.phaseCorrelate(previous, middle, window)
                if response > 0.1:
                    self.heading += np.degrees(np.arctan(-dx / focal))
                    self.pitch += np.degrees(np.arctan(dy / focal))
            previous = middle

            if facing is not None:
                if self.heading is None:
                    self.heading = facing
                else:
                    self.heading += HEADING_PULL * ((facing - self.heading + 180) % 360 - 180)
            measured = self.measured_pitch
            if measured is not None and measured[1] > used_pitch:
                used_pitch = measured[1]
                self.pitch = measured[0] if self.pitch is None else self.pitch + PITCH_PULL * (measured[0] - self.pitch)
            if self.pitch is None:
                self.pitch = 0.0

            if place is not None and self.heading is not None:
                where = self.position(np.array(place, float), time.perf_counter())
                self.publish(self._marker(where, screen))
            self.stopped.wait(max(0.0, PERIOD - (time.perf_counter() - started)))

    def _pitch_loop(self):
        """Reads the pitch from vertical edges a few times a second, beside the fast loop."""
        grab, screen = self.grabber()
        scene_left, scene_top, scene_width, scene_height = scenery_region(*screen)
        scale = WORK_WIDTH / scene_width
        while not self._done():
            started = time.perf_counter()
            scene = cv2.cvtColor(cv2.resize(grab(scene_left, scene_top, scene_width, scene_height), None,
                                            fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            value = estimate_pitch(scene, (scene_left, scene_top), scale, screen)
            if value is not None:
                self.measured_pitch = (value, time.perf_counter())
            self.stopped.wait(max(0.0, PITCH_PERIOD - (time.perf_counter() - started)))

    def _marker(self, player, screen):
        per_metre = geometry.map_distance(self.map_item, 1.0)       # map percent in one metre
        east, south = (np.array(self.spot) - np.array(player)) / per_metre
        metres = float(np.hypot(east, south))
        ring = project([(east + RING_RADIUS * np.cos(a), south + RING_RADIUS * np.sin(a), 0.0)
                        for a in np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)], self.heading, self.pitch, screen)
        foot, head = project([(east, south, 0.0), (east, south, PIN_HEIGHT)], self.heading, self.pitch, screen)
        bearing = np.degrees(np.arctan2(south, east))
        return {
            "ring": None if any(point is None for point in ring) else ring,
            "pin": None if foot is None or head is None else (foot, head),
            "metres": metres,
            "turn": float((bearing - self.heading + 180) % 360 - 180),     # degrees to the right
        }
