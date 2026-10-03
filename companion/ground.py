"""A marker on the ground at the lineup's standing spot, drawn where it is in the game world.

Placing a point of the world on screen needs where the camera is and how it is
turned. All of it comes from the screen:

- position: the player's icon on the minimap;
- heading: the pointer on that icon, which shows the way the player faces;
- pitch (looking up or down): vertical edges of walls and doors lean towards a
  common point that moves with the pitch;
- between those readings, the camera's turn from one frame to the next, from
  scenery matched between the two frames, which follows every small movement
  of the mouse and ignores anything fixed on screen (crosshair, HUD).

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
from .aim import HORIZONTAL_FOV, WEAPON, _OneEuro, _bearings, _mask, _rotations, scenery_region, screen_grabber
from .minimap import ROI_FRACTION

EYE_HEIGHT = 1.55               # metres from the floor to the camera
RING_RADIUS = 0.55              # metres
PIN_HEIGHT = 2.2                # metres: a post standing in the ring, visible over low cover
RING_POINTS = 36
PERIOD = 1 / 40
WORK_WIDTH = 960                # the scene is read at this width
TURN_WIDTH = 640                # frames are matched to each other at this width
TURN_FEATURES = 700
TURN_RANSAC_ROUNDS = 128
TURN_INLIER_DEGREES = 0.5
MIN_TURN_MATCHES = 12
STILL_DEGREES = 0.05            # a match that moved less than this stayed put on screen
MOVING_SHARE = 0.5              # the camera turned if more than this share of the matches moved
# The absolute readings are noisy; each only nudges the estimate, which follows the frame-to-frame turns.
HEADING_PULL = 0.03             # per minimap reading (about 30 a second): settles in a second or so
PITCH_PULL = 0.08               # per vertical-edge reading (4 a second): settles in a few seconds
MAX_STEP_METRES = 2.5           # a minimap position this far from the last one is a misread
PITCH_PERIOD = 0.25             # seconds between vertical-edge readings (a separate thread)
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

    def __init__(self, reader, map_item, spot, publish, also_stop=None, grabber=screen_grabber):
        self.reader, self.map_item, self.spot, self.publish = reader, map_item, spot, publish
        self.grabber = grabber
        self.registration = None          # kept up to date by the watcher's minimap reads
        self.stopped = threading.Event()
        self.also_stop = also_stop or threading.Event()
        self.heading = self.pitch = None
        self.position = _OneEuro(min_cutoff=0.5, beta=0.4)
        self.last_place = None
        self.icon = None
        self.measured_pitch = None      # (degrees, when) from the vertical-edge thread
        self.used_pitch = 0.0

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
        area = scenery_region(*screen)
        scale = TURN_WIDTH / area[2]
        orb, matcher = cv2.ORB_create(TURN_FEATURES), cv2.BFMatcher(cv2.NORM_HAMMING)
        work_screen = (int(round(width * scale)), int(round(height * scale)))
        work_origin = (int(round(area[0] * scale)), int(round(area[1] * scale)))
        mask = None
        previous = None
        random = np.random.default_rng(0)
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

            # The camera's turn since the last frame, from scenery matched between the two.
            gray = cv2.cvtColor(cv2.resize(grab(*area), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            if mask is None or mask.shape != gray.shape:
                mask = _mask(*work_screen, work_origin, (gray.shape[1], gray.shape[0]))
            keypoints, descriptors = orb.detectAndCompute(gray, mask)
            current = None
            if descriptors is not None and len(keypoints) >= MIN_TURN_MATCHES:
                points = np.float64([k.pt for k in keypoints]) + work_origin
                current = (_bearings(points, *work_screen), descriptors)
            if previous is not None and current is not None and self.heading is not None:
                turn = self._turn(previous, current, matcher, random)
                if turn is not None:
                    self.heading += turn[0]
                    self.pitch += turn[1]
            previous = current

            if facing is not None:
                if self.heading is None:
                    self.heading = facing
                else:
                    self.heading += HEADING_PULL * ((facing - self.heading + 180) % 360 - 180)
            measured = self.measured_pitch
            if measured is not None and measured[1] > self.used_pitch:
                self.used_pitch = measured[1]
                self.pitch = measured[0] if self.pitch is None else self.pitch + PITCH_PULL * (measured[0] - self.pitch)
            if self.pitch is None:
                self.pitch = 0.0

            if place is not None and self.last_place is not None and \
                    geometry.metres(self.map_item, place, self.last_place) > MAX_STEP_METRES:
                place = None                      # the icon search caught something else for a moment
            if place is not None:
                self.last_place = place
            if self.last_place is not None and self.heading is not None:
                where = self.position(np.array(self.last_place, float), time.perf_counter())
                self.publish(self._marker(where, screen))
            self.stopped.wait(max(0.0, PERIOD - (time.perf_counter() - started)))

    @staticmethod
    def _turn(previous, current, matcher, random):
        """(degrees right, degrees up) the camera turned between two frames, or None if they do not match."""
        pairs = matcher.knnMatch(previous[1], current[1], k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < MIN_TURN_MATCHES:
            return None
        source = previous[0][[m.queryIdx for m in good]]
        target = current[0][[m.trainIdx for m in good]]
        # Things fixed on screen (crosshair, banners, the HUD) match themselves in place. Once the
        # scenery is clearly moving, those matches say nothing about the turn: leave them out.
        moved = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", source, target), -1, 1)))
        # Only when most of it moves: drifting clouds or water alone are not the camera turning.
        if (moved > STILL_DEGREES).sum() >= MIN_TURN_MATCHES and (moved > STILL_DEGREES).mean() > MOVING_SHARE:
            keep = moved > STILL_DEGREES
            source, target = source[keep], target[keep]
        picks = random.integers(0, len(source), (TURN_RANSAC_ROUNDS, 2))
        picks = picks[picks[:, 0] != picks[:, 1]]
        candidates = _rotations(source[picks], target[picks])
        inliers = np.einsum("rij,nj,ni->rn", candidates, source, target) > np.cos(np.radians(TURN_INLIER_DEGREES))
        best = inliers[np.argmax(inliers.sum(axis=1))]
        if best.sum() < MIN_TURN_MATCHES:
            return None
        rotation = _rotations(source[best], target[best])
        # Where the last frame's straight-ahead now appears: left of centre after turning right, below after looking up.
        x, y, z = rotation @ np.array([0.0, 0.0, 1.0])
        return float(-np.degrees(np.arctan2(x, z))), float(np.degrees(np.arctan2(y, np.hypot(x, z))))

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
