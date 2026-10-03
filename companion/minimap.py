"""Reads the planted spike, the player, spotted enemies and teammate deaths off the in-game minimap.

The minimap is matched against the map's reference silhouette at any
rotation, which gives a screen-to-map transform; icon positions found on
screen are then converted to Strats.gg map coordinates.

Tuned on 1080p match footage. Sizes scale with the calibrated minimap zoom.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .map_shape import VIEW_SIZE
from .paths import CALIBRATION_FILE, DATA_DIR

ROI_FRACTION = 0.48            # minimap lives in the top-left square of this fraction of screen height
MIN_SCORE = 0.13               # at the calibrated zoom: real matches 0.18+, no minimap under 0.07
MIN_CALIBRATION_SCORE = 0.16   # a wrong zoom scores under 0.15
TEMPLATE_CACHE = {0.25: 400, 0.5: 160, 1.0: 40}   # outlines kept per resolution
COARSE_ANGLE_STEP = 3
FINE_ANGLE_STEP = 0.75
TRACK_ANGLE_RANGE = 9
CALIBRATION_SCALES = np.arange(0.22, 0.90, 0.02)   # screen pixels per map view unit
DEFAULT_SCALE = 0.405          # zoom seen with default minimap settings at both 1080p and 1440p
LINE_THRESHOLD = 18
ICON_RADIUS = 27.5             # player icon radius in map view units
MIN_WHITE_RING = 0.35
MIN_POINTER_PIXELS = 3        # the facing pointer on the player's icon, after thin lines are removed
MAX_TEAL_RING = 0.15          # teammates' icons have a teal ring
MIN_RED_RING = 0.35           # spotted enemies' icons have a red ring...
MAX_RED_INSIDE = 0.35         # ...around a portrait, not a red fill...
MAX_RED_OUTSIDE = 0.2         # ...on the map, not amid red. Fire seen through the minimap fails these.
# Dead teammates are marked with a blue X (dead enemies with a red one, which is ignored).
DEATH_SIZE = (0.3, 1.3)       # X width and height as a fraction of an icon's diameter
MAX_DEATH_FILL = 0.8          # an X covers part of its box; a solid blob covers nearly all of it
SPIKE_AREA = (0.0001, 0.0015)  # blob area as a fraction of the on-screen map square
# Spike indicator that replaces the round timer once the spike is planted,
# as (left, top, right, bottom) fractions of the frame.
PLANTED_REGION = (0.47, 0.01, 0.53, 0.085)
MIN_PLANTED_RED = 0.15


@dataclass
class Registration:
    score: float
    scale: float
    angle: float
    center: tuple

    def _rotation(self):
        radians = np.deg2rad(-self.angle)
        return np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])

    def to_screen(self, point):
        offset = (np.asarray(point, dtype=float) / 100 * VIEW_SIZE - VIEW_SIZE / 2) * self.scale
        return self._rotation() @ offset + np.asarray(self.center)

    def to_map(self, pixel):
        offset = self._rotation().T @ (np.asarray(pixel, dtype=float) - np.asarray(self.center))
        return tuple((offset / self.scale + VIEW_SIZE / 2) / VIEW_SIZE * 100)


@dataclass
class Reading:
    registration: Registration = None
    planted: bool = False
    spike: tuple = None
    player: tuple = None
    enemies: list = field(default_factory=list)
    deaths: list = field(default_factory=list)      # where teammates died


def _pointer_angle(roi, center, radius):
    """Screen angle (degrees; 0 right, 90 down) of the white pointer on the player's icon ring, or None.

    The pointer is a small solid white triangle on the ring, on the side the
    player faces; thin white map outlines are eroded away first.
    """
    reach = int(radius * 1.7) + 2
    x0, y0 = max(0, int(center[0]) - reach), max(0, int(center[1]) - reach)
    patch = roi[y0:int(center[1]) + reach, x0:int(center[0]) + reach]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 235)).astype(np.uint8)
    white = cv2.erode(white, np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    dx, dy = xs + x0 - center[0], ys + y0 - center[1]
    distance = np.hypot(dx, dy)
    ring = white & (distance > radius) & (distance < radius * 1.6)
    if ring.sum() < MIN_POINTER_PIXELS:
        return None
    angles = np.degrees(np.arctan2(dy[ring], dx[ring]))
    counts, edges = np.histogram(angles, bins=36, range=(-180, 180))
    peak = edges[int(np.argmax(counts))] + 5
    near = np.abs((angles - peak + 180) % 360 - 180) <= 25
    return float(np.degrees(np.arctan2(np.sin(np.radians(angles[near])).mean(),
                                       np.cos(np.radians(angles[near])).mean())))


def spike_planted(frame):
    height, width = frame.shape[:2]
    left, top, right, bottom = PLANTED_REGION
    region = cv2.cvtColor(frame[int(top * height):int(bottom * height), int(left * width):int(right * width)], cv2.COLOR_BGR2HSV)
    red = cv2.inRange(region, (0, 140, 90), (8, 255, 255)) | cv2.inRange(region, (170, 140, 90), (180, 255, 255))
    return float((red > 0).mean()) >= MIN_PLANTED_RED


def _line_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return cv2.GaussianBlur((tophat > LINE_THRESHOLD).astype(np.float32), (0, 0), 1.2)


class _Lines:
    """Line maps of one minimap region, computed once per resolution."""

    def __init__(self, roi):
        self.roi = roi
        self._levels = {}

    def at(self, down):
        if down not in self._levels:
            image = self.roi if down == 1.0 else cv2.resize(self.roi, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
            lines = _line_map(image)
            self._levels[down] = (lines, float(lines.mean()))
        return self._levels[down]


class MinimapReader:
    def __init__(self, silhouette, frame_height, scale=None):
        self.silhouette = silhouette
        self.inside = cv2.dilate(silhouette, np.ones((25, 25), np.uint8))
        # Player icons sit on ledges and map edges the silhouette does not cover.
        self.nearby = cv2.dilate(silhouette, np.ones((121, 121), np.uint8))
        # Where other players can stand: the walkable map, give or take a few pixels.
        self.walkable = cv2.dilate(silhouette, np.ones((9, 9), np.uint8))
        self.frame_height = frame_height
        self.scale = scale
        self.last = None
        self._templates = {}

    # ---- calibration --------------------------------------------------------

    @staticmethod
    def stored_scale(frame_shape):
        if CALIBRATION_FILE.exists():
            return json.loads(CALIBRATION_FILE.read_text()).get(f"{frame_shape[1]}x{frame_shape[0]}")
        return None

    def calibrate(self, frame, thorough=True):
        """Find the minimap zoom from one frame that shows the minimap. Returns the match score.

        The quick passes cover the usual zoom and non-rotating minimaps in a few
        seconds. `thorough` adds a sweep of every zoom at every rotation, which
        takes a minute or two.
        """
        lines = _Lines(self._roi(frame))
        best = self._locate(lines, DEFAULT_SCALE)
        if best.score < MIN_CALIBRATION_SCORE:
            for scale in CALIBRATION_SCALES:
                upright = self._search(lines, float(scale), (0, 90, 180, 270), 0.25)
                candidate = self._refine(lines, float(scale), upright.angle, upright.center)
                if candidate.score > best.score:
                    best = candidate
        if best.score < MIN_CALIBRATION_SCORE and thorough:
            for scale in CALIBRATION_SCALES:
                candidate = self._locate(lines, float(scale))
                if candidate.score > best.score:
                    best = candidate
        # The score peaks sharply at the true zoom, so finish with small steps around the best.
        for scale in np.arange(best.scale - 0.015, best.scale + 0.0151, 0.005):
            candidate = self._refine(lines, float(scale), best.angle, best.center)
            if candidate.score > best.score:
                best = candidate
        if best.score >= MIN_CALIBRATION_SCORE:
            self.scale = round(best.scale, 3)
            self.last = None
            self._templates.clear()
            stored = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
            stored[f"{frame.shape[1]}x{frame.shape[0]}"] = self.scale
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CALIBRATION_FILE.write_text(json.dumps(stored, indent=1))
        return best.score

    # ---- registration -------------------------------------------------------

    def _roi(self, frame):
        side = int(frame.shape[0] * ROI_FRACTION)
        return frame[:side, :side]

    def _template(self, scale, angle, down):
        cache = self._templates.setdefault(down, OrderedDict())
        key = (round(scale, 4), round(float(angle) % 360, 2))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        size = max(8, int(round(VIEW_SIZE * scale * down)))
        small = cv2.resize(self.silhouette, (size, size), interpolation=cv2.INTER_AREA)
        outline = cv2.morphologyEx((small > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        canvas = int(np.ceil(size * 1.42)) | 1
        pad = (canvas - size) // 2
        padded = np.zeros((canvas, canvas), np.float32)
        padded[pad:pad + size, pad:pad + size] = outline
        rotation = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), float(angle), 1.0)
        template = cv2.GaussianBlur(cv2.warpAffine(padded, rotation, (canvas, canvas)), (0, 0), 1.2)
        cache[key] = (template, float(template.sum()))
        if len(cache) > TEMPLATE_CACHE[down]:
            cache.popitem(last=False)
        return cache[key]

    def _search(self, lines, scale, angles, down, near=None, radius=0):
        """Best placement of the outline over the frame's line map.

        With `near` (a full-resolution centre), only placements within `radius`
        full-resolution pixels of it are tried, which is far cheaper.
        """
        image, density = lines.at(down)
        best = Registration(-1.0, scale, 0.0, (0.0, 0.0))
        for angle in angles:
            template, total = self._template(scale, angle, down)
            canvas = template.shape[0]
            margin = canvas // 2
            padded = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
            left = top = 0
            if near is not None:
                reach = max(1, int(round(radius * down)))
                limit_x, limit_y = padded.shape[1] - canvas, padded.shape[0] - canvas
                center_x = int(round(near[0] * down + margin - canvas / 2))
                center_y = int(round(near[1] * down + margin - canvas / 2))
                left, top = min(max(center_x - reach, 0), limit_x), min(max(center_y - reach, 0), limit_y)
                right, bottom = min(max(center_x + reach, 0), limit_x), min(max(center_y + reach, 0), limit_y)
                padded = padded[top:bottom + canvas, left:right + canvas]
            _, peak, _, location = cv2.minMaxLoc(cv2.matchTemplate(padded, template, cv2.TM_CCORR))
            score = peak / total - density
            if score > best.score:
                center = ((location[0] + left - margin + canvas / 2) / down, (location[1] + top - margin + canvas / 2) / down)
                best = Registration(score, scale, float(angle) % 360, center)
        return best

    def _refine(self, lines, scale, angle, center, spread=COARSE_ANGLE_STEP):
        """Sharpen a rough placement: half resolution first, then full resolution."""
        middle = self._search(lines, scale, np.arange(angle - spread, angle + spread + 0.01, 1.5), 0.5, center, 10)
        fine_angles = np.arange(middle.angle - 1.5, middle.angle + 1.51, FINE_ANGLE_STEP)
        return self._search(lines, scale, fine_angles, 1.0, middle.center, 3)

    def _locate(self, lines, scale):
        """Placement of the map with no prior knowledge of its rotation."""
        # Fixed (non-rotating) minimaps sit at a right angle; try those before the full sweep.
        upright = self._search(lines, scale, (0, 90, 180, 270), 0.25)
        best = self._refine(lines, scale, upright.angle, upright.center)
        if best.score >= MIN_CALIBRATION_SCORE:
            return best
        placements = [self._search(lines, scale, (angle,), 0.25) for angle in range(0, 360, COARSE_ANGLE_STEP)]
        for coarse in sorted(placements, key=lambda item: item.score, reverse=True)[:3]:
            candidate = self._refine(lines, scale, coarse.angle, coarse.center)
            if candidate.score > best.score:
                best = candidate
        return best

    def register(self, frame):
        """Screen-to-map transform for this frame, or None when no minimap is visible."""
        lines = _Lines(self._roi(frame))
        found = None
        if self.last is not None:
            found = self._refine(lines, self.scale, self.last.angle, self.last.center, TRACK_ANGLE_RANGE)
        if found is None or found.score < MIN_SCORE:
            found = self._locate(lines, self.scale)
        self.last = found if found.score >= MIN_SCORE else None
        return self.last

    # ---- icons --------------------------------------------------------------

    @staticmethod
    def _on_map(registration, pixel, region):
        x, y = (np.array(registration.to_map(pixel)) / 100 * VIEW_SIZE).astype(int)
        return 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE and region[y, x] > 0

    def _find_spike(self, roi, registration):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.dilate(cv2.inRange(hsv, (20, 150, 170), (34, 255, 255)), np.ones((3, 3), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        map_area = (VIEW_SIZE * registration.scale) ** 2
        best = None
        for index in range(1, count):
            area, width, height = stats[index][cv2.CC_STAT_AREA], stats[index][cv2.CC_STAT_WIDTH], stats[index][cv2.CC_STAT_HEIGHT]
            if not SPIKE_AREA[0] * map_area <= area <= SPIKE_AREA[1] * map_area:
                continue
            if max(width, height) > 2.5 * min(width, height) or not self._on_map(registration, centers[index], self.inside):
                continue
            if best is None or area > best[0]:
                best = (area, centers[index])
        return None if best is None else tuple(best[1])

    def _find_icons(self, roi, registration, offset=(0, 0)):
        """Screen positions of the player's own icon (or None) and of spotted enemies' icons.

        `roi` may be a part of the minimap region starting at `offset`.
        """
        radius = ICON_RADIUS * registration.scale
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 205)
        teal = cv2.inRange(hsv, (70, 70, 120), (100, 255, 255)) > 0
        red = (cv2.inRange(hsv, (0, 120, 120), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))) > 0
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=radius, param1=120, param2=14,
            minRadius=max(3, int(radius * 0.75)), maxRadius=int(np.ceil(radius * 1.3)),
        )
        if circles is None:
            return None, []
        height, width = white.shape
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        player, enemies = None, []

        def share(mask, center_x, center_y, ring_radius):
            xs = np.clip((center_x + ring_radius * np.cos(angles)).astype(int), 0, width - 1)
            ys = np.clip((center_y + ring_radius * np.sin(angles)).astype(int), 0, height - 1)
            return float(mask[ys, xs].mean())

        for center_x, center_y, found_radius in circles[0]:
            ring = teammate = enemy = 0.0
            for ring_radius in (found_radius - 1, found_radius, found_radius + 1):
                ring = max(ring, share(white, center_x, center_y, ring_radius))
                teammate = max(teammate, share(teal, center_x, center_y, ring_radius))
                enemy = max(enemy, share(red, center_x, center_y, ring_radius))
            center = (float(center_x) + offset[0], float(center_y) + offset[1])
            if not self._on_map(registration, center, self.nearby):
                continue
            if enemy >= MIN_RED_RING:
                if (share(red, center_x, center_y, found_radius * 0.5) <= MAX_RED_INSIDE
                        and share(red, center_x, center_y, found_radius * 1.6) <= MAX_RED_OUTSIDE
                        and self._on_map(registration, center, self.walkable)):
                    enemies.append(center)
            elif ring >= MIN_WHITE_RING and teammate <= MAX_TEAL_RING and (player is None or ring > player[0]):
                player = (ring, center)
        return (None if player is None else player[1]), enemies

    def _find_deaths(self, roi, registration):
        diameter = 2 * ICON_RADIUS * registration.scale
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (100, 90, 120), (125, 255, 255))
        count, _, stats, centers = cv2.connectedComponentsWithStats(cv2.dilate(blue, np.ones((2, 2), np.uint8)))
        deaths = []
        for index in range(1, count):
            width, height, area = stats[index][cv2.CC_STAT_WIDTH], stats[index][cv2.CC_STAT_HEIGHT], stats[index][cv2.CC_STAT_AREA]
            if not all(DEATH_SIZE[0] * diameter <= side <= DEATH_SIZE[1] * diameter for side in (width, height)):
                continue
            if area > MAX_DEATH_FILL * width * height or not self._on_map(registration, centers[index], self.inside):
                continue
            deaths.append(tuple(centers[index]))
        return deaths

    def player_and_facing(self, roi, registration, near=None, origin=(0, 0)):
        """(map position, facing as a map-direction angle in degrees, icon screen position), each possibly None.

        `roi` is the minimap region, or the part of it starting at `origin`. With
        `near` (the icon's last screen position) only the area around it is
        searched, which is quick enough to run many times a second.
        """
        offset = origin
        if near is not None:
            reach = self.icon_reach(registration)
            left, top = max(origin[0], int(near[0]) - reach), max(origin[1], int(near[1]) - reach)
            roi = roi[top - origin[1]:int(near[1]) + reach - origin[1], left - origin[0]:int(near[0]) + reach - origin[0]]
            offset = (left, top)
        player, _ = self._find_icons(roi, registration, offset)
        if player is None:
            return None, None, None
        local = (player[0] - offset[0], player[1] - offset[1])
        angle = _pointer_angle(roi, local, ICON_RADIUS * registration.scale)
        facing = None
        if angle is not None:
            # From a screen angle to a direction on the map art (the minimap may be turned).
            start = np.array(registration.to_map(player))
            end = np.array(registration.to_map((player[0] + np.cos(np.radians(angle)) * 10,
                                                player[1] + np.sin(np.radians(angle)) * 10)))
            facing = float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))
        return registration.to_map(player), facing, player

    @staticmethod
    def icon_reach(registration):
        """How far around its last position the player's icon is looked for, in pixels."""
        return int(ICON_RADIUS * registration.scale * 5)

    def find_dropped_spike(self, frame, registration):
        """Map position of a spike icon lying on the map although none is planted (test mode), or None."""
        spike = self._find_spike(self._roi(frame), registration)
        return None if spike is None else registration.to_map(spike)

    def read(self, frame):
        reading = Reading(planted=spike_planted(frame))
        reading.registration = self.register(frame)
        if reading.registration is None:
            return reading
        roi = self._roi(frame)
        player, enemies = self._find_icons(roi, reading.registration)
        if player is not None:
            reading.player = reading.registration.to_map(player)
        reading.enemies = [reading.registration.to_map(enemy) for enemy in enemies]
        reading.deaths = [reading.registration.to_map(death) for death in self._find_deaths(roi, reading.registration)]
        if reading.planted:
            spike = self._find_spike(roi, reading.registration)
            if spike is not None:
                reading.spike = reading.registration.to_map(spike)
        return reading
