"""Reads the planted spike and the player's position off the in-game minimap.

The minimap is matched against the map's reference silhouette at any
rotation, which gives a screen-to-map transform; icon positions found on
screen are then converted to Strats.gg map coordinates.

Tuned on 1080p match footage. Sizes scale with the calibrated minimap zoom.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np

from .map_shape import VIEW_SIZE
from .paths import CALIBRATION_FILE, DATA_DIR

ROI_FRACTION = 0.48            # minimap lives in the top-left square of this fraction of screen height
MIN_SCORE = 0.17               # real matches score 0.2+, a wrong zoom under 0.15, no minimap about 0.03
COARSE_ANGLE_STEP = 3
FINE_ANGLE_STEP = 0.75
TRACK_ANGLE_RANGE = 9
CALIBRATION_SCALES = np.arange(0.22, 0.66, 0.02)   # at 1080p; multiplied by height / 1080
LINE_THRESHOLD = 18
ICON_RADIUS = 27.5             # player icon radius in map view units
MIN_WHITE_RING = 0.35
MAX_TEAL_RING = 0.15          # teammates' icons have a teal ring
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


class MinimapReader:
    def __init__(self, silhouette, frame_height, scale=None):
        self.silhouette = silhouette
        self.inside = cv2.dilate(silhouette, np.ones((25, 25), np.uint8))
        # Player icons sit on ledges and map edges the silhouette does not cover.
        self.nearby = cv2.dilate(silhouette, np.ones((121, 121), np.uint8))
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

    def calibrate(self, frame):
        """Find the minimap zoom from one frame that shows the minimap. Returns the match score."""
        roi = self._roi(frame)
        best = None
        # Half-resolution scores barely separate zoom levels; the full-resolution score does.
        for scale in CALIBRATION_SCALES * self.frame_height / 1080:
            coarse = self._search(roi, float(scale), range(0, 360, COARSE_ANGLE_STEP), 0.5)
            candidate = self._search(roi, float(scale), self._fine_angles(coarse.angle), 1.0)
            if best is None or candidate.score > best.score:
                best = candidate
        for scale in (best.scale - 0.01, best.scale + 0.01):
            candidate = self._search(roi, float(scale), self._fine_angles(best.angle), 1.0)
            if candidate.score > best.score:
                best = candidate
        if best.score >= MIN_SCORE:
            self.scale = round(best.scale, 3)
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
        key = (round(scale, 3), round(float(angle) % 360, 2), down)
        if key not in self._templates:
            size = max(8, int(round(VIEW_SIZE * scale * down)))
            small = cv2.resize(self.silhouette, (size, size), interpolation=cv2.INTER_AREA)
            outline = cv2.morphologyEx((small > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            canvas = int(np.ceil(size * 1.42)) | 1
            pad = (canvas - size) // 2
            padded = np.zeros((canvas, canvas), np.float32)
            padded[pad:pad + size, pad:pad + size] = outline
            rotation = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), float(angle), 1.0)
            template = cv2.GaussianBlur(cv2.warpAffine(padded, rotation, (canvas, canvas)), (0, 0), 1.2)
            if len(self._templates) > 400:
                self._templates.clear()
            self._templates[key] = (template, float(template.sum()))
        return self._templates[key]

    @staticmethod
    def _fine_angles(angle):
        return np.arange(angle - COARSE_ANGLE_STEP, angle + COARSE_ANGLE_STEP + 0.01, FINE_ANGLE_STEP)

    def _search(self, roi, scale, angles, down, lines=None):
        if lines is None:
            small = roi if down == 1.0 else cv2.resize(roi, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
            lines = _line_map(small)
        density = float(lines.mean())
        best = Registration(-1.0, scale, 0.0, (0.0, 0.0))
        for angle in angles:
            template, total = self._template(scale, angle, down)
            canvas = template.shape[0]
            margin = canvas // 2
            padded = cv2.copyMakeBorder(lines, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
            _, peak, _, location = cv2.minMaxLoc(cv2.matchTemplate(padded, template, cv2.TM_CCORR))
            score = peak / total - density
            if score > best.score:
                center = ((location[0] - margin + canvas / 2) / down, (location[1] - margin + canvas / 2) / down)
                best = Registration(score, scale, float(angle) % 360, center)
        return best

    def register(self, frame):
        """Screen-to-map transform for this frame, or None when no minimap is visible."""
        roi = self._roi(frame)
        half = _line_map(cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        coarse = None
        if self.last is not None:
            nearby = np.arange(self.last.angle - TRACK_ANGLE_RANGE, self.last.angle + TRACK_ANGLE_RANGE + 0.1, COARSE_ANGLE_STEP)
            coarse = self._search(roi, self.scale, nearby, 0.5, half)
            if coarse.score < MIN_SCORE * 0.8:
                coarse = None
        if coarse is None:
            coarse = self._search(roi, self.scale, range(0, 360, COARSE_ANGLE_STEP), 0.5, half)
        fine = self._search(roi, self.scale, self._fine_angles(coarse.angle), 1.0)
        self.last = fine if fine.score >= MIN_SCORE else None
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

    def _find_player(self, roi, registration):
        radius = ICON_RADIUS * registration.scale
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 205)
        teal = cv2.inRange(hsv, (70, 70, 120), (100, 255, 255)) > 0
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=radius, param1=120, param2=14,
            minRadius=max(3, int(radius * 0.75)), maxRadius=int(np.ceil(radius * 1.3)),
        )
        if circles is None:
            return None
        height, width = white.shape
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        best = None
        for center_x, center_y, found_radius in circles[0]:
            ring = teammate = 0.0
            for ring_radius in (found_radius - 1, found_radius, found_radius + 1):
                xs = np.clip((center_x + ring_radius * np.cos(angles)).astype(int), 0, width - 1)
                ys = np.clip((center_y + ring_radius * np.sin(angles)).astype(int), 0, height - 1)
                ring = max(ring, float(white[ys, xs].mean()))
                teammate = max(teammate, float(teal[ys, xs].mean()))
            if ring < MIN_WHITE_RING or teammate > MAX_TEAL_RING:
                continue
            if self._on_map(registration, (center_x, center_y), self.nearby) and (best is None or ring > best[0]):
                best = (ring, (float(center_x), float(center_y)))
        return None if best is None else best[1]

    def read(self, frame):
        reading = Reading(planted=spike_planted(frame))
        reading.registration = self.register(frame)
        if reading.registration is None:
            return reading
        roi = self._roi(frame)
        player = self._find_player(roi, reading.registration)
        if player is not None:
            reading.player = reading.registration.to_map(player)
        if reading.planted:
            spike = self._find_spike(roi, reading.registration)
            if spike is not None:
                reading.spike = reading.registration.to_map(spike)
        return reading
