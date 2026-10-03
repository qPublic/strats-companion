"""Where a lineup's aim point is on the live screen, found by matching its aim screenshot.

A Strats.gg aim screenshot is a full game screenshot taken on the standing spot
with the crosshair on the aim point, in the middle of the picture. Standing on
the same spot, the live view differs from it only by how the camera is turned,
and Valorant's field of view is fixed (103 degrees across a 16:9 screen). So
matching the scenery of the two pictures gives the camera rotation between them,
and that rotation says where the aim point is: on screen, or which way to turn.

Standing a little off the spot shifts near and far scenery differently, so the
rotation is finally fitted to the scenery around the aim point, which is about
as far away as the aim point itself.

The HUD, the minimap and the weapon differ between the two pictures and are left
out of the matching.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

HORIZONTAL_FOV = 103.0
WORK_WIDTH = 960                # both pictures are matched at this width
FEATURES = 1500
RATIO = 0.75                    # Lowe's ratio test for a distinctive match
MIN_INLIERS = 15
RANSAC_ROUNDS = 256
INLIER_DEGREES = 0.4            # a match agreeing with the rotation to within this angle
NEAR_AIM = 0.2                  # scenery within this fraction of the screen width of the aim point...
MIN_NEAR = 8                    # ...decides the final fit when there is this much of it
# Parts of the screen used for matching, as (left, top, right, bottom) fractions;
# what is left out is the HUD and the weapon in the lower right.
SCENERY = (0.12, 0.11, 0.88, 0.80)
WEAPON = (0.58, 0.36, 1.0, 1.0)
# Also left out: things drawn in the same place on every screen, which match themselves and say
# nothing about where the camera points. The minimap corner (16:9) and the performance graphs
# many players, and Strats.gg's own screenshots, show at the top right.
FIXED_ON_SCREEN = ((0.0, 0.0, 0.27, 0.48), (0.80, 0.0, 1.0, 0.32))


@dataclass
class Aim:
    point: tuple            # where the aim point is in screen pixels (may be off screen)
    on_screen: bool
    yaw: float              # degrees to turn right (negative: left) to put the crosshair on it
    pitch: float            # degrees to look up (negative: down)
    matches: int


def scenery_region(width, height):
    """The part of a width x height screen worth capturing for matching, as (left, top, width, height)."""
    left, top, right, bottom = SCENERY
    return int(left * width), int(top * height), int((right - left) * width), int((bottom - top) * height)


def _mask(width, height, origin=(0, 0), size=None):
    """Matching mask for a crop of a width x height screen (working pixels) starting at `origin`."""
    mask = np.zeros((height, width), np.uint8)
    left, top, right, bottom = SCENERY
    mask[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 255
    for left, top, right, bottom in (WEAPON, *FIXED_ON_SCREEN):
        mask[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0
    if size is None:
        return mask
    crop = np.zeros((size[1], size[0]), np.uint8)
    part = mask[origin[1]:origin[1] + size[1], origin[0]:origin[0] + size[0]]
    crop[:part.shape[0], :part.shape[1]] = part
    return crop


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _gray(image, scale):
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Even out lighting so the same wall matches under different exposure.
    return _clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))


def _focal(width):
    return (width / 2) / np.tan(np.radians(HORIZONTAL_FOV / 2))


def _bearings(points, width, height):
    """Unit view directions (x right, y down, z forward) for pixel positions on a width x height screen."""
    focal = _focal(width)
    rays = np.column_stack([(points[:, 0] - width / 2) / focal, (points[:, 1] - height / 2) / focal, np.ones(len(points))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotations(source, target):
    """Least-squares rotations taking source directions onto target directions (Kabsch), for stacks of sets."""
    u, _, vt = np.linalg.svd(np.einsum("...ki,...kj->...ij", source, target))
    v, ut = np.swapaxes(vt, -1, -2), np.swapaxes(u, -1, -2)
    flip = np.sign(np.linalg.det(v @ ut))
    fix = np.zeros(u.shape)
    fix[..., 0, 0] = fix[..., 1, 1] = 1.0
    fix[..., 2, 2] = flip
    return v @ fix @ ut


class AimGuide:
    """Everything needed to find one lineup's aim point on screen: the features of its aim screenshot.

    Built from the screenshot once (`from_picture`), stored (`save`), and loaded
    again in a few milliseconds (`load`), so nothing is fetched or computed when
    the lineup comes up in a match.
    """

    def __init__(self, points, descriptors, size):
        self.orb = cv2.ORB_create(FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.points = np.asarray(points, np.float64).reshape(-1, 2)
        self.descriptors = None if descriptors is None or len(descriptors) == 0 else np.asarray(descriptors, np.uint8)
        self.size = (int(size[0]), int(size[1]))
        if self.descriptors is not None and len(self.points):
            # Data stored by older versions may hold features from parts now left out of matching.
            mask = _mask(*self.size)
            keep = mask[np.clip(self.points[:, 1].astype(int), 0, self.size[1] - 1),
                        np.clip(self.points[:, 0].astype(int), 0, self.size[0] - 1)] > 0
            self.points, self.descriptors = self.points[keep], self.descriptors[keep]
        self.keypoints = self.points             # only their number and positions are used
        width, height = self.size
        self.bearings = _bearings(self.points, width, height)
        self.near = np.hypot(self.points[:, 0] - width / 2, self.points[:, 1] - height / 2) <= NEAR_AIM * width
        self.random = np.random.default_rng(0)

    @classmethod
    def from_picture(cls, reference):
        gray = _gray(reference, WORK_WIDTH / reference.shape[1])
        keypoints, descriptors = cv2.ORB_create(FEATURES).detectAndCompute(gray, _mask(gray.shape[1], gray.shape[0]))
        points = np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1, 2)
        return cls(points, descriptors, (gray.shape[1], gray.shape[0]))

    def save(self, path):
        descriptors = np.zeros((0, 32), np.uint8) if self.descriptors is None else self.descriptors
        np.savez_compressed(path, points=self.points.astype(np.float32), descriptors=descriptors,
                            size=np.array(self.size))

    @classmethod
    def load(cls, path):
        with np.load(path) as stored:
            return cls(stored["points"], stored["descriptors"], stored["size"])

    def locate(self, image, origin=(0, 0), screen=None):
        """An Aim, or None when too little of the screenshot's scenery is in view.

        `image` is the screen, or the part of it at `origin` (pixels) when `screen`
        gives the whole screen's (width, height).
        """
        if self.descriptors is None or len(self.keypoints) < MIN_INLIERS:
            return None
        screen = screen or (image.shape[1], image.shape[0])
        scale = WORK_WIDTH / screen[0]
        gray = _gray(image, scale)
        work_screen = (int(round(screen[0] * scale)), int(round(screen[1] * scale)))
        work_origin = (int(round(origin[0] * scale)), int(round(origin[1] * scale)))
        mask = _mask(*work_screen, work_origin, (gray.shape[1], gray.shape[0]))
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        reference = np.array([match.queryIdx for match in good])
        source = self.bearings[reference]
        found = np.float64([keypoints[match.trainIdx].pt for match in good]) + work_origin
        target = _bearings(found, *work_screen)

        # RANSAC over rotations from pairs of matches, all rounds at once.
        picks = self.random.integers(0, len(good), (RANSAC_ROUNDS, 2))
        picks = picks[picks[:, 0] != picks[:, 1]]
        candidates = _rotations(source[picks], target[picks])
        agreement = np.einsum("rij,nj,ni->rn", candidates, source, target)
        inliers = agreement > np.cos(np.radians(INLIER_DEGREES))
        best = inliers[np.argmax(inliers.sum(axis=1))]
        if int(best.sum()) < MIN_INLIERS:
            return None
        rotation = _rotations(source[best], target[best])
        # Off the exact spot, near and far scenery disagree; trust what is around the aim point.
        near = best & self.near[reference]
        if near.sum() >= MIN_NEAR:
            rotation = _rotations(source[near], target[near])

        # The screenshot's crosshair looks straight ahead; where does that direction lie now?
        x, y, z = rotation @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.degrees(np.arctan2(x, z)))
        pitch = float(-np.degrees(np.arctan2(y, np.hypot(x, z))))
        focal = _focal(work_screen[0])
        if z > 1e-6:
            point = ((x / z * focal + work_screen[0] / 2) / scale, (y / z * focal + work_screen[1] / 2) / scale)
        else:
            point = (np.sign(x) * 1e6, np.sign(y) * 1e6)
        on_screen = z > 0 and 0 <= point[0] < screen[0] and 0 <= point[1] < screen[1]
        return Aim((float(point[0]), float(point[1])), bool(on_screen), yaw, pitch, int(best.sum()))


# ---- following the aim point between full matches ---------------------------

FOLLOW_PERIOD = 1 / 90          # the reticle follows the scenery this often
MATCH_PERIOD = 1 / 12           # full matches, which correct any drift, this often
PATCH = 72                      # side of the patch of scenery around the aim point that is followed (screen pixels)
SEARCH = 150                    # how far that patch may move between two follows
MIN_FOLLOW_SCORE = 0.6
UNIQUE_GAP = 6                  # pixels around the best place that are the same place
MIN_UNIQUENESS = 0.12           # how much better the best place must fit than anywhere else
FOLLOW_SILENCE = 0.15           # seconds without a follow before full matches are shown as they are
MIN_PATCH_CONTRAST = 6.0        # a featureless patch (sky, a flat wall) cannot be followed
DRIFT_PIXELS = 1.5              # a full match disagreeing by more than this corrects the follower
HISTORY_SECONDS = 1.0


class _OneEuro:
    """One-euro filter: steadies a still point without making a moving one lag."""

    def __init__(self, min_cutoff=1.5, beta=0.06, derivative_cutoff=1.0):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.value = self.speed = self.time = None

    @staticmethod
    def _alpha(cutoff, step):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / step)

    def __call__(self, value, now):
        value = np.asarray(value, float)
        if self.value is None or now <= self.time:
            self.value, self.speed, self.time = value, np.zeros_like(value), now
            return value
        step = now - self.time
        speed = (value - self.value) / step
        self.speed = self.speed + self._alpha(self.derivative_cutoff, step) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.speed)
        self.value = self.value + self._alpha(cutoff, step) * (value - self.value)
        self.time = now
        return self.value

    def reset(self):
        self.value = None


def screen_grabber():
    """A grab(left, top, width, height) -> BGR function for the primary monitor, and its size."""
    from . import win

    return (lambda left, top, width, height: win.capture_screen((left, top, width, height), copy=False)), win.screen_size()


class Tracker:
    """Keeps the aim point locked to the scenery while the player is on the spot.

    Two loops: full matches against the aim screenshot (AimGuide.locate, tens of
    milliseconds) find the aim point, and in between a small patch of scenery
    around it is followed from frame to frame by template matching (about a
    millisecond), so the reticle moves with the world as the camera turns. A
    full match is compared with where the follower was when that match's picture
    was taken, so a late result corrects drift without pulling the reticle back.
    """

    def __init__(self, matcher, publish, grabber=screen_grabber, also_stop=None):
        self.matcher, self.publish, self.grabber = matcher, publish, grabber
        self.stopped = threading.Event()
        self.also_stop = also_stop or threading.Event()
        self.lock = threading.Lock()
        self.point = self.template = self.aim = None
        self.history = deque()
        self.filter = _OneEuro()
        self.last_follow = 0.0

    def start(self):
        for loop in (self._match_loop, self._follow_loop):
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

    def _send(self, aim):
        if not self._done():
            self.publish(aim)

    def _point_at(self, moment):
        """Where the follower had the aim point at `moment`, or None."""
        if not self.history:
            return None
        when, point = min(self.history, key=lambda item: abs(item[0] - moment))
        return point if abs(when - moment) <= 0.1 else None

    def _match_loop(self):
        grab, screen = self.grabber()
        left, top, width, height = scenery_region(*screen)
        while not self._done():
            started = time.perf_counter()
            image = grab(left, top, width, height)
            aim = self.matcher.locate(image, (left, top), screen)
            if aim is None or not aim.on_screen:
                with self.lock:
                    self.point = self.template = None
                    self.history.clear()
                self.filter.reset()
                self._send(aim)
            else:
                template = self._patch(image, aim.point[0] - left, aim.point[1] - top)
                with self.lock:
                    seen = self._point_at(started)
                    if self.point is None or seen is None:
                        self.point = np.array(aim.point)
                    else:
                        drift = np.array(aim.point) - seen
                        if np.linalg.norm(drift) > DRIFT_PIXELS:
                            self.point = self.point + drift
                    self.template, self.aim = template, aim
                if template is None or time.perf_counter() - self.last_follow > FOLLOW_SILENCE:
                    self._send(aim)          # nothing (reliable) to follow: show the full match as it is
            self.stopped.wait(max(0.0, MATCH_PERIOD - (time.perf_counter() - started)))

    @staticmethod
    def _patch(image, x, y):
        half = PATCH // 2
        x, y = int(round(x)), int(round(y))
        if not (half <= x < image.shape[1] - half and half <= y < image.shape[0] - half):
            return None
        patch = cv2.cvtColor(np.ascontiguousarray(image[y - half:y + half, x - half:x + half]), cv2.COLOR_BGR2GRAY)
        return patch if float(patch.std()) >= MIN_PATCH_CONTRAST else None

    def _follow_loop(self):
        grab, screen = self.grabber()
        while not self._done():
            started = time.perf_counter()
            with self.lock:
                point, template, aim = self.point, self.template, self.aim
            if point is not None and template is not None:
                found = self._follow(grab, screen, point, template)
                if found is not None:
                    now = time.perf_counter()
                    with self.lock:
                        # A full match may have moved the point meanwhile; keep its correction.
                        if self.point is point:
                            self.point = found
                        self.history.append((started, found))
                        while self.history and now - self.history[0][0] > HISTORY_SECONDS:
                            self.history.popleft()
                    self.last_follow = now
                    steady = self.filter(found, now)
                    on_screen = 0 <= steady[0] < screen[0] and 0 <= steady[1] < screen[1]
                    self._send(Aim((float(steady[0]), float(steady[1])), bool(on_screen), aim.yaw, aim.pitch, aim.matches))
            self.stopped.wait(max(0.0, FOLLOW_PERIOD - (time.perf_counter() - started)))

    @staticmethod
    def _follow(grab, screen, point, template):
        """Where the template is now, near `point`, to a fraction of a pixel; None when it is lost."""
        half = PATCH // 2
        size = 2 * (SEARCH + half)
        left = int(max(0, min(screen[0] - size, point[0] - SEARCH - half)))
        top = int(max(0, min(screen[1] - size, point[1] - SEARCH - half)))
        window = cv2.cvtColor(np.ascontiguousarray(grab(left, top, size, size)), cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        if best < MIN_FOLLOW_SCORE:
            return None
        # Along a straight edge or across a plain wall the patch fits almost as well a little way off;
        # then the follower could slide. Only a clear single best place counts.
        others = scores.copy()
        others[max(0, y - UNIQUE_GAP):y + UNIQUE_GAP + 1, max(0, x - UNIQUE_GAP):x + UNIQUE_GAP + 1] = -1
        if best - float(others.max()) < MIN_UNIQUENESS:
            return None

        def offset(before, peak, after):
            curve = before - 2 * peak + after
            return 0.0 if curve == 0 else 0.5 * (before - after) / curve

        dx = offset(scores[y, x - 1], best, scores[y, x + 1]) if 0 < x < scores.shape[1] - 1 else 0.0
        dy = offset(scores[y - 1, x], best, scores[y + 1, x]) if 0 < y < scores.shape[0] - 1 else 0.0
        return np.array([left + x + dx + half, top + y + dy + half])
