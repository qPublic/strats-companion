"""Where a lineup's aim point is on the live screen, found by matching its aim screenshot.

A Strats.gg aim screenshot is a full game screenshot taken on the standing spot
with the crosshair on the aim point, in the middle of the picture. Standing on
the same spot, the live view differs from it only by how the camera is turned,
and Valorant's field of view is fixed (103 degrees across a 16:9 screen). So
matching the scenery of the two pictures gives the camera rotation between them,
and that rotation says where the aim point is: on screen, or which way to turn.

The HUD, the minimap and the weapon differ between the two pictures and are left
out of the matching.
"""

from dataclasses import dataclass

import cv2
import numpy as np

HORIZONTAL_FOV = 103.0
WORK_WIDTH = 1280               # both pictures are matched at this width
FEATURES = 2500
RATIO = 0.75                    # Lowe's ratio test for a distinctive match
MIN_INLIERS = 15
RANSAC_ROUNDS = 300
INLIER_DEGREES = 0.35           # a match agreeing with the rotation to within this angle
# Parts of the screen used for matching, as (left, top, right, bottom) fractions;
# what is left out is the HUD and the weapon in the lower right.
SCENERY = (0.12, 0.11, 0.88, 0.80)
WEAPON = (0.58, 0.36, 1.0, 1.0)


@dataclass
class Aim:
    point: tuple            # where the aim point is in frame pixels (may be off screen)
    on_screen: bool
    yaw: float              # degrees to turn right (negative: left) to put the crosshair on it
    pitch: float            # degrees to look up (negative: down)
    matches: int


def _mask(height, width):
    mask = np.zeros((height, width), np.uint8)
    left, top, right, bottom = SCENERY
    mask[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 255
    left, top, right, bottom = WEAPON
    mask[int(top * height):int(bottom * height), int(left * width):int(right * width)] = 0
    return mask


def _prepare(image):
    scale = WORK_WIDTH / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Even out lighting so the same wall matches under different exposure.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), scale


def _focal(width):
    return (width / 2) / np.tan(np.radians(HORIZONTAL_FOV / 2))


def _bearings(points, width, height):
    """Unit view directions (x right, y down, z forward) for pixel positions."""
    focal = _focal(width)
    rays = np.column_stack([(points[:, 0] - width / 2) / focal, (points[:, 1] - height / 2) / focal, np.ones(len(points))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _rotation(source, target):
    """The rotation taking `source` directions onto `target` directions, least squares (Kabsch)."""
    u, _, vt = np.linalg.svd(source.T @ target)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


class AimGuide:
    def __init__(self, reference):
        self.orb = cv2.ORB_create(FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        gray, _ = _prepare(reference)
        self.size = (gray.shape[1], gray.shape[0])
        self.keypoints, self.descriptors = self.orb.detectAndCompute(gray, _mask(*gray.shape))
        self.random = np.random.default_rng(0)

    def locate(self, frame):
        """An Aim for this frame, or None when too little of the screenshot's scenery is in view."""
        if self.descriptors is None or len(self.keypoints) < MIN_INLIERS:
            return None
        gray, scale = _prepare(frame)
        keypoints, descriptors = self.orb.detectAndCompute(gray, _mask(*gray.shape))
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        width, height = gray.shape[1], gray.shape[0]
        source = _bearings(np.float64([self.keypoints[m.queryIdx].pt for m in good]), *self.size)
        target = _bearings(np.float64([keypoints[m.trainIdx].pt for m in good]), width, height)

        limit = np.cos(np.radians(INLIER_DEGREES))
        best = None
        for _ in range(RANSAC_ROUNDS):
            pick = self.random.choice(len(good), 2, replace=False)
            rotation = _rotation(source[pick], target[pick])
            inliers = np.einsum("ij,ij->i", source @ rotation.T, target) > limit
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or int(best.sum()) < MIN_INLIERS:
            return None
        rotation = _rotation(source[best], target[best])

        # The screenshot's crosshair looks straight ahead; where does that direction lie now?
        x, y, z = rotation @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.degrees(np.arctan2(x, z)))
        pitch = float(-np.degrees(np.arctan2(y, np.hypot(x, z))))
        focal = _focal(width)
        if z > 1e-6:
            point = ((x / z * focal + width / 2) / scale, (y / z * focal + height / 2) / scale)
        else:
            point = (np.sign(x) * 1e6, np.sign(y) * 1e6)
        on_screen = z > 0 and 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]
        return Aim((float(point[0]), float(point[1])), bool(on_screen), yaw, pitch, int(best.sum()))
