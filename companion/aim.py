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
    left, top, right, bottom = WEAPON
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
    def __init__(self, reference):
        self.orb = cv2.ORB_create(FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        gray = _gray(reference, WORK_WIDTH / reference.shape[1])
        width, height = gray.shape[1], gray.shape[0]
        self.keypoints, self.descriptors = self.orb.detectAndCompute(gray, _mask(width, height))
        points = np.float64([keypoint.pt for keypoint in self.keypoints]).reshape(-1, 2)
        self.bearings = _bearings(points, width, height)
        self.near = np.hypot(points[:, 0] - width / 2, points[:, 1] - height / 2) <= NEAR_AIM * width
        self.random = np.random.default_rng(0)

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
