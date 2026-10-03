"""Where a lineup is really thrown from, read off the minimap in its aim screenshot.

Strats.gg's standing spots are dots placed by hand on its map, and can be a
metre or two out. The aim screenshot was taken standing on the spot, and its
minimap shows the author's icon there, so registering that minimap gives the
spot as the game drew it. Authors use their own minimap zoom and rotation, so
the zooms already seen are tried first, and a full search runs only for a new one.
"""

import json

import cv2
import numpy as np

from . import geometry, map_shape, strats_api
from .minimap import CALIBRATION_SCALES, MIN_CALIBRATION_SCORE, ROI_FRACTION, MinimapReader, _Lines
from .paths import CALIBRATION_FILE

MAX_CORRECTION_METRES = 4.0     # a reading further than this from Strats.gg's dot is a misread
SCALES_KEY = "screenshot_scales"


def minimap_path(lineup_id):
    return strats_api.AIM_DIR / f"{lineup_id}_minimap.jpg"


def _result_path(lineup_id):
    return strats_api.AIM_DIR / f"{lineup_id}_spot.json"


def save_minimap(lineup_id, picture):
    """Keep the minimap corner of an aim screenshot (full resolution) for reading the spot later."""
    side = int(picture.shape[0] * ROI_FRACTION)
    strats_api.AIM_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(minimap_path(lineup_id)), picture[:side, :side], [cv2.IMWRITE_JPEG_QUALITY, 92])


def known(lineup_id):
    return _result_path(lineup_id).exists()


def _known_scales():
    try:
        return json.loads(CALIBRATION_FILE.read_text()).get(SCALES_KEY, [])
    except (OSError, ValueError):
        return []


def _remember_scale(scale):
    try:
        stored = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
    except (OSError, ValueError):
        stored = {}
    scales = stored.get(SCALES_KEY, [])
    if scale not in scales:
        stored[SCALES_KEY] = scales + [scale]
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FILE.write_text(json.dumps(stored, indent=1))


def true_spot(lineup, map_item):
    """The standing spot read from the lineup's aim screenshot (map coordinates), or None if it cannot be.

    Slow the first time for a lineup (seconds; a minute for an author's new
    zoom), then remembered.
    """
    result = _result_path(lineup["id"])
    if result.exists():
        stored = json.loads(result.read_text())["spot"]
        return None if stored is None else tuple(stored)
    crop_path = minimap_path(lineup["id"])
    if not crop_path.exists():
        _fetch_minimap(lineup)
    if not crop_path.exists():
        return None
    crop = cv2.imread(str(crop_path))
    frame_height = int(round(crop.shape[0] / ROI_FRACTION))
    reader = MinimapReader(map_shape.silhouette(map_item, lineup["side"]), frame_height)
    lines = _Lines(crop)
    best = None
    for scale in _known_scales():
        placement = reader._locate(lines, float(scale))
        if placement.score >= MIN_CALIBRATION_SCORE and (best is None or placement.score > best.score):
            best = placement
    if best is None:
        # An author with a zoom not seen before: try them all (this is the slow part).
        for scale in CALIBRATION_SCALES:
            placement = reader._locate(lines, float(scale))
            if best is None or placement.score > best.score:
                best = placement
        if best.score >= MIN_CALIBRATION_SCORE:
            for scale in np.arange(best.scale - 0.015, best.scale + 0.0151, 0.005):
                placement = reader._refine(lines, float(scale), best.angle, best.center)
                if placement.score > best.score:
                    best = placement
            _remember_scale(round(best.scale, 3))
    spot = None
    if best is not None and best.score >= MIN_CALIBRATION_SCORE:
        icon, _ = reader._find_icons(crop, best)
        if icon is not None:
            found = tuple(float(v) for v in best.to_map(icon))
            dot = geometry.standing_spot(lineup)
            if geometry.metres(map_item, found, dot) <= MAX_CORRECTION_METRES:
                spot = found
    result.write_text(json.dumps({"spot": spot}))
    return spot


def _fetch_minimap(lineup):
    """For aim data stored before minimap corners were kept: get the screenshot again if it can be had."""
    from . import app_cache

    try:
        key = strats_api.lineup_detail(lineup["id"]).get("imageKey")
    except Exception:  # noqa: BLE001 - no detail, no screenshot
        return
    content = app_cache.file(key) if key else None
    if content:
        picture = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if picture is not None:
            save_minimap(lineup["id"], picture)
