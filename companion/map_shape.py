"""Reference silhouette of a map, taken from the same SVG that Strats.gg draws.

The silhouette is in Strats.gg map coordinates (a 1024 px square is 100%), so
anything matched against it converts straight to lineup coordinates.
"""

import re

import cv2
import numpy as np
import requests

from .paths import CACHE_DIR

CDN_URL = "https://cdn.strats.gg"
ASSET_DIR = CACHE_DIR / "assets"
VIEW_SIZE = 1024
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
ARGUMENT_COUNT = {"m": 2, "l": 2, "h": 1, "v": 1, "s": 4, "c": 6, "q": 4, "t": 2, "z": 0}


def _svg_text(map_item, side):
    key = map_item["attackerMapImageKey"] if side == "attack" else map_item["defenderMapImageKey"]
    path = ASSET_DIR / f"{map_item['id']}_{side}.svg"
    if not path.exists():
        response = requests.get(f"{CDN_URL}/{key}", timeout=20)
        response.raise_for_status()
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path.read_text(encoding="utf-8")


def _path_polygons(data):
    """Polygons for an SVG path, with curves flattened to their end points."""
    polygons, current = [], []
    x = y = 0.0
    for command, arguments in re.findall(r"([a-zA-Z])([^a-zA-Z]*)", data):
        values = [float(number) for number in NUMBER.findall(arguments)]
        kind, relative = command.lower(), command.islower()
        if kind == "z":
            if current:
                polygons.append(current)
                x, y = current[0]
            current = []
            continue
        size = ARGUMENT_COUNT[kind]
        for start in range(0, len(values), size):
            chunk = values[start:start + size]
            if kind == "h":
                x = x + chunk[0] if relative else chunk[0]
            elif kind == "v":
                y = y + chunk[0] if relative else chunk[0]
            else:
                end_x, end_y = chunk[-2], chunk[-1]
                x, y = (x + end_x, y + end_y) if relative else (end_x, end_y)
            if kind == "m" and start == 0:
                if current:
                    polygons.append(current)
                current = []
            current.append((x, y))
    if current:
        polygons.append(current)
    return polygons


def silhouette(map_item, side):
    """1024x1024 uint8 mask (255 inside the walkable map) for this map and side."""
    cache = ASSET_DIR / f"{map_item['id']}_{side}_mask.png"
    if cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
    body = re.search(r'<g id="body">(.*?)</g>', _svg_text(map_item, side), re.S).group(1)
    mask = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.uint8)
    for data in re.findall(r'\sd="([^"]+)"', body):
        polygons = [np.round(np.array(polygon)).astype(np.int32) for polygon in _path_polygons(data) if len(polygon) >= 3]
        cv2.fillPoly(mask, polygons, 255)
    cv2.imwrite(str(cache), mask)
    return mask
