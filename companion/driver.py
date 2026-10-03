"""Drives the Strats.gg desktop window's lineup tool.

Clicks are posted to the window as messages, so the real cursor never moves
and the window does not need focus. Positions below are for the fixed-size
Strats.gg main window (verified on app version 2026.10.2); every step is
checked against a capture of the window before the next one is taken.
"""

import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

from . import geometry, win

WINDOW_TITLE = "Strats.gg"
ASSETS = Path(__file__).resolve().parent / "assets"

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001
WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x0100, 0x0101, 0x0102
VK_ESCAPE = 0x1B
SW_SHOWNOACTIVATE = 4
SW_SHOWMINNOACTIVE = 7
MIN_WINDOW_WIDTH = 1000        # the Strats.gg window is a fixed 1748 px wide once restored

NAV_LINEUPS = (624, 73)
VIDEO = (660, 450)            # middle of the lineup page's video
DETAILS = (1500, 470)         # the lineup's description, beside the video: plain text, safe to release a click on
VIDEO_AREA = (50, 150, 1290, 800)
BACK_BUTTON = (81, 128)
CHANGE_MAP = (262, 313)
CHANGE_AGENT = (255, 465)
SIDE_BUTTONS = {"attack": (97, 373), "defense": (251, 373)}
SIDE_SWATCHES = {"attack": (35, 368, 55, 378), "defense": (190, 368, 210, 378)}
MAP_GRID = {"origin": (311.5, 326.0), "pitch": (223.0, 129.0), "columns": 4}
AGENT_GRID = {"origin": (369.0, 298.0), "pitch": (110.75, 110.75), "columns": 6}
# Where each reference image is expected, as (left, top, right, bottom) with slack.
TEMPLATE_REGIONS = {
    "select_map": (187, 223, 320, 264),
    "select_agent": (301, 204, 446, 245),
    "back": (42, 108, 120, 149),
}
TEMPLATE_THRESHOLD = 0.8
DEFAULT_MAP_RECT = (377.6, 106.9, 851.7)
MARKER_RADIUS = 14
COLOR_TOLERANCE = 60
MIN_MARKER_PIXELS = 15
# On the right page these measure about 1.0 and 0.9; on a wrong one at most 0.45 and 0.2.
MIN_COVERED = 0.75
MIN_EXPLAINED = 0.6

MAP_DIALOG, AGENT_DIALOG, LINEUP_PAGE, MAP_VIEW, OTHER = "map-dialog", "agent-dialog", "lineup", "map-view", "other"


STRATS_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Strats.gg" / "Strats.gg.exe"
# Chromium suspends a window that is fully covered by other windows and then
# ignores posted clicks; these switches keep Strats.gg responsive when covered.
KEEP_ACTIVE_FLAGS = [
    "--disable-features=CalculateNativeWinOcclusion",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


class DriverError(RuntimeError):
    pass


def _running_command_line():
    """Command line of the main Strats.gg process, or None when it is not running."""
    query = (
        "Get-CimInstance Win32_Process -Filter \"Name='Strats.gg.exe'\" | "
        "Where-Object { $_.CommandLine -notmatch '--type=' } | "
        "Select-Object -First 1 -ExpandProperty CommandLine"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)
    return result.stdout.strip() or None


def launch(restart=False):
    """Start Strats.gg with the keep-active switches.

    Returns 'running' if it already has them, 'started' if it was launched, or
    'needs-restart' if it is running without them and `restart` is False.
    """
    command_line = _running_command_line()
    if command_line is not None:
        if KEEP_ACTIVE_FLAGS[0] in command_line:
            return "running"
        if not restart:
            return "needs-restart"
        subprocess.run(["taskkill", "/F", "/IM", "Strats.gg.exe"], capture_output=True)
        time.sleep(3)
    # An inherited ELECTRON_RUN_AS_NODE (set by editors such as VS Code) makes the app exit at once.
    environment = {key: value for key, value in os.environ.items() if key != "ELECTRON_RUN_AS_NODE"}
    subprocess.Popen(
        [str(STRATS_EXE), *KEEP_ACTIVE_FLAGS], env=environment,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        time.sleep(1)
        if win.find_windows(WINDOW_TITLE):
            time.sleep(4)
            return "started"
    raise DriverError("Strats.gg did not open a window after being started.")


class StratsWindow:
    def __init__(self):
        self.hwnd = None
        self.offset = (0, 0)
        self.templates = {name: cv2.imread(str(ASSETS / f"{name}.png")) for name in TEMPLATE_REGIONS}
        self.map_rect = DEFAULT_MAP_RECT
        self.shown = None
        self.fullscreen = False

    def attach(self):
        windows = win.find_windows(WINDOW_TITLE)
        if not windows:
            raise DriverError("The Strats.gg window is not open. Open it and keep it visible (not minimised).")
        self.hwnd, rect = windows[0]
        if win.user32.IsIconic(self.hwnd):
            # A minimised window cannot be captured or clicked; bring it back without giving it
            # focus, then wait until it is full size and has drawn its page again.
            deadline = time.time() + 10
            while time.time() < deadline:
                if win.user32.IsIconic(self.hwnd):
                    win.user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
                time.sleep(0.5)
                _, rect = win.find_windows(WINDOW_TITLE)[0]
                if rect[2] - rect[0] >= MIN_WINDOW_WIDTH and self.capture().mean() > 5:
                    time.sleep(0.5)
                    break
            else:
                raise DriverError("The Strats.gg window is minimised and did not come back. Restore it by hand.")
        origin = wintypes.POINT(0, 0)
        win.user32.ClientToScreen(self.hwnd, ctypes.byref(origin))
        self.offset = (origin.x - rect[0], origin.y - rect[1])

    def capture(self):
        return win.capture_window(self.hwnd)

    def _post(self, message, flags, x, y):
        client_x, client_y = int(round(x)) - self.offset[0], int(round(y)) - self.offset[1]
        win.user32.PostMessageW(self.hwnd, message, flags, (client_y << 16) | (client_x & 0xFFFF))

    def key(self, virtual_key, character=None):
        """Press a key in the page without focusing the window."""
        win.user32.PostMessageW(self.hwnd, WM_KEYDOWN, virtual_key, 1)
        if character is not None:
            win.user32.PostMessageW(self.hwnd, WM_CHAR, ord(character), 1)
        time.sleep(0.05)
        win.user32.PostMessageW(self.hwnd, WM_KEYUP, virtual_key, 0xC0000001)

    def minimize(self):
        """Minimise Strats.gg without taking focus; attach() brings it back when it is needed."""
        if self.hwnd is not None and not win.user32.IsIconic(self.hwnd):
            self.leave_fullscreen()
            win.user32.ShowWindow(self.hwnd, SW_SHOWMINNOACTIVE)

    def _video_playing(self):
        left, top, right, bottom = VIDEO_AREA
        first = self.capture()[top:bottom, left:right].astype(np.int16)
        time.sleep(0.5)
        second = self.capture()[top:bottom, left:right].astype(np.int16)
        return float(np.abs(first - second).mean()) > 1.0

    def fullscreen_video(self):
        """Make the open lineup's video fill the Strats.gg window, still playing.

        The player toggles full screen with F once it has keyboard focus. Pressing
        the mouse on the video gives it focus; releasing over the description
        instead of the video keeps that from counting as a click, which would pause it.
        """
        self.move(*VIDEO)
        time.sleep(0.05)
        self._post(WM_LBUTTONDOWN, MK_LBUTTON, *VIDEO)
        time.sleep(0.05)
        self._post(WM_MOUSEMOVE, MK_LBUTTON, *DETAILS)
        time.sleep(0.05)
        self._post(WM_LBUTTONUP, 0, *DETAILS)
        time.sleep(0.2)
        self.key(ord("F"), "f")
        time.sleep(0.8)
        self.fullscreen = self.state(self.capture()) != LINEUP_PAGE
        if not self._video_playing():
            self.key(ord("K"), "k")

    def leave_fullscreen(self):
        if self.fullscreen:
            self.key(VK_ESCAPE)
            self.fullscreen = False
            time.sleep(0.6)

    def move(self, x, y):
        self._post(WM_MOUSEMOVE, 0, x, y)

    def click(self, x, y):
        self.move(x, y)
        time.sleep(0.08)
        self._post(WM_LBUTTONDOWN, MK_LBUTTON, x, y)
        time.sleep(0.05)
        self._post(WM_LBUTTONUP, 0, x, y)

    # ---- screen recognition -------------------------------------------------

    def _has(self, image, name):
        left, top, right, bottom = TEMPLATE_REGIONS[name]
        region = image[top:bottom, left:right]
        template = self.templates[name]
        if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
            return False
        return cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED).max() >= TEMPLATE_THRESHOLD

    def _selected_side(self, image):
        for side, (left, top, right, bottom) in SIDE_SWATCHES.items():
            blue, green, red = image[top:bottom, left:right].reshape(-1, 3).mean(axis=0)
            if red > 70 and red > 1.6 * max(blue, green):
                return side
        return None

    def state(self, image=None):
        image = self.capture() if image is None else image
        if self._has(image, "select_map"):
            return MAP_DIALOG
        if self._has(image, "select_agent"):
            return AGENT_DIALOG
        if self._has(image, "back"):
            return LINEUP_PAGE
        if self._selected_side(image):
            return MAP_VIEW
        return OTHER

    def _wait_for_change(self, previous, timeout=4.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.25)
            current = self.state()
            if current != previous:
                time.sleep(0.6)
                return self.state()
        return previous

    # ---- page check --------------------------------------------------------

    def shows(self, image, groups, color):
        """Whether the lineup map on screen is the one these marker groups belong to.

        Two tests, both on pixels in the agent's marker colour: nearly every
        expected marker has colour around it, and nearly all of that colour sits
        on an expected marker. A different map, agent or side fails at least one.
        """
        if not groups:
            return False
        target = np.array([int(color[i:i + 2], 16) for i in (5, 3, 1)], dtype=np.int16)
        mask = np.abs(image.astype(np.int16) - target).max(axis=2) < COLOR_TOLERANCE
        mask[:, :330] = False
        ys, xs = np.nonzero(mask)
        if len(xs) < MIN_MARKER_PIXELS:
            return False
        pixels = np.stack([xs, ys], axis=1).astype(np.float32)
        points = np.array([self.to_window(group["point"]) for group in groups], dtype=np.float32)
        distances = np.linalg.norm(points[:, None, :] - pixels[None, :, :], axis=2)
        covered = ((distances < MARKER_RADIUS + 2).sum(axis=1) >= MIN_MARKER_PIXELS).mean()
        explained = (distances.min(axis=0) < MARKER_RADIUS + 6).mean()
        return covered >= MIN_COVERED and explained >= MIN_EXPLAINED

    def to_window(self, point):
        left, top, size = self.map_rect
        return left + point[0] * size / 100, top + point[1] * size / 100

    # ---- navigation ---------------------------------------------------------

    @staticmethod
    def _grid_cell(grid, index):
        row, column = divmod(index, grid["columns"])
        return grid["origin"][0] + column * grid["pitch"][0], grid["origin"][1] + row * grid["pitch"][1]

    def show_map(self, maps, agents, map_item, agent_item, side, groups):
        """Bring the window to the lineup map for this map, agent and side.

        Without `groups` (the lineup data is not known yet) the map and agent are
        picked without checking the markers afterwards.
        """
        self.attach()
        listed_maps = [item for item in maps if item.get("showInLineups", True)]
        listed_agents = [item for item in agents if item.get("showInLineups", True)]
        map_index = next(i for i, item in enumerate(listed_maps) if item["id"] == map_item["id"])
        agent_index = next(i for i, item in enumerate(listed_agents) if item["id"] == agent_item["id"])
        wanted = (map_item["id"], agent_item["id"])
        map_done = agent_done = self.shown == wanted
        retries = 0
        escaped = False

        for _ in range(12):
            image = self.capture()
            state = self.state(image)
            if state == LINEUP_PAGE:
                self.click(*BACK_BUTTON)
            elif state == MAP_DIALOG:
                self.click(*self._grid_cell(MAP_GRID, map_index))
                map_done = True
            elif state == AGENT_DIALOG:
                self.click(*self._grid_cell(AGENT_GRID, agent_index))
                agent_done = True
            elif state == OTHER and not escaped:
                # Possibly a full-screen video, perhaps left by an earlier run that no longer knows it
                # made it: Escape leaves full screen, and does no harm on any other page.
                self.fullscreen = True
                self.leave_fullscreen()
                escaped = True
                continue
            elif state == OTHER:
                self.click(*NAV_LINEUPS)
            else:
                # Park the pointer off the map so no marker is left in its dimmed hover state.
                self.move(*CHANGE_MAP)
                time.sleep(0.3)
                image = self.capture()
                if self._selected_side(image) != side:
                    self.click(*SIDE_BUTTONS[side])
                    time.sleep(1.2)
                    continue
                if groups is not None and self.shows(image, groups, agent_item["color"]):
                    self.shown = wanted
                    return
                if not map_done:
                    self.click(*CHANGE_MAP)
                elif not agent_done:
                    self.click(*CHANGE_AGENT)
                else:
                    # Map and agent were both just picked; give the markers a moment to load.
                    time.sleep(1.0)
                    if not groups or self.shows(self.capture(), groups, agent_item["color"]):
                        self.shown = wanted
                        return
                    retries += 1
                    if retries > 1:
                        break
                    map_done = agent_done = False
                    self.shown = None
                    continue
            self._wait_for_change(state)
        self.shown = None
        raise DriverError(
            "Could not get Strats.gg to the requested lineup map. Check that its map is not zoomed or dragged "
            "and that the lineup filters are on All."
        )

    def _click_point_for(self, groups, group):
        """A point inside this group's marker that no later-drawn marker covers."""
        center = np.array(self.to_window(group["point"]))
        later = [np.array(self.to_window(other["point"])) for other in groups if other["index"] > group["index"]]
        offsets = sorted(
            ((dx, dy) for dx in range(-10, 11, 2) for dy in range(-10, 11, 2) if dx * dx + dy * dy <= 100),
            key=lambda offset: offset[0] ** 2 + offset[1] ** 2,
        )
        for offset in offsets:
            candidate = center + offset
            if all(np.linalg.norm(candidate - other) > MARKER_RADIUS + 1 for other in later):
                return candidate
        return None

    def open_lineup(self, groups, group, lineup):
        """From the lineup map, open one lineup's page. Returns False if its marker is fully covered."""
        point = self._click_point_for(groups, group)
        if point is None:
            return False
        self.click(*point)
        if len(group["lineups"]) > 1:
            time.sleep(0.7)
            self.click(*self.to_window((lineup["left"], lineup["top"])))
        return self._wait_for_change(MAP_VIEW) == LINEUP_PAGE
