"""Transparent drawing over the game: where to stand (on the minimap) and where to aim.

The window covers the primary monitor, ignores the mouse and never takes focus,
so the game keeps all input. It is left out of screen captures, so the
companion's own minimap reads never see what it draws. Valorant has to run in
Windowed Fullscreen for any window to show over it.
"""

import base64
import ctypes
import tkinter as tk

import cv2

from . import win

KEY = "#010203"                 # drawn in this colour = see-through
STAND = "#ffd23f"
AIM = "#3dfc8e"
TEXT = "#ffffff"
SHADOW = "#000000"
GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x00080000, 0x00000020, 0x00000080, 0x08000000
WDA_EXCLUDEFROMCAPTURE = 0x11
EDGE_MARGIN = 80
PANEL_WIDTH_FRACTION = 0.2


class Overlay:
    def __init__(self, root):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.configure(bg=KEY)
        self.window.attributes("-transparentcolor", KEY)
        self.window.attributes("-topmost", True)
        user32 = ctypes.windll.user32
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.size = (width, height)
        self.window.geometry(f"{width}x{height}+0+0")
        self.canvas = tk.Canvas(self.window, width=width, height=height, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack()
        self.window.update_idletasks()
        handle = user32.GetParent(self.window.winfo_id()) or self.window.winfo_id()
        style = user32.GetWindowLongW(handle, GWL_EXSTYLE)
        user32.SetWindowLongW(handle, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        user32.SetWindowDisplayAffinity(handle, WDA_EXCLUDEFROMCAPTURE)
        self.handle = handle
        self.photo = None
        self.visible = False
        self.static_key = self.aim_mode = self.aim_key = None

    def _text(self, x, y, text, color=TEXT, size=16, anchor="center", tag="static"):
        font = ("Segoe UI", size, "bold")
        self.canvas.create_text(x + 2, y + 2, text=text, fill=SHADOW, font=font, anchor=anchor, tags=tag)
        self.canvas.create_text(x, y, text=text, fill=color, font=font, anchor=anchor, tags=tag)

    def show(self, guide):
        """Draw a guide (see Watcher._show_guide), or hide everything for None.

        The minimap part is redrawn only when it changes, and the reticle is moved
        rather than redrawn, so the aim point can follow the camera smoothly.
        """
        if guide is None:
            self.canvas.delete("all")
            self.static_key = self.aim_mode = self.aim_key = None
            if self.visible:
                self.window.withdraw()
                self.visible = False
            return
        scale = self.size[1] / 1440
        self._draw_static(guide, scale)
        self._draw_aim(guide, scale)
        if not self.visible:
            self.window.deiconify()
            self.window.attributes("-topmost", True)
            self.visible = True

    def _draw_static(self, guide, scale):
        """Where to stand, on the minimap: a ring on the spot, a dot where it lands, a line between."""
        stand, land = guide.get("stand"), guide.get("land")
        label = "In position" if guide.get("in_position") else (
            f"{guide['metres']:.0f} m" if guide.get("metres") is not None else "Stand here")
        key = (stand and tuple(round(v) for v in stand), land and tuple(round(v) for v in land), label)
        if key == self.static_key:
            return
        self.static_key = key
        self.canvas.delete("static")
        if stand is not None and land is not None:
            self.canvas.create_line(*stand, *land, fill=STAND, width=max(2, int(3 * scale)), dash=(6, 4), tags="static")
            self.canvas.create_oval(land[0] - 5, land[1] - 5, land[0] + 5, land[1] + 5, fill=STAND, outline="", tags="static")
        if stand is not None:
            radius = 11 * scale
            self.canvas.create_oval(stand[0] - radius, stand[1] - radius, stand[0] + radius, stand[1] + radius,
                                    outline=STAND, width=max(2, int(3 * scale)), tags="static")
            self._text(stand[0], stand[1] - radius - 14 * scale, label, STAND, int(13 * scale))

    def _draw_aim(self, guide, scale):
        """Where to aim: a reticle on the spot, an arrow at the edge pointing the way to turn, or the screenshot."""
        aim = guide.get("aim")
        if aim is not None and aim.on_screen:
            x, y = aim.point
            if self.aim_mode != "reticle":
                self.canvas.delete("aim")
                r = 26 * scale
                thickness = max(2, int(3 * scale))
                self.canvas.create_oval(-r, -r, r, r, outline=AIM, width=thickness, tags="aim")
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    self.canvas.create_line(dx * r * 0.45, dy * r * 0.45, dx * r * 1.5, dy * r * 1.5,
                                            fill=AIM, width=thickness, tags="aim")
                self._text(0, r + 24 * scale, "Aim here", AIM, int(17 * scale), tag="aim")
                self.aim_mode, self.aim_key = "reticle", (0.0, 0.0)
            self.canvas.move("aim", x - self.aim_key[0], y - self.aim_key[1])
            self.aim_key = (x, y)
        elif aim is not None:
            key = (round(aim.yaw), round(aim.pitch))
            if self.aim_mode != "arrow" or key != self.aim_key:
                self.canvas.delete("aim")
                self._arrow(aim, scale)
                self.aim_mode, self.aim_key = "arrow", key
        elif guide.get("in_position") and guide.get("picture") is not None:
            # The scenery could not be matched: show the aim screenshot, as the Strats.gg overlay does.
            if self.aim_mode != "panel" or self.aim_key != guide.get("title"):
                self.canvas.delete("aim")
                self._panel(guide["picture"], guide.get("title", ""))
                self.aim_mode, self.aim_key = "panel", guide.get("title")
        elif self.aim_mode is not None:
            self.canvas.delete("aim")
            self.aim_mode = self.aim_key = None

    def _arrow(self, aim, scale):
        width, height = self.size
        cx, cy = width / 2, height / 2
        dx, dy = aim.yaw, -aim.pitch
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / length, dy / length
        # Walk from the middle towards the aim point until the arrow sits just inside the screen edge.
        reach = min(
            (cx - EDGE_MARGIN) / abs(ux) if ux else float("inf"),
            (cy - EDGE_MARGIN) / abs(uy) if uy else float("inf"),
        )
        tip = (cx + ux * reach, cy + uy * reach)
        tail = (tip[0] - ux * 120 * scale, tip[1] - uy * 120 * scale)
        self.canvas.create_line(*tail, *tip, fill=AIM, width=max(5, int(12 * scale)), arrow="last",
                                arrowshape=(30 * scale, 36 * scale, 14 * scale), tags="aim")
        turn = []
        if abs(aim.yaw) >= 1:
            turn.append(f"{abs(aim.yaw):.0f}° {'right' if aim.yaw > 0 else 'left'}")
        if abs(aim.pitch) >= 1:
            turn.append(f"{abs(aim.pitch):.0f}° {'up' if aim.pitch > 0 else 'down'}")
        self._text(tail[0] - ux * 60 * scale, tail[1] - uy * 60 * scale, "Turn " + ", ".join(turn), AIM,
                   int(20 * scale), tag="aim")

    def _panel(self, picture, title):
        width, height = self.size
        panel_width = int(width * PANEL_WIDTH_FRACTION)
        panel_height = int(panel_width * picture.shape[0] / picture.shape[1])
        resized = cv2.resize(picture, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        self.photo = tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", resized)[1].tobytes()))
        x, y = width - panel_width - 30, (height - panel_height) // 2
        self.canvas.create_image(x, y, image=self.photo, anchor="nw", tags="aim")
        self.canvas.create_rectangle(x, y, x + panel_width, y + panel_height, outline=AIM, width=2, tags="aim")
        self._text(x, y - 18, title or "Aim like this", TEXT, 13, anchor="w", tag="aim")

    def close(self):
        self.window.destroy()


def excluded_from_capture(handle):
    """Whether Windows keeps this window out of screen captures (Windows 10 2004 or later)."""
    affinity = ctypes.c_uint(0)
    win.user32.GetWindowDisplayAffinity(handle, ctypes.byref(affinity))
    return affinity.value == WDA_EXCLUDEFROMCAPTURE
