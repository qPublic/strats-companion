"""Small control window: pick the match (or leave it on Auto), start watching, see what is detected."""

import base64
import json
import queue
import threading
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import font as tkfont
from tkinter import ttk

import cv2

from . import __version__, broken, prefetch, strats_api, updater, win
from .paths import CAPTURE_DIR, DATA_DIR
from .driver import WINDOW_TITLE as STRATS_TITLE
from .overlay import Overlay
from .watcher import PREVIEW_SIZE, Watcher

AUTO = "Auto"
BACKGROUND = "#16181d"
PANEL = "#1f2229"
TEXT = "#e6e8ec"
MUTED = "#8b919c"
ACCENT = "#ff4655"
STRATS_TOPMOST_REFRESH_MS = 1000
MIN_WIDTH, MIN_HEIGHT = 760, 520
MIN_PREVIEW = 160
UPDATE_CHECK_MS = 10 * 60 * 1000      # look for a new release this often while the window is open
SETTINGS_FILE = DATA_DIR / "settings.json"
SCALE_LIMITS = (0.85, 2.5)            # how far text and controls shrink or grow with the window
SCALED_FONTS = ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkHeadingFont")
WRAP_LENGTH = 250
STATE_ROWS = (("match", "Match"), ("minimap", "Minimap"), ("spike", "Spike"), ("player", "You"), ("lineup", "Lineup"))


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = self.watcher = self.photo = None

        root.title(f"Strats Companion {__version__}")
        self.update = None
        self.update_checked = False
        root.configure(bg=BACKGROUND)
        style = self.style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BACKGROUND, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT)
        style.configure("TButton", background=PANEL, foreground=TEXT, padding=6)
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", padding=6)
        style.map("Accent.TButton", background=[("active", "#ff6b77")])
        style.map("TButton", background=[("disabled", BACKGROUND)], foreground=[("disabled", MUTED)])
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT, arrowcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])

        try:
            maps = [item["name"] for item in strats_api.maps() if item.get("showInLineups", True)]
            agents = [item["name"] for item in strats_api.agents() if item.get("showInLineups", True)]
        except Exception as error:
            maps, agents = [], []
            self.events.put(("log", f"Could not load maps and agents from Strats.gg: {error}"))

        controls = ttk.Frame(root, padding=(14, 14, 14, 6))
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.map_choice = self._combo(controls, "Map", [AUTO, *maps], 0)
        self.agent_choice = self._combo(controls, "Agent", [AUTO, *agents], 1)
        self.side_choice = self._combo(controls, "Side", [AUTO, "attack", "defense"], 2)
        self.drive = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Open lineups in Strats.gg", variable=self.drive).grid(row=1, column=3, padx=(14, 0))
        pins = ttk.Frame(root, padding=(14, 0, 14, 4))
        pins.grid(row=3, column=0, columnspan=2, sticky="w")
        self.pin_self = tk.BooleanVar(value=False)
        self.pin_strats = tk.BooleanVar(value=False)
        ttk.Checkbutton(pins, text="Keep this window on top", variable=self.pin_self, command=self.apply_pins).grid(row=0, column=0)
        ttk.Checkbutton(pins, text="Keep Strats.gg on top", variable=self.pin_strats, command=self.apply_pins).grid(row=0, column=1, padx=(18, 0))
        self.fullscreen = tk.BooleanVar(value=True)
        self.hide_strats = tk.BooleanVar(value=True)
        ttk.Checkbutton(pins, text="Full-screen the lineup video", variable=self.fullscreen, command=self.apply_options).grid(row=0, column=2, padx=(18, 0))
        ttk.Checkbutton(pins, text="Minimise Strats.gg between lineups", variable=self.hide_strats, command=self.apply_options).grid(row=0, column=3, padx=(18, 0))
        self.show_guide = tk.BooleanVar(value=True)
        self.test_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(pins, text="Test mode: a dropped spike counts as planted", variable=self.test_mode, command=self.apply_options).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(pins, text="Show in-game guide", variable=self.show_guide, command=self.apply_options).grid(row=0, column=4, padx=(18, 0))
        self.overlay = Overlay(root)
        self.toggle = ttk.Button(controls, text="Start", style="Accent.TButton", command=self.toggle_watching, width=10)
        self.toggle.grid(row=1, column=4, padx=(14, 0))

        status = ttk.Frame(root, padding=(14, 6))
        status.grid(row=1, column=0, sticky="nw")
        self.state = {}
        self.wrapped = []
        for row, (key, label) in enumerate(STATE_ROWS):
            ttk.Label(status, text=label, style="Muted.TLabel", width=9).grid(row=row, column=0, sticky="w", pady=3)
            self.state[key] = tk.StringVar(value="")
            value = ttk.Label(status, textvariable=self.state[key], width=34, wraplength=WRAP_LENGTH)
            value.grid(row=row, column=1, sticky="w", pady=3)
            self.wrapped.append(value)
        buttons = ttk.Frame(status)
        buttons.grid(row=len(STATE_ROWS), column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Button(buttons, text="Recalibrate minimap", command=self.recalibrate).grid(row=0, column=0)
        ttk.Button(buttons, text="Save screenshot", command=self.save_screenshot).grid(row=0, column=1, padx=(8, 0))
        self.check_button = ttk.Button(buttons, text="Check for updates", command=self.check_for_updates)
        self.check_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Button(buttons, text="Broken lineup", command=self.mark_broken).grid(row=1, column=0, pady=(8, 0), sticky="ew")
        self.next_button = ttk.Button(buttons, text="Next lineup", command=self.next_lineup)
        self.next_button.grid(row=1, column=1, padx=(8, 0), pady=(8, 0), sticky="ew")
        self.clear_button = ttk.Button(buttons, text="", command=self.clear_broken)
        self.clear_button.grid(row=1, column=2, padx=(8, 0), pady=(8, 0), sticky="ew")
        self._refresh_lineup_buttons()
        self.update_button = ttk.Button(status, text="", style="Accent.TButton", command=self.install_update)
        self.update_button.grid(row=len(STATE_ROWS) + 1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.update_button.grid_remove()

        self.preview = tk.Label(root, bg=PANEL, width=PREVIEW_SIZE, height=PREVIEW_SIZE, bd=0)
        self.blank = tk.PhotoImage(width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        self.preview.configure(image=self.blank)
        self.preview.grid(row=1, column=1, padx=(0, 14), pady=6, sticky="nsew")

        self.log = tk.Text(root, height=8, width=92, bg=PANEL, fg=TEXT, bd=0, padx=8, pady=6, state="disabled", wrap="word")
        self.log.grid(row=2, column=0, columnspan=2, padx=14, pady=(6, 8), sticky="nsew")
        # The window can be resized: the preview and the log take up the extra room.
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=3)
        root.rowconfigure(2, weight=1)
        root.minsize(MIN_WIDTH, MIN_HEIGHT)
        # Fix the opening size, so the window keeps whatever size it has when the preview changes.
        root.update_idletasks()
        self.base_size = (root.winfo_reqwidth(), root.winfo_reqheight())
        self.fonts = {name: tkfont.nametofont(name) for name in SCALED_FONTS}
        self.font_sizes = {name: font.cget("size") for name, font in self.fonts.items()}
        self.scale = 1.0
        self.rescale_pending = None
        self.resume_after_update = False
        settings = self._load_settings()
        self._refresh_lineup_buttons()
        root.geometry(settings.get("geometry") or f"{self.base_size[0]}x{self.base_size[1]}")
        root.bind("<Configure>", self._on_resize)

        root.protocol("WM_DELETE_WINDOW", self.close)
        updater.remove_previous()
        self.check_for_updates(manual=False)
        if settings.get("resume"):
            # Restarted by an update while watching: carry on.
            root.after(500, self.toggle_watching)
        self.closing = threading.Event()
        threading.Thread(
            target=prefetch.run, args=(self.closing, lambda text: self.events.put(("log", text))), daemon=True
        ).start()
        root.after(100, self.drain)
        root.after(STRATS_TOPMOST_REFRESH_MS, self._keep_strats_pinned)

    # ---- settings and size ----------------------------------------------------

    def _settings_vars(self):
        return {
            "map": self.map_choice, "agent": self.agent_choice, "side": self.side_choice,
            "drive": self.drive, "pin_self": self.pin_self, "pin_strats": self.pin_strats,
            "fullscreen": self.fullscreen, "hide_strats": self.hide_strats, "show_guide": self.show_guide,
            "test_mode": self.test_mode,
        }

    def _load_settings(self):
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        for name, variable in self._settings_vars().items():
            if name in settings:
                variable.set(settings[name])
        if settings.get("pin_self") or settings.get("pin_strats"):
            self.root.after(300, self.apply_pins)
        if settings.get("resume"):
            self._save_settings(resume=False)
        return settings

    def _save_settings(self, resume=False):
        settings = {name: variable.get() for name, variable in self._settings_vars().items()}
        settings.update(geometry=self.root.geometry(), resume=resume)
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(settings, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _on_resize(self, event):
        if event.widget is self.root:
            if self.rescale_pending is not None:
                self.root.after_cancel(self.rescale_pending)
            self.rescale_pending = self.root.after(120, self._rescale)

    def _rescale(self):
        """Grow or shrink text, buttons and checkboxes with the window."""
        self.rescale_pending = None
        width, height = self.root.winfo_width(), self.root.winfo_height()
        scale = min(width / self.base_size[0], height / self.base_size[1])
        scale = round(min(max(scale, SCALE_LIMITS[0]), SCALE_LIMITS[1]), 2)
        if abs(scale - self.scale) < 0.04:
            return
        self.scale = scale
        for name, font in self.fonts.items():
            size = self.font_sizes[name]
            font.configure(size=int(round(size * scale)) or (1 if size > 0 else -1))
        padding = max(2, int(round(6 * scale)))
        self.style.configure("TButton", padding=padding)
        self.style.configure("Accent.TButton", padding=padding)
        self.style.configure("TCheckbutton", indicatorsize=max(8, int(round(11 * scale))))
        self.style.configure("TCombobox", arrowsize=max(10, int(round(13 * scale))), padding=max(1, int(round(2 * scale))))
        for label in self.wrapped:
            label.configure(wraplength=int(WRAP_LENGTH * scale))

    def _combo(self, parent, label, values, column):
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 10, 0))
        choice = ttk.Combobox(parent, values=values, state="readonly", width=13)
        choice.set(AUTO)
        choice.grid(row=1, column=column, padx=(0 if column == 0 else 10, 0))
        return choice

    # ---- worker -------------------------------------------------------------

    def toggle_watching(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop.set()
            self.toggle.configure(text="Stopping...", state="disabled")
            return
        map_name, agent_name, side = self.map_choice.get(), self.agent_choice.get(), self.side_choice.get()
        if (map_name == AUTO) != (agent_name == AUTO):
            self.write_log("Pick both a map and an agent, or leave both on Auto.")
            return
        self.stop = threading.Event()
        self.watcher = Watcher(
            map_name=None if map_name == AUTO else map_name,
            agent_name=None if agent_name == AUTO else agent_name,
            side=None if side == AUTO else side,
            drive=self.drive.get(),
            fullscreen=self.fullscreen.get(),
            hide=self.hide_strats.get(),
            on_guide=lambda guide: self.events.put(("guide", guide)),
            test_mode=self.test_mode.get(),
            on_log=lambda text: self.events.put(("log", text)),
            on_state=lambda **changes: self.events.put(("state", changes)),
            on_preview=lambda image: self.events.put(("preview", image)),
        )
        self.worker = threading.Thread(target=self._work, args=(self.watcher, self.stop), daemon=True)
        self.worker.start()
        self.toggle.configure(text="Stop")

    def _work(self, watcher, stop):
        try:
            watcher.run(stop)
        except Exception as error:
            self.events.put(("log", f"Stopped by an error: {error}"))
            self.events.put(("log", traceback.format_exc(limit=3)))
        self.events.put(("stopped", None))

    def _refresh_lineup_buttons(self):
        count = len(broken.ids())
        self.clear_button.configure(text=f"Clear broken ({count})", state="normal" if count else "disabled")
        # Trying lineup after lineup on one plant is for practice only.
        self.next_button.configure(state="normal" if self.test_mode.get() else "disabled")

    def _send(self, request):
        if self.watcher is None or self.worker is None or not self.worker.is_alive():
            self.write_log("Start watching first.")
            return False
        self.watcher.request = request
        return True

    def mark_broken(self):
        if self._send("broken"):
            self.root.after(1500, self._refresh_lineup_buttons)

    def next_lineup(self):
        self._send("next")

    def clear_broken(self):
        self.write_log(f"Cleared {broken.clear()} broken lineup marks.")
        self._refresh_lineup_buttons()

    def recalibrate(self):
        if self.watcher is not None:
            self.watcher.recalibrate = True
            self.write_log("Recalibrating the minimap zoom on the next frame.")

    def save_screenshot(self):
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAPTURE_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.png"
        cv2.imwrite(str(path), win.capture_screen())
        self.write_log(f"Saved {path}")

    # ---- updates ------------------------------------------------------------

    def check_for_updates(self, manual=True):
        if manual:
            self.check_button.configure(state="disabled", text="Checking...")
        threading.Thread(target=self._check_update, args=(manual,), daemon=True).start()
        if not manual:
            self.root.after(UPDATE_CHECK_MS, self.check_for_updates, False)

    def _check_update(self, manual):
        try:
            self.events.put(("update", (updater.check(), manual)))
        except updater.UpdateError as error:
            self.events.put(("update", (None, manual)))
            self.events.put(("log", f"Update check failed: {error}"))

    def install_update(self):
        if not updater.can_install():
            self.write_log("Running from source: update with git pull instead.")
            return
        self.resume_after_update = self.worker is not None and self.worker.is_alive()
        self.update_button.configure(state="disabled", text="Downloading 0%")
        threading.Thread(target=self._download_update, daemon=True).start()

    def _download_update(self):
        try:
            updater.install(self.update, lambda fraction: self.events.put(("progress", fraction)))
        except updater.UpdateError as error:
            self.events.put(("log", str(error)))
            self.events.put(("progress", None))
            return
        self.events.put(("restart", None))

    # ---- always on top -----------------------------------------------------

    def apply_options(self):
        if self.watcher is not None:
            self.watcher.fullscreen, self.watcher.hide = self.fullscreen.get(), self.hide_strats.get()
            self.watcher.test_mode = self.test_mode.get()
        self._refresh_lineup_buttons()
        if not self.show_guide.get():
            self.overlay.show(None)

    def apply_pins(self):
        self.root.attributes("-topmost", self.pin_self.get())
        for hwnd, _ in win.find_windows(STRATS_TITLE):
            win.set_topmost(hwnd, self.pin_strats.get())
        self._raise_self()

    def _raise_self(self):
        """With both pinned, keep this window above Strats.gg: the last window pinned goes on top."""
        if self.pin_self.get():
            win.set_topmost(int(self.root.wm_frame(), 16), True)

    def _keep_strats_pinned(self):
        # Strats.gg gets a new window whenever it is restarted, so re-apply the pin. Clicking
        # Strats.gg also brings it in front of this window, so put this one back on top.
        if self.pin_strats.get():
            for hwnd, _ in win.find_windows(STRATS_TITLE):
                win.set_topmost(hwnd, True)
            self._raise_self()
        self.root.after(STRATS_TOPMOST_REFRESH_MS, self._keep_strats_pinned)

    # ---- display ------------------------------------------------------------

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain(self):
        latest_preview = latest_guide = None
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.write_log(payload)
                elif kind == "state":
                    for key, value in payload.items():
                        self.state[key].set(value)
                elif kind == "preview":
                    latest_preview = payload
                elif kind == "guide":
                    latest_guide = (payload,)
                elif kind == "update":
                    found, manual = payload
                    self.check_button.configure(state="normal", text="Check for updates")
                    if found is None:
                        if manual or not self.update_checked:
                            self.write_log(f"Version {__version__} is up to date.")
                    elif self.update is None or found["version"] != self.update["version"]:
                        self.write_log(f"Version {found['version']} is available.")
                        self.update_button.configure(text=f"Install {found['version']} now")
                        self.update_button.grid()
                    if found is not None:
                        self.update = found
                    self.update_checked = True
                elif kind == "progress":
                    if payload is None:
                        self.update_button.configure(state="normal", text=f"Install {self.update['version']} now")
                    else:
                        self.update_button.configure(text=f"Downloading {payload:.0%}")
                elif kind == "restart":
                    # Relaunch on the new version with the same settings, still watching if it was.
                    self._save_settings(resume=self.resume_after_update)
                    updater.restart()
                elif kind == "stopped":
                    self.overlay.show(None)
                    self.toggle.configure(text="Start", state="normal")
                    self.write_log("Stopped.")
        except queue.Empty:
            pass
        if latest_guide is not None:
            self.overlay.show(latest_guide[0] if self.show_guide.get() else None)
        if latest_preview is not None:
            # Fit the preview to the room the window gives it.
            size = max(MIN_PREVIEW, min(self.preview.winfo_width(), self.preview.winfo_height()) - 4)
            scale = size / max(latest_preview.shape[:2])
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            fitted = cv2.resize(latest_preview, None, fx=scale, fy=scale, interpolation=interpolation)
            encoded = cv2.imencode(".png", fitted)[1].tobytes()
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded))
            self.preview.configure(image=self.photo, width=1, height=1)
        # Redraw quickly while the in-game guide is up, so the aim reticle follows the camera.
        self.root.after(15 if self.overlay.visible else 100, self.drain)

    def close(self):
        self._save_settings()
        self.stop.set()
        self.closing.set()
        for hwnd, _ in win.find_windows(STRATS_TITLE):
            win.set_topmost(hwnd, False)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
