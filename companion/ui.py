"""Small control window: pick the match (or leave it on Auto), start watching, see what is detected."""

import base64
import queue
import threading
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import ttk

import cv2

from . import __version__, strats_api, updater, win
from .paths import CAPTURE_DIR
from .driver import WINDOW_TITLE as STRATS_TITLE
from .watcher import PREVIEW_SIZE, Watcher

AUTO = "Auto"
BACKGROUND = "#16181d"
PANEL = "#1f2229"
TEXT = "#e6e8ec"
MUTED = "#8b919c"
ACCENT = "#ff4655"
STRATS_TOPMOST_REFRESH_MS = 3000
STATE_ROWS = (("match", "Match"), ("minimap", "Minimap"), ("spike", "Spike"), ("player", "You"), ("lineup", "Lineup"))


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = self.watcher = self.photo = None

        root.title(f"Strats Companion {__version__}")
        self.update = None
        root.configure(bg=BACKGROUND)
        root.resizable(False, False)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BACKGROUND, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT)
        style.configure("TButton", background=PANEL, foreground=TEXT, padding=6)
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", padding=6)
        style.map("Accent.TButton", background=[("active", "#ff6b77")])
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
        self.toggle = ttk.Button(controls, text="Start", style="Accent.TButton", command=self.toggle_watching, width=10)
        self.toggle.grid(row=1, column=4, padx=(14, 0))

        status = ttk.Frame(root, padding=(14, 6))
        status.grid(row=1, column=0, sticky="nw")
        self.state = {}
        for row, (key, label) in enumerate(STATE_ROWS):
            ttk.Label(status, text=label, style="Muted.TLabel", width=9).grid(row=row, column=0, sticky="w", pady=3)
            self.state[key] = tk.StringVar(value="")
            ttk.Label(status, textvariable=self.state[key], width=34, wraplength=250).grid(row=row, column=1, sticky="w", pady=3)
        buttons = ttk.Frame(status)
        buttons.grid(row=len(STATE_ROWS), column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Button(buttons, text="Recalibrate minimap", command=self.recalibrate).grid(row=0, column=0)
        ttk.Button(buttons, text="Save screenshot", command=self.save_screenshot).grid(row=0, column=1, padx=(8, 0))
        self.update_button = ttk.Button(status, text="", style="Accent.TButton", command=self.install_update)
        self.update_button.grid(row=len(STATE_ROWS) + 1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.update_button.grid_remove()

        self.preview = tk.Label(root, bg=PANEL, width=PREVIEW_SIZE, height=PREVIEW_SIZE, bd=0)
        self.blank = tk.PhotoImage(width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        self.preview.configure(image=self.blank)
        self.preview.grid(row=1, column=1, padx=(0, 14), pady=6)

        self.log = tk.Text(root, height=8, width=92, bg=PANEL, fg=TEXT, bd=0, padx=8, pady=6, state="disabled", wrap="word")
        self.log.grid(row=2, column=0, columnspan=2, padx=14, pady=(6, 8))

        root.protocol("WM_DELETE_WINDOW", self.close)
        updater.remove_previous()
        threading.Thread(target=self._check_update, daemon=True).start()
        root.after(100, self.drain)
        root.after(STRATS_TOPMOST_REFRESH_MS, self._keep_strats_pinned)

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

    def _check_update(self):
        try:
            self.events.put(("update", updater.check()))
        except updater.UpdateError as error:
            self.events.put(("log", f"Update check failed: {error}"))

    def install_update(self):
        if not updater.can_install():
            self.write_log("Running from source: update with git pull instead.")
            return
        self.stop.set()
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

    def apply_pins(self):
        self.root.attributes("-topmost", self.pin_self.get())
        for hwnd, _ in win.find_windows(STRATS_TITLE):
            win.set_topmost(hwnd, self.pin_strats.get())

    def _keep_strats_pinned(self):
        # Strats.gg gets a new window whenever it is restarted, so re-apply the pin.
        if self.pin_strats.get():
            for hwnd, _ in win.find_windows(STRATS_TITLE):
                win.set_topmost(hwnd, True)
        self.root.after(STRATS_TOPMOST_REFRESH_MS, self._keep_strats_pinned)

    # ---- display ------------------------------------------------------------

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain(self):
        latest_preview = None
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
                elif kind == "update":
                    self.update = payload
                    if payload is None:
                        self.write_log(f"Version {__version__} is up to date.")
                    else:
                        self.write_log(f"Version {payload['version']} is available.")
                        self.update_button.configure(text=f"Update to {payload['version']} and restart")
                        self.update_button.grid()
                elif kind == "progress":
                    if payload is None:
                        self.update_button.configure(state="normal", text=f"Update to {self.update['version']} and restart")
                    else:
                        self.update_button.configure(text=f"Downloading {payload:.0%}")
                elif kind == "restart":
                    updater.restart()
                elif kind == "stopped":
                    self.toggle.configure(text="Start", state="normal")
                    self.write_log("Stopped.")
        except queue.Empty:
            pass
        if latest_preview is not None:
            encoded = cv2.imencode(".png", latest_preview)[1].tobytes()
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded))
            self.preview.configure(image=self.photo, width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        self.root.after(100, self.drain)

    def close(self):
        self.stop.set()
        for hwnd, _ in win.find_windows(STRATS_TITLE):
            win.set_topmost(hwnd, False)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
