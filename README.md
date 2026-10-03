# Strats.gg lineup companion

Picks the Strats.gg lineup that fits the current spike and player position and
opens it in the Strats.gg desktop window. Clicks are sent to that window in the
background, so the mouse cursor and the game's focus are never touched.

## Install

Download `StratsCompanion.exe` from the latest release and put it anywhere.
Double-click it to open the control window: leave Map, Agent and Side on Auto
(or pick them), press Start, and it restarts Strats.gg if needed. The window
shows what it detects, a minimap preview with the matched outline, and a log. Any command below also works as
`StratsCompanion.exe <command>`. Its data lives in
`%LOCALAPPDATA%\StratsCompanion`. The exe is unsigned, so Windows SmartScreen
will warn on first run.

## Always on top

Two checkboxes at the bottom of the window pin either the companion window or
the Strats.gg window above everything else, including the game in Windowed
Fullscreen. Keep a pinned window clear of the minimap (top left) and the round
timer (top centre): the companion reads those from the screen and cannot see
through a window covering them. Closing the companion releases Strats.gg.

## Updates

The window checks GitHub for a newer release each time it opens and shows an
"Update and restart" button when there is one. The old exe is kept as `StratsCompanion.old.exe` until the next start.

## Run from source

```
python -m venv .venv
.venv\Scripts\python -m pip install requests numpy opencv-python mss keyboard
```

Build the exe with `pyinstaller --onefile --console --name StratsCompanion
--add-data "companion/assets;companion/assets" strats_companion.py`.

## Commands

From source, run `.venv\Scripts\python -m companion <command>` in this folder.

| Command | What it does |
|---|---|
| `ui` | Opens the control window (the default when no command is given). |
| `launch [--restart]` | Starts Strats.gg with switches that keep it responsive while covered by other windows. `--restart` restarts it if it is already running without them. |
| `show --map ascent --agent viper --side attack --spike 20.8,27.4 --player 29.6,40.3` | Opens the best lineup for a spike and player position given by hand. Without `--spike` it only shows the lineup map. |
| `run [--map ascent --agent viper] [--side attack] [--dry-run] [--video file]` | Watches the minimap about once a second and opens the best post-plant lineup when the spike is planted. Map, agent and side come from the match unless given. `--video` analyses a recording instead of the screen; `--dry-run` only prints the choice. |
| `follow [--side attack]` | Keeps Strats.gg on the map, agent and side of the match in progress. |
| `update` | Installs the newest release of the exe, if there is one. |
| `capture [--key f9]` | Saves a screenshot of the primary monitor on each key press, for tuning minimap detection. |

Positions are `left,top` in percent of the Strats.gg map image for that side.

## How a lineup is chosen

A lineup qualifies when it lands within 3.5% of the map (about 5 m) of the
spike. Lineups using a post-plant ability (mollies, shock darts and similar,
listed in `companion/selector.py`) win over others. Among those, the one
thrown from closest to your team's spawn is opened, as long as its standing
spot is within 25% of the map (about 35 m in a straight line) of you. If none
is that close, the nearest standing spot is opened instead. The 25% limit is
`MAX_WALK` in `companion/selector.py`.

The pick can change while you move. Once you are within 25 m of the opened
lineup's standing spot it is locked in for the rest of the round and nothing
else is opened until the spike is gone (`LOCK_METRES` in
`companion/watcher.py`).

## Status

- Working and tested: lineup data, lineup choice, driving the Strats.gg window.
- Working on recorded footage (one 1080p Ascent match from YouTube, rotating
  minimap) and on live 1440p frames (Ascent, fixed minimap): detecting a
  planted spike, its position and the player's position.
- Not yet confirmed on a live game: a full plant-to-lineup run, and any map
  other than Ascent.

## How your side is decided

With Side on Auto, the Riot client's team name gives a first guess, which can
be wrong (it was in a custom game). A fixed (non-rotating) minimap then
corrects it: it is drawn with your own spawn at the bottom, so if it keeps
matching the map art upside down, the side is flipped. No lineup is picked
while that check is pending.

Once known, the side follows the round count: teams swap after 12 rounds
(4 in Swiftplay) and every round in overtime.

With a rotating minimap, or with Fixed Orientation set to "Always the Same",
the minimap cannot show the side; pick Side by hand in the window if the first
guess is wrong. Picking a side by hand turns the automatic logic off.

## How the minimap is read

The red spike indicator that replaces the round timer says the spike is
planted. The minimap is matched against the map outline from Strats.gg's own
map art at any rotation, which converts screen positions to lineup
coordinates. The spike is the yellow icon on the map; the player is the icon
with a white ring (teammates have teal rings). The minimap zoom is found once
per screen resolution and stored in `calibration.json`; press Recalibrate in
the window (or delete that file) after changing minimap size or zoom in
Valorant. Default settings calibrate in a couple of seconds; an unusual zoom
on a rotating minimap needs a full search of a minute or two.

## Limits

- Click positions are measured for Strats.gg 2026.10.2. A layout change in a
  later version needs the constants at the top of `companion/driver.py` updated.
- Strats.gg must be open and not minimised (covered is fine after `launch`).
- Valorant must run in Windowed Fullscreen for the screen to be capturable, on
  the primary monitor, with the minimap in its default top-left position.
- The spike is missed while teammates' icons sit on top of it.
- Riot's third-party policy does not approve tools that give live decision
  help during a round. Using this in real matches is at your own risk.
