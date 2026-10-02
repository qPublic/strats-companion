# Strats.gg lineup companion

Picks the Strats.gg lineup that fits the current spike and player position and
opens it in the Strats.gg desktop window. Clicks are sent to that window in the
background, so the mouse cursor and the game's focus are never touched.

## Install

Download `StratsCompanion.exe` from the latest release and put it anywhere.
Double-click it to start watching the game (same as `StratsCompanion.exe run
--restart`); it restarts Strats.gg if needed. Any command below also works as
`StratsCompanion.exe <command>`. Its data lives in
`%LOCALAPPDATA%\StratsCompanion`. The exe is unsigned, so Windows SmartScreen
will warn on first run.

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
| `launch [--restart]` | Starts Strats.gg with switches that keep it responsive while covered by other windows. `--restart` restarts it if it is already running without them. |
| `show --map ascent --agent viper --side attack --spike 20.8,27.4 --player 29.6,40.3` | Opens the best lineup for a spike and player position given by hand. Without `--spike` it only shows the lineup map. |
| `run [--map ascent --agent viper] [--side attack] [--dry-run] [--video file]` | Watches the minimap about once a second and opens the best post-plant lineup when the spike is planted. Map, agent and side come from the match unless given. `--video` analyses a recording instead of the screen; `--dry-run` only prints the choice. |
| `follow [--side attack]` | Keeps Strats.gg on the map, agent and side of the match in progress. |
| `capture [--key f9]` | Saves a screenshot of the primary monitor on each key press, for tuning minimap detection. |

Positions are `left,top` in percent of the Strats.gg map image for that side.

## How a lineup is chosen

A lineup qualifies when it lands within 3.5% of the map (about 5 m) of the
spike. Lineups using a post-plant ability (mollies, shock darts and similar,
listed in `companion/selector.py`) win over others. Among those, the one whose
standing spot is closest to the player is opened.

## Status

- Working and tested: lineup data, lineup choice, driving the Strats.gg window.
- Working on recorded footage (one 1080p Ascent match from YouTube, rotating
  minimap): detecting a planted spike, its position and the player's position,
  and `run` end to end from a video file.
- Not yet tested on a live game: screen capture of Valorant itself, the match
  lookup and side detection (`follow`, and `run` without `--map/--agent`), and
  any map other than Ascent.

## How the minimap is read

The red spike indicator that replaces the round timer says the spike is
planted. The minimap is matched against the map outline from Strats.gg's own
map art at any rotation, which converts screen positions to lineup
coordinates. The spike is the yellow icon on the map; the player is the icon
with a white ring (teammates have teal rings). The minimap zoom is found once
per screen resolution and stored in `calibration.json`; delete that file after
changing minimap size or zoom in Valorant.

## Limits

- Click positions are measured for Strats.gg 2026.10.2. A layout change in a
  later version needs the constants at the top of `companion/driver.py` updated.
- Strats.gg must be open and not minimised (covered is fine after `launch`).
- Valorant must run in Windowed Fullscreen for the screen to be capturable, on
  the primary monitor, with the minimap in its default top-left position.
- The spike is missed while teammates' icons sit on top of it, and the first
  calibration takes up to a minute.
- Riot's third-party policy does not approve tools that give live decision
  help during a round. Using this in real matches is at your own risk.
