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

The window checks GitHub for a newer release when it opens, every 10 minutes
while it is open, and whenever you press "Check for updates". When there is
one, an "Install ... now" button appears. Installing downloads the new exe and
relaunches it on its own, with the same settings and window size, and carries
on watching if it was. The old exe is kept as `StratsCompanion.old.exe` until
the next start.

The window can be resized; text, buttons and checkboxes scale with it. Its
settings are kept in `%LOCALAPPDATA%\StratsCompanion\settings.json`.

## Run from source

```
python -m venv .venv
.venv\Scripts\python -m pip install requests numpy opencv-python mss keyboard dxcam
```

Build the exe with `pyinstaller --onefile --console --name StratsCompanion
--add-data "companion/assets;companion/assets" --collect-submodules dxcam
--hidden-import comtypes.stream strats_companion.py`.

The screen is captured with Windows' Desktop Duplication (dxcam), falling back to
GDI (mss) if that is not available; `selftest` reports which is in use.

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
| `selftest` | Reports how the screen is captured and how fast. |
| `capture [--key f9]` | Saves a screenshot of the primary monitor on each key press, for tuning minimap detection. |

Positions are `left,top` in percent of the Strats.gg map image for that side.

## Where the lineup data comes from

The companion asks Strats.gg's API for each map, agent and side and keeps a
copy in `%LOCALAPPDATA%\StratsCompanion\cache`. While the window is open it
also downloads every agent on every map in the background.

Strats.gg sometimes refuses requests that don't come from a browser. Then the
companion uses, in order: the lineups the Strats.gg app itself has loaded
(read from the app's own cache on this PC), its saved copy, and finally, for a
match it has no data for, it opens that map and agent in Strats.gg so the app
loads them, and reads them from there.

## How a lineup is chosen

A lineup qualifies when it lands within 4.5 m of the spike and uses the
agent's molly (or a similar thrown ability that damages a defuser, such as
Sova's shock darts or Killjoy's Nanoswarm, listed in `companion/selector.py`).
Smokes, recon, traps and ultimates are never picked for an agent that has a
molly; if none of its mollies lands on the spike, nothing is opened. Agents
without one can use any lineup that lands there.

With "Full-screen the lineup video" ticked, an opened lineup's video fills the
Strats.gg window (Strats.gg does not allow true full screen). With "Minimise
Strats.gg between lineups" ticked, Strats.gg is set up on the right map and
then minimised, and is restored when a lineup opens.

Standing spots within 14 m of a threat are skipped. A
threat is an enemy seen on the minimap (red-ringed icon) or a place where a
teammate died (blue X), and each one is forgotten 10 seconds after it was last
seen, on the assumption that the enemy has moved. If every spot is that close
to a threat, the one farthest from them is opened.

Of the rest, the one thrown from closest to the spike inside a right-angle
cone opening south from the spike (down the map, towards your own side) is
opened, preferring spots within 35 m of you. If no spot
is in the cone, the standing spot nearest you is opened instead. The limits
are `SPIKE_RADIUS`, `MAX_WALK` and `DANGER_RADIUS` in `companion/selector.py`,
in metres and converted with each map's own scale;
`THREAT_SECONDS` is in `companion/watcher.py`.

The pick can change while you move. Once you are within 25 m of the opened
lineup's standing spot it is locked in for the rest of the round, threats
included, and nothing else is opened until the spike is gone (`LOCK_METRES`
in `companion/watcher.py`).

To stay unpredictable, once a lineup has been used twice for the same plant
(spikes within 3 m of each other) during a match, the next plant there gets
the best other lineup; when every lineup has had two turns, the least used
one is picked (`REPEAT_LIMIT` in `companion/selector.py`). The count starts
again on a new map or side.

When the spike is defused or explodes (its indicator gone for 3 reads in a
row) or the next round starts, the lineup is closed and Strats.gg goes back to
the map, so the next lineup is a single click away.

## Status

- Working and tested: lineup data, lineup choice, driving the Strats.gg window.
- Working on recorded footage (one 1080p Ascent match from YouTube, rotating
  minimap) and on live 1440p frames (Ascent, fixed minimap): detecting a
  planted spike, its position and the player's position.
- Not yet confirmed on a live game: a full plant-to-lineup run, and any map
  other than Ascent.

## In-game guide

With "Show in-game guide" ticked, the chosen lineup is drawn over the game
(Valorant must run in Windowed Fullscreen):

- **Where to stand**, on the minimap: a ring on the standing spot, a dashed
  line to where it lands, and the distance, which turns to "In position"
  within 2 m.
- **Where to stand, in the world**: a ring on the floor at the standing spot
  with a post and the distance, drawn in perspective where the spot is, so you
  can walk to it without looking at the minimap. When the spot is behind you
  or off screen, an arrow along the bottom says which way to turn. The camera
  is worked out from the screen: position from your minimap icon, facing from
  the pointer on it, looking up or down from how vertical edges lean, and
  every small turn in between from how the picture shifts (about 30 times a
  second). The floor is taken to be level with your feet.
- **Where to aim**, once in position: the lineup's aim screenshot is matched
  against the screen. Standing on the spot, the two views differ only by how
  the camera is turned (the field of view is fixed), so the match gives that
  rotation, and a reticle is drawn on the aim point. When the aim point is off
  screen, an arrow at the edge says which way and how far to turn. If the
  scenery cannot be matched, the screenshot itself is shown on the right.

The guide comes up as soon as the lineup is picked; Strats.gg opens the video
alongside it. The drawing ignores the mouse, never takes focus, and is kept out of screen
captures, so the companion's own minimap reads never see it. It uses only the
screen, never the game's memory. The aim screenshot comes from Strats.gg, or
from the Strats.gg app's cache once the lineup has been opened there. It is
stored as its matching features plus a small preview (about 100 KB per lineup,
in the cache folder's aim directory). The background download fetches these
for every agent's mollies, and when a match starts, the stored data for its
map, agent and side is loaded, so nothing is fetched when the spike goes down.

## Test mode

With "Test mode: a dropped spike counts as planted" ticked, a spike lying on
the ground (seen on the minimap) is treated as planted there, so lineups can be
practised without planting or starting new rounds. Drop the spike, walk away
from it (a spike within 3 m of you is taken to be the one you carry), and the
lineup opens. Pick it up and drop it somewhere else for the next one.

## Broken lineups and trying the next one

Some lineups no longer work (a map change can break a bounce). Press "Broken
lineup" while one is open: it is never picked again and the next best one opens
straight away. "Clear broken (N)" forgets all the marks. The list is kept in
`%LOCALAPPDATA%\StratsCompanion\broken_lineups.json`.

In test mode, "Next lineup" opens the next lineup for the same plant, ignoring
the lock, to try several standing spots in a row; after the last one it starts
again from the first.

"Next lineup hotkey" does the same from inside the game (F8 unless changed).
Untick it to turn it off; "Change" waits for the next key or key combination
you press and uses that. The key still reaches the game.

## How your side is decided

With Side on Auto, the Riot client's team name gives a first guess, which can
be wrong (it was in a custom game). Then:

- **Respawns.** At the start of a round you are put in your team's spawn. A
  jump of more than 25 m between two reads can only be that respawn, and if
  you then stay at a spawn for 3 reads, your side is the side of that spawn.
  Where you stand later in the round is never used.
- **Team changes.** If the Riot client reports that your team changed during
  the match (custom games can swap teams), the side swaps with it.
- **Minimap orientation.** A fixed minimap set to follow your side is drawn
  with your own spawn at the bottom; if it keeps matching the map art upside
  down, the side is flipped. The first respawn shows whether your minimap
  follows your side or is always drawn the same way, and that is remembered;
  a minimap that is always the same way is not used for this. No lineup is
  picked while this check is pending.

Once known, the side follows the round count: teams swap after 12 rounds
(4 in Swiftplay) and every round in overtime. Picking a side by hand in the
window turns the automatic logic off.

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
