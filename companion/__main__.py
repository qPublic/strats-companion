"""Command line entry point: python -m companion <command>"""

import argparse
import sys
import time
from datetime import datetime

from . import driver, geometry, selector, strats_api
from .driver import DriverError, StratsWindow
from .paths import CAPTURE_DIR

MATCH_POLL_SECONDS = 5


def _point(text):
    left, top = (float(part) for part in text.split(","))
    return left, top


def _approved(map_item, agent_item, side):
    lineups = strats_api.lineups(map_item["id"], agent_item["id"], side)
    return [item for item in lineups if item["status"] == "approved"]


def _ensure_strats(restart):
    status = driver.launch(restart=restart)
    if status == "needs-restart":
        sys.exit(
            "Strats.gg is running without the keep-active switches, so it ignores clicks while covered.\n"
            "Run `python -m companion launch --restart` (or add --restart to this command) to restart it."
        )
    return status


def launch(args):
    """Start Strats.gg so it stays responsive while covered by other windows."""
    print({"running": "Strats.gg is already running correctly.", "started": "Strats.gg started."}[_ensure_strats(args.restart)])


def show(args):
    """Open the best lineup in Strats.gg for a spike and player position given by hand."""
    maps, agents = strats_api.maps(), strats_api.agents()
    map_item = strats_api.find_by_name(maps, args.map)
    agent_item = strats_api.find_by_name(agents, args.agent)
    if map_item is None:
        sys.exit(f"Unknown map: {args.map}")
    if agent_item is None:
        sys.exit(f"Unknown agent: {args.agent}")
    _ensure_strats(args.restart)
    lineups = _approved(map_item, agent_item, args.side)
    groups = geometry.group_lineups(lineups)
    window = StratsWindow()
    try:
        window.show_map(maps, agents, map_item, agent_item, args.side, groups)
        if args.spike is None:
            print(f"Showing {map_item['name']} / {agent_item['name']} / {args.side} ({len(lineups)} lineups).")
            return
        lineup = selector.choose(lineups, args.spike, args.player, selector.post_plant_ability_ids(agent_item))
        if lineup is None:
            print("No lineup lands on that spike position.")
            return
        opened = window.open_lineup(groups, selector.group_of(groups, lineup), lineup)
    except DriverError as error:
        sys.exit(str(error))
    print(f"{'Opened' if opened else 'Could not open'}: {lineup['title']} (#{lineup['id']})")


def follow(args):
    """Keep Strats.gg on the map, agent and side of the match you are playing."""
    from .riot_local import NotAvailable, RiotClient

    _ensure_strats(args.restart)
    maps, agents = strats_api.maps(), strats_api.agents()
    riot, window, shown = RiotClient(), StratsWindow(), None
    print("Following your match. Press Ctrl+C to stop.")
    while True:
        try:
            match = riot.current_match()
        except NotAvailable as reason:
            if shown != str(reason):
                shown = str(reason)
                print(f"Waiting: {reason}")
            time.sleep(MATCH_POLL_SECONDS)
            continue
        map_item = next((item for item in maps if item["assetPath"].lower() == match["map_path"].lower()), None)
        agent_item = next((item for item in agents if item["id"].lower() == match["agent_id"]), None)
        side = args.side or match["side"]
        target = (map_item and map_item["name"], agent_item and agent_item["name"], side)
        if map_item and agent_item and target != shown:
            groups = geometry.group_lineups(_approved(map_item, agent_item, side))
            try:
                window.show_map(maps, agents, map_item, agent_item, side, groups)
                shown = target
                print(f"Showing {target[0]} / {target[1]} / {side}")
            except DriverError as error:
                print(f"Strats.gg: {error}")
        time.sleep(MATCH_POLL_SECONDS)


def _frames(video, step):
    """Frames to analyse: the live primary screen, or a video file sampled every `step` seconds."""
    import cv2

    from . import win

    if video is None:
        while True:
            started = time.time()
            yield win.capture_screen()
            time.sleep(max(0.0, step - (time.time() - started)))
    source = cv2.VideoCapture(video)
    position = 0.0
    while True:
        source.set(cv2.CAP_PROP_POS_MSEC, position * 1000)
        ok, frame = source.read()
        if not ok:
            return
        yield frame
        position += step


def _same_spot(first, second, tolerance=2.0):
    return first is not None and second is not None and geometry.distance(first, second) <= tolerance


def run(args):
    """Watch the minimap and open the best post-plant lineup in Strats.gg whenever the spike is planted."""
    from . import map_shape
    from .minimap import MinimapReader
    from .riot_local import NotAvailable, RiotClient

    maps, agents = strats_api.maps(), strats_api.agents()
    manual = args.map and args.agent
    riot = None if manual else RiotClient()
    if not args.dry_run:
        _ensure_strats(args.restart)
    window = StratsWindow()
    target = reader = lineups = groups = map_item = agent_item = None
    last_match_check = 0.0
    last_spike = last_player = opened = pending = None
    print("Watching the minimap. Press Ctrl+C to stop.")

    for frame in _frames(args.video, args.interval):
        if manual:
            wanted = (args.map, args.agent, args.side or "attack")
        elif time.time() - last_match_check >= MATCH_POLL_SECONDS:
            last_match_check = time.time()
            try:
                match = riot.current_match()
            except NotAvailable:
                continue
            found_map = next((item for item in maps if item["assetPath"].lower() == match["map_path"].lower()), None)
            found_agent = next((item for item in agents if item["id"].lower() == match["agent_id"]), None)
            if found_map is None or found_agent is None:
                continue
            wanted = (found_map["name"], found_agent["name"], args.side or match["side"])
        elif target is None:
            continue
        else:
            wanted = target

        if wanted != target:
            target = wanted
            map_item = strats_api.find_by_name(maps, target[0])
            agent_item = strats_api.find_by_name(agents, target[1])
            if map_item is None or agent_item is None:
                sys.exit(f"Unknown map or agent: {target[0]} / {target[1]}")
            lineups = _approved(map_item, agent_item, target[2])
            groups = geometry.group_lineups(lineups)
            reader = MinimapReader(map_shape.silhouette(map_item, target[2]), frame.shape[0], MinimapReader.stored_scale(frame.shape))
            opened = pending = None
            print(f"Match: {map_item['name']} / {agent_item['name']} / {target[2]} ({len(lineups)} lineups)")

        if reader.scale is None:
            score = reader.calibrate(frame)
            if reader.scale is None:
                continue
            print(f"Minimap zoom calibrated ({reader.scale}, match score {score:.2f}).")

        reading = reader.read(frame)
        if reading.player is not None:
            last_player = reading.player
        if not reading.planted:
            last_spike = None
            if opened is not None:
                opened = pending = None
                print("Spike no longer planted.")
            continue
        if reading.spike is None:
            continue
        # Act only on a spike position seen in two reads in a row.
        steady = _same_spot(reading.spike, last_spike)
        last_spike = reading.spike
        if not steady:
            continue
        lineup = selector.choose(lineups, reading.spike, last_player, selector.post_plant_ability_ids(agent_item))
        if lineup is None or lineup["id"] == opened:
            continue
        if lineup["id"] != pending:
            pending = lineup["id"]
            continue
        opened = lineup["id"]
        where = f"spike {reading.spike[0]:.0f},{reading.spike[1]:.0f}"
        if last_player is not None:
            where += f" / you {last_player[0]:.0f},{last_player[1]:.0f}"
        print(f"Lineup: {lineup['title']} (#{lineup['id']}) for {where}")
        if args.dry_run:
            continue
        try:
            window.show_map(maps, agents, map_item, agent_item, target[2], groups)
            window.open_lineup(groups, selector.group_of(groups, lineup), lineup)
        except DriverError as error:
            print(f"Strats.gg: {error}")


def capture(args):
    """Save a screenshot of the game screen each time a hotkey is pressed (for tuning detection)."""
    import cv2
    import keyboard

    from . import win

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    def save():
        path = CAPTURE_DIR / f"{datetime.now():%Y%m%d-%H%M%S-%f}.png"
        cv2.imwrite(str(path), win.capture_screen())
        print(f"Saved {path.name}")

    keyboard.add_hotkey(args.key, save)
    print(f"Press {args.key.upper()} in game to save a screenshot into {CAPTURE_DIR}. Press Ctrl+C here to stop.")
    keyboard.wait()


def main():
    parser = argparse.ArgumentParser(prog="companion")
    commands = parser.add_subparsers(dest="command", required=True)

    launch_parser = commands.add_parser("launch", help=launch.__doc__)
    launch_parser.set_defaults(run=launch)

    show_parser = commands.add_parser("show", help=show.__doc__)
    show_parser.add_argument("--map", required=True)
    show_parser.add_argument("--agent", required=True)
    show_parser.add_argument("--side", choices=["attack", "defense"], default="attack")
    show_parser.add_argument("--spike", type=_point, help="spike position as left,top in map percent")
    show_parser.add_argument("--player", type=_point, help="player position as left,top in map percent")
    show_parser.set_defaults(run=show)

    follow_parser = commands.add_parser("follow", help=follow.__doc__)
    follow_parser.add_argument("--side", choices=["attack", "defense"], help="force a side instead of detecting it")
    follow_parser.set_defaults(run=follow)

    run_parser = commands.add_parser("run", help=run.__doc__)
    run_parser.add_argument("--map", help="skip match lookup and use this map (needs --agent)")
    run_parser.add_argument("--agent", help="skip match lookup and use this agent (needs --map)")
    run_parser.add_argument("--side", choices=["attack", "defense"], help="force a side instead of detecting it")
    run_parser.add_argument("--interval", type=float, default=1.0, help="seconds between minimap reads")
    run_parser.add_argument("--video", help="analyse a video file instead of the live screen")
    run_parser.add_argument("--dry-run", action="store_true", help="print the chosen lineups without touching Strats.gg")
    run_parser.set_defaults(run=run)

    capture_parser = commands.add_parser("capture", help=capture.__doc__)
    capture_parser.add_argument("--key", default="f9")
    capture_parser.set_defaults(run=capture)

    for sub in (launch_parser, show_parser, follow_parser, run_parser):
        sub.add_argument("--restart", action="store_true", help="restart Strats.gg if it is running without the keep-active switches")

    # Double-clicking the exe gives no arguments: watch the game, restarting Strats.gg if needed.
    double_clicked = len(sys.argv) == 1
    args = parser.parse_args(["run", "--restart"] if double_clicked else None)
    try:
        args.run(args)
    except KeyboardInterrupt:
        pass
    except SystemExit as stop:
        if double_clicked and stop.code:
            print(stop.code)
            input("Press Enter to close.")
        raise


if __name__ == "__main__":
    main()
