"""Current match details (map, agent, side) from the running Riot Client.

Uses the client's local lockfile API plus the same match endpoint that rank
trackers use. Read-only, and only about the logged-in player's own match.
"""

import base64
import json
import os
import re
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOCKFILE = Path(os.environ.get("LOCALAPPDATA", "")) / "Riot Games" / "Riot Client" / "Config" / "lockfile"
CLIENT_PLATFORM = base64.b64encode(json.dumps({
    "platformType": "PC", "platformOS": "Windows",
    "platformOSVersion": "10.0.19042.1.256.64bit", "platformChipset": "Unknown",
}).encode()).decode()
SHARDS = {"latam": "na", "br": "na", "na": "na", "pbe": "pbe", "eu": "eu", "ap": "ap", "kr": "kr"}
ATTACKERS_FIRST = "Red"
HALF_LENGTHS = {"swiftplay": 4}
DEFAULT_HALF_LENGTH = 12


class NotAvailable(RuntimeError):
    """The Riot Client is not running, or the player is not in a match."""


class RiotClient:
    def __init__(self):
        self._client_version = None

    def _local(self, path):
        if not LOCKFILE.exists():
            raise NotAvailable("Riot Client is not running.")
        _, _, port, password, _ = LOCKFILE.read_text().split(":")
        try:
            response = requests.get(f"https://127.0.0.1:{port}{path}", auth=("riot", password), verify=False, timeout=5)
        except requests.ConnectionError as error:
            raise NotAvailable("Riot Client is not responding.") from error
        if response.status_code != 200:
            raise NotAvailable(f"Riot Client returned {response.status_code} for {path}.")
        return response.json()

    def _region(self):
        sessions = self._local("/product-session/v1/external-sessions")
        for session in sessions.values():
            for argument in session.get("launchConfiguration", {}).get("arguments", []):
                match = re.match(r"-ares-deployment=(\w+)", argument)
                if match:
                    return match.group(1)
        raise NotAvailable("VALORANT is not running.")

    def _headers(self, tokens):
        if self._client_version is None:
            version = requests.get("https://valorant-api.com/v1/version", timeout=10).json()["data"]
            self._client_version = version["riotClientVersion"]
        return {
            "Authorization": f"Bearer {tokens['accessToken']}",
            "X-Riot-Entitlements-JWT": tokens["token"],
            "X-Riot-ClientPlatform": CLIENT_PLATFORM,
            "X-Riot-ClientVersion": self._client_version,
        }

    def _presence(self, puuid):
        for presence in self._local("/chat/v4/presences").get("presences", []):
            if presence.get("puuid") == puuid and presence.get("product") == "valorant":
                return json.loads(base64.b64decode(presence["private"]))
        return {}

    def current_match(self):
        """{'map_path', 'agent_id', 'team', 'side', 'rounds_played'} for the match in progress."""
        tokens = self._local("/entitlements/v1/token")
        puuid = tokens["subject"]
        region = self._region()
        base = f"https://glz-{region}-1.{SHARDS.get(region, region)}.a.pvp.net"
        headers = self._headers(tokens)
        player = requests.get(f"{base}/core-game/v1/players/{puuid}", headers=headers, timeout=10)
        if player.status_code != 200:
            raise NotAvailable("Not in a match.")
        match = requests.get(f"{base}/core-game/v1/matches/{player.json()['MatchID']}", headers=headers, timeout=10)
        if match.status_code != 200:
            raise NotAvailable("Match details are not available yet.")
        details = match.json()
        me = next(item for item in details["Players"] if item["Subject"] == puuid)

        presence = self._presence(puuid)
        scores = presence.get("partyPresenceData", presence)
        rounds_played = (scores.get("partyOwnerMatchScoreAllyTeam") or 0) + (scores.get("partyOwnerMatchScoreEnemyTeam") or 0)
        queue = (presence.get("queueId") or presence.get("matchPresenceData", {}).get("queueId") or "").lower()
        return {
            "map_path": details["MapID"],
            "agent_id": me["CharacterID"].lower(),
            "team": me["TeamID"],
            "rounds_played": rounds_played,
            "half_length": HALF_LENGTHS.get(queue, DEFAULT_HALF_LENGTH),
            "side": side_for(me["TeamID"], rounds_played, HALF_LENGTHS.get(queue, DEFAULT_HALF_LENGTH)),
        }


def sides_swapped(rounds_played, half_length=DEFAULT_HALF_LENGTH):
    """Whether teams are on the opposite side from the one they started on."""
    if rounds_played < half_length:
        return False
    if rounds_played < 2 * half_length:
        return True
    return (rounds_played - 2 * half_length) % 2 == 1


def side_for(team, rounds_played, half_length=DEFAULT_HALF_LENGTH):
    """Which side `team` plays in the round after `rounds_played` rounds.

    The team-name rule is only a first guess: it has been seen wrong in a
    custom game. The watcher corrects it from what is on screen.
    """
    starts_attacking = team == ATTACKERS_FIRST
    return "attack" if starts_attacking != sides_swapped(rounds_played, half_length) else "defense"
