"""Checks GitHub for a newer release of the exe and swaps it in.

The repository is private, so requests carry the GitHub token that Git
Credential Manager already holds on this machine. Without a token the same
requests go out anonymously, which works if the repository is made public.
"""

import os
import subprocess
import sys
from pathlib import Path

import requests

from . import __version__

REPO = "qPublic/strats-companion"
ASSET_NAME = "StratsCompanion.exe"
API_URL = f"https://api.github.com/repos/{REPO}"


class UpdateError(RuntimeError):
    pass


def _parse(version):
    return tuple(int(part) for part in version.lstrip("v").split("."))


def _token():
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        filled = subprocess.run(
            ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", env=environment,
            capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return next((line.split("=", 1)[1] for line in filled.splitlines() if line.startswith("password=")), None)


def _headers(accept):
    headers = {"Accept": accept}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def check():
    """The newer release as {'version', 'asset_url'}, or None when this build is current."""
    try:
        response = requests.get(f"{API_URL}/releases/latest", headers=_headers("application/vnd.github+json"), timeout=15)
    except requests.RequestException as error:
        raise UpdateError(f"Could not reach GitHub: {error}") from error
    if response.status_code == 404:
        raise UpdateError("No access to the releases. Sign in to GitHub with Git, or make the repository public.")
    if response.status_code != 200:
        raise UpdateError(f"GitHub returned {response.status_code} for the latest release.")
    release = response.json()
    if _parse(release["tag_name"]) <= _parse(__version__):
        return None
    asset = next((item for item in release["assets"] if item["name"] == ASSET_NAME), None)
    if asset is None:
        raise UpdateError(f"Release {release['tag_name']} has no {ASSET_NAME}.")
    return {"version": release["tag_name"].lstrip("v"), "asset_url": asset["url"]}


def can_install():
    return getattr(sys, "frozen", False)


def install(update, on_progress=None):
    """Download the new exe and put it in place of the running one. Call restart() afterwards."""
    if not can_install():
        raise UpdateError("Updates install only into the packaged exe. From source, use git pull.")
    current = Path(sys.executable)
    incoming, previous = current.with_suffix(".new.exe"), current.with_suffix(".old.exe")
    try:
        with requests.get(update["asset_url"], headers=_headers("application/octet-stream"), stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(incoming, "wb") as target:
                for chunk in response.iter_content(1 << 20):
                    target.write(chunk)
                    done += len(chunk)
                    if on_progress and total:
                        on_progress(done / total)
    except (requests.RequestException, OSError) as error:
        incoming.unlink(missing_ok=True)
        raise UpdateError(f"Download failed: {error}") from error
    # Windows lets a running exe be renamed, so move it aside and put the new one in its place.
    previous.unlink(missing_ok=True)
    current.rename(previous)
    try:
        incoming.rename(current)
    except OSError as error:
        previous.rename(current)
        raise UpdateError(f"Could not replace the exe: {error}") from error


def restart():
    """Start the (replaced) exe fresh and end this process."""
    # Without this the new one-file exe would reuse this process's unpacked files.
    environment = {**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}
    subprocess.Popen([sys.executable], env=environment, close_fds=True,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    os._exit(0)


def remove_previous():
    """Delete the exe left behind by the last update."""
    if can_install():
        try:
            Path(sys.executable).with_suffix(".old.exe").unlink(missing_ok=True)
        except OSError:
            pass
