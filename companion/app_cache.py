"""Lineup data the Strats.gg app has already loaded, read from its HTTP cache.

The app is a Chromium browser and keeps a disk cache (the "blockfile" format)
of the responses it receives. Its copies of the lineup queries are saved into
the companion's own cache, so whatever the app has shown keeps working while
Strats.gg refuses the companion's own requests.
"""

import ctypes
import json
import msvcrt
import os
import re
import struct
import time
import urllib.parse
import zlib
from ctypes import wintypes
from pathlib import Path

from . import devalue, strats_api

CACHE_DATA = Path(os.environ.get("APPDATA", "")) / "Strats.gg" / "Cache" / "Cache_Data"
API_PREFIX = "https://strats.gg/api/trpc/"
BLOCK_FILE_HEADER = 8192
ENTRY_SIZE = 256
BLOCK_SIZES = {2: 256, 3: 1024, 4: 4096}
WINDOWS_EPOCH_OFFSET = 11644473600     # seconds from 1601-01-01 to 1970-01-01
SAVED_NAMES = {"valorant.lineups.maps": "maps", "valorant.lineups.agents": "agents"}

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.CreateFileW.argtypes = (
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
)


def _open(path):
    """Open a file Chromium holds open; it only allows readers that also share delete access."""
    handle = _kernel32.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f"cannot open {path}")
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb")


def _read(address, size):
    """Bytes stored at a cache address: a block in one of the data_N files, or a separate f_ file."""
    if not address & 0x80000000 or size <= 0:
        return b""
    kind = (address >> 28) & 0x7
    if kind == 0:
        with _open(CACHE_DATA / f"f_{address & 0x0FFFFFFF:06x}") as file:
            return file.read(size)
    if kind not in BLOCK_SIZES:
        return b""
    with _open(CACHE_DATA / f"data_{(address >> 16) & 0xFF}") as file:
        file.seek(BLOCK_FILE_HEADER + (address & 0xFFFF) * BLOCK_SIZES[kind])
        return file.read(size)


def _entries():
    """(url, unix time stored, headers, body) for every cached Strats.gg API response."""
    with _open(CACHE_DATA / "data_1") as file:
        entries = file.read()
    for offset in range(BLOCK_FILE_HEADER, len(entries) - ENTRY_SIZE + 1, ENTRY_SIZE):
        entry = entries[offset:offset + ENTRY_SIZE]
        created, key_length, long_key = struct.unpack_from("<qiI", entry, 24)
        if not 0 < key_length < 1 << 20:
            continue
        try:
            # A key up to about 1 KB is stored inline, running on into the entry's following blocks.
            key = _read(long_key, key_length) if long_key else entries[offset + 96:offset + 96 + key_length]
            if API_PREFIX.encode() not in key:
                continue
            sizes = struct.unpack_from("<4i", entry, 40)
            addresses = struct.unpack_from("<4I", entry, 56)
            headers, body = _read(addresses[0], sizes[0]), _read(addresses[1], sizes[1])
        except OSError:
            continue
        if not body:
            continue
        key = key.decode("latin-1")
        yield key[key.rindex(API_PREFIX):], created / 1e6 - WINDOWS_EPOCH_OFFSET, headers, body


def _decode(headers, body):
    encoding = re.search(rb"content-encoding:\s*([\w-]+)", headers, re.IGNORECASE)
    if encoding is None:
        return body
    if encoding.group(1).lower() == b"gzip":
        return zlib.decompress(body, 47)
    raise ValueError(f"unsupported encoding {encoding.group(1)!r}")


def _results(url, headers, body):
    """(procedure, input, result) for each query batched into one cached response."""
    parts = urllib.parse.urlsplit(url)
    procedures = urllib.parse.unquote(parts.path[len("/api/trpc/"):]).split(",")
    inputs = json.loads(urllib.parse.parse_qs(parts.query).get("input", ["{}"])[0])
    answers = json.loads(_decode(headers, body))
    for index, (procedure, answer) in enumerate(zip(procedures, answers)):
        if "result" not in answer:
            continue
        payload = inputs.get(str(index))
        yield procedure, None if payload is None else devalue.parse(payload), devalue.parse(answer["result"]["data"])


def harvest():
    """Save the app's cached lineup data where newer than the companion's copy. Returns what was saved."""
    if not (CACHE_DATA / "data_1").exists():
        return []
    saved = []
    try:
        entries = list(_entries())
    except OSError:
        return []
    for url, stored, headers, body in entries:
        try:
            results = list(_results(url, headers, body))
        except (ValueError, KeyError, TypeError, zlib.error):
            continue
        for procedure, payload, result in results:
            if procedure == "valorant.lineups.all" and isinstance(payload, dict):
                path = strats_api.lineups_path(payload["mapId"], payload["agentId"], payload["side"])
            elif procedure in SAVED_NAMES:
                path = strats_api.CACHE_DIR / f"{SAVED_NAMES[procedure]}.json"
            else:
                continue
            if path.exists() and path.stat().st_mtime >= stored:
                continue
            strats_api.save(path, result)
            os.utime(path, (time.time(), stored))
            if path.stem not in saved:
                saved.append(path.stem)
    return saved
