"""Where the companion keeps its data: next to the code, or in AppData when run as an exe."""

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StratsCompanion"
else:
    DATA_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = DATA_DIR / "cache"
CAPTURE_DIR = DATA_DIR / "captures"
CALIBRATION_FILE = DATA_DIR / "calibration.json"
