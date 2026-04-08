from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "_vendor"
LEGACY_SITE_PACKAGES = APP_DIR / ".venv" / "Lib" / "site-packages"

for candidate in (LEGACY_SITE_PACKAGES, VENDOR_DIR):
    if candidate.exists():
        candidate_path = str(candidate)
        if candidate_path not in sys.path:
            sys.path.insert(0, candidate_path)
