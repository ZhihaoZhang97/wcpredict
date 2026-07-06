#!/usr/bin/env python3
"""Sync data/ with the upstream openfootball/worldcup.json 2026 files.

Downloads the tournament JSON files from GitHub, validates that each one
parses, and reports what changed. Stdlib only — runs before any
dependencies are installed:

    python3 scripts/sync_data.py

Data source: https://github.com/openfootball/worldcup.json (CC0).
After a sync, `uv run python -m wcpredict.check` validates the data
against the pipeline's expectations.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

UPSTREAM = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026"
FILES = [
    "worldcup.json",
    "worldcup.squads.json",
    "worldcup.teams.json",
    "worldcup.quali_playoffs.json",
    "worldcup.groups.json",
    "worldcup.stadiums.json",
]
DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    changed = 0
    for name in FILES:
        url = f"{UPSTREAM}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read()
        except OSError as exc:
            print(f"  FAIL  {name}: {exc}", file=sys.stderr)
            return 1
        try:
            json.loads(raw)
        except ValueError as exc:
            print(f"  FAIL  {name}: upstream is not valid JSON ({exc})", file=sys.stderr)
            return 1

        target = DATA_DIR / name
        old = target.read_bytes() if target.exists() else None
        if old == raw:
            print(f"  ok    {name} (unchanged)")
        else:
            target.write_bytes(raw)
            delta = f"{len(raw) - len(old):+d} bytes" if old is not None else "new"
            print(f"  SYNC  {name} ({delta})")
            changed += 1

    print(f"\n{changed} file(s) updated" if changed else "\nalready up to date")
    if changed:
        print("run `uv run python -m wcpredict.check` to validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
