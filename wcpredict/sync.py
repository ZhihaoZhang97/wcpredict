"""Sync data/ with the upstream openfootball/worldcup.json 2026 files.

Stdlib only. Used by scripts/sync_data.py (manual / CI) and by the
predict CLI, which syncs before every prediction so results are current.

Data source: https://github.com/openfootball/worldcup.json (CC0).
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
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def sync_files(data_dir: Path = DEFAULT_DATA_DIR, timeout: int = 30) -> list[str]:
    """Download the upstream files; return the names that changed.

    Raises OSError on network failure and ValueError on invalid upstream
    JSON — callers decide whether that is fatal (CI) or a warning (CLI).
    """
    data_dir.mkdir(exist_ok=True)
    changed = []
    for name in FILES:
        with urllib.request.urlopen(f"{UPSTREAM}/{name}", timeout=timeout) as resp:
            raw = resp.read()
        json.loads(raw)  # reject invalid upstream content before writing

        target = data_dir / name
        if not target.exists() or target.read_bytes() != raw:
            target.write_bytes(raw)
            changed.append(name)
    return changed


def main() -> int:
    try:
        changed = sync_files()
    except (OSError, ValueError) as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        return 1
    for name in FILES:
        print(f"  {'SYNC' if name in changed else 'ok  '}  {name}")
    print(f"\n{len(changed)} file(s) updated" if changed else "\nalready up to date")
    if changed:
        print("run `uv run python -m wcpredict.check` to validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
