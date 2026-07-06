#!/usr/bin/env python3
"""Thin wrapper so the sync runs standalone: python3 scripts/sync_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wcpredict.sync import main

if __name__ == "__main__":
    sys.exit(main())
