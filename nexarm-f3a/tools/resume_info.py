"""Print "<dataset name> <episodes missing> <dataset path>" for the newest local dataset starting with PREFIX.

Used by 6-record-continue.cmd. Exits with 1 if no recorded dataset with that prefix exists.
Usage: python resume_info.py PREFIX TARGET_EPISODES   (HF_LEROBOT_HOME must be set, see _env.cmd)
"""

import json
import os
import sys
from pathlib import Path

prefix, target = sys.argv[1], int(sys.argv[2])
local = Path(os.environ["HF_LEROBOT_HOME"]) / "local"


def episodes(dataset: Path) -> int:
    info = dataset / "meta" / "info.json"
    return json.loads(info.read_text(encoding="utf-8"))["total_episodes"] if info.is_file() else 0


candidates = sorted(d for d in local.glob(f"{prefix}_*") if episodes(d) > 0)
if not candidates:
    sys.exit(1)
newest = candidates[-1]
# The path goes last: it may contain spaces (the .cmd reads it with tokens=1,2*).
# lerobot-record --resume needs it as --dataset.root.
print(newest.name, max(0, target - episodes(newest)), newest)
