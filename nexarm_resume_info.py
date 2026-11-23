"""Print "<dataset name> <episodes missing> <dataset path>" for the newest local dataset whose name starts with PREFIX.

Used by nexarm-record-continue.cmd. Exits with 1 if no recorded dataset with that prefix exists.
Usage: python nexarm_resume_info.py PREFIX TARGET_EPISODES
"""

import json
import sys

from lerobot.utils.constants import HF_LEROBOT_HOME

prefix, target = sys.argv[1], int(sys.argv[2])
candidates = sorted(
    d
    for d in (HF_LEROBOT_HOME / "local").glob(f"{prefix}_*")
    if (d / "meta" / "info.json").is_file()
    and json.loads((d / "meta" / "info.json").read_text())["total_episodes"] > 0
)
if not candidates:
    sys.exit(1)
newest = candidates[-1]
recorded = json.loads((newest / "meta" / "info.json").read_text())["total_episodes"]
# The path goes last: it may contain spaces (the .cmd reads it with tokens=1,2*).
# lerobot_record --resume needs it as --dataset.root, resume() refuses root=None.
print(newest.name, max(0, target - recorded), newest)
