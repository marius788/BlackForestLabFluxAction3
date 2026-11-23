"""Check that a recorded NexArm dataset matches what FLUX-3-Action (SO-101 contract) expects.

Hard checks (FAIL):
  * action / observation.state have the six SO-101 joint names
  * cameras observation.images.scene and observation.images.wrist exist
  * arm joints are in degrees (|value| <= 180) and the gripper is in 0..100 -- raw servo ticks
    (values around 1000..3500) mean the old Hiwonder fork was used for recording
Information (no FAIL):
  * per joint: state range next to the q01/q99 state range of the F3A SO-101 checkpoint
  * per joint: range of the per-frame command step action[t] - state[t]

Usage:
    python check_f3a_dataset.py                          newest local/<prefix>_* in HF_LEROBOT_HOME
    python check_f3a_dataset.py nexarm_f3a_stack_boxes_20261113_150000
    python check_f3a_dataset.py C:\\path\\to\\dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EXPECTED_NAMES = [f"{j}.pos" for j in JOINTS]
EXPECTED_CAMERAS = ["observation.images.scene", "observation.images.wrist"]
CONTRACT = Path(__file__).resolve().parent.parent / "plugin" / "reference" / "contract_report.json"


def find_dataset(name: str | None, prefix: str) -> Path:
    if name and (Path(name) / "meta" / "info.json").is_file():
        return Path(name)
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        sys.exit("HF_LEROBOT_HOME is not set; start this through check-dataset.cmd or pass a full path.")
    local = Path(home) / "local"
    if name:
        if not (local / name / "meta" / "info.json").is_file():
            sys.exit(f"No dataset at {local / name}")
        return local / name
    candidates = sorted(d for d in local.glob(f"{prefix}_*") if (d / "meta" / "info.json").is_file())
    if not candidates:
        sys.exit(f"No dataset local/{prefix}_* found in {local}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="dataset folder name in local/ or a full path")
    parser.add_argument("--prefix", default="nexarm_f3a_stack_boxes")
    args = parser.parse_args()

    root = find_dataset(args.name, args.prefix)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    print(f"Dataset: {root}")
    print(f"  {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps, "
          f"robot_type={info.get('robot_type')}")

    problems: list[str] = []
    for key in ("action", "observation.state"):
        names = features.get(key, {}).get("names")
        if names != EXPECTED_NAMES:
            problems.append(f"{key}: joint names are {names}, expected {EXPECTED_NAMES}")
    for cam in EXPECTED_CAMERAS:
        if cam not in features:
            present = [k for k in features if k.startswith("observation.images.")]
            problems.append(f"camera {cam} is missing (present: {present})")
    if info["fps"] != 30:
        problems.append(f"fps is {info['fps']}, the setup records at 30")

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        print("\nFAIL: no recorded frames (data/*.parquet missing)")
        return 1
    data = pd.concat(pd.read_parquet(p, columns=["action", "observation.state"]) for p in files)
    action = np.stack(data["action"].to_numpy()).astype(float)
    state = np.stack(data["observation.state"].to_numpy()).astype(float)
    step = action - state

    reference = None
    if CONTRACT.is_file():
        reference = json.loads(CONTRACT.read_text(encoding="utf-8"))["statistics"]["state"]

    print("\n  joint            state min..max        action min..max       action-state step     F3A SO-101 state q01..q99")
    for i, joint in enumerate(JOINTS):
        ref = f"{reference['q01'][i]:8.1f} .. {reference['q99'][i]:6.1f}" if reference else "n/a"
        print(
            f"  {joint:<14} {state[:, i].min():8.1f} .. {state[:, i].max():8.1f}   "
            f"{action[:, i].min():8.1f} .. {action[:, i].max():8.1f}   "
            f"{step[:, i].min():7.1f} .. {step[:, i].max():6.1f}    {ref}"
        )
        values = np.concatenate([state[:, i], action[:, i]])
        if joint == "gripper":
            if values.min() < -1 or values.max() > 101:
                problems.append(f"gripper: values {values.min():.0f}..{values.max():.0f} are not in 0..100")
        elif np.abs(values).max() > 180:
            problems.append(
                f"{joint}: values up to {np.abs(values).max():.0f} are not degrees "
                "(raw servo ticks? recorded with the old Hiwonder fork?)"
            )

    print()
    if problems:
        print("FAIL - this dataset does not match the FLUX-3-Action contract:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK - units, joint names and camera names match the FLUX-3-Action SO-101 contract.")
    print("Note: the F3A column is the SO-101 training range, only for orientation. Signs and the")
    print("zero pose must still match the SO-101 convention (see README.md, section Kalibrierung).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
