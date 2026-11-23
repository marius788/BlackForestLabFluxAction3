"""Quality check for a recorded NexArm LeRobot dataset.

For every episode it reports:
  * length (frames / seconds)
  * per camera: decoded video frames and the share of duplicated frames
    (camera delivered fewer than 30 fps, LeRobot repeated the last image)
  * follower lag: how many frames the follower state trails the leader action
    (best shift per joint), separately for the first and second half of the episode
    so a lag that grows over time becomes visible
  * remaining offset between action and state after removing the lag

Usage (inside open-nexarm-shell.cmd):
    python ..\\check_dataset.py                       # newest local/nexarm_stack_boxes_* dataset
    python ..\\check_dataset.py nexarm_stack_boxes_20261113_122530
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.utils.constants import HF_LEROBOT_HOME

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_LAG_FRAMES = 30
DUPLICATE_DIFF = 0.5  # mean absolute pixel difference below which two frames count as identical


def find_dataset(name: str | None, prefix: str) -> Path:
    local = HF_LEROBOT_HOME / "local"
    if name:
        path = local / name
        if not (path / "meta" / "info.json").is_file():
            sys.exit(f"No dataset at {path}")
        return path
    candidates = sorted(d for d in local.glob(f"{prefix}_*") if (d / "meta" / "info.json").is_file())
    if not candidates:
        sys.exit(f"No dataset local/{prefix}_* found in {local}")
    return candidates[-1]


def video_frames(path: Path, start_s: float, end_s: float) -> list[np.ndarray]:
    """Decode the frames of one episode (small grayscale copies are enough to detect duplicates)."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(start_s - 1.0, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s - 1e-3:
                continue
            if t >= end_s - 1e-3:
                break
            frames.append(frame.to_ndarray(format="gray")[::4, ::4].astype(np.int16))
    return frames


def duplicate_share(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    dups = sum(np.abs(a - b).mean() < DUPLICATE_DIFF for a, b in zip(frames, frames[1:]))
    return dups / (len(frames) - 1)


def best_lag(action: np.ndarray, state: np.ndarray) -> tuple[int, float]:
    """Shift k (frames) for which state[t + k] matches action[t] best; returns (k, mean abs error)."""
    errors = []
    for k in range(0, min(MAX_LAG_FRAMES, len(action) - 2) + 1):
        a = action[: len(action) - k]
        s = state[k:]
        errors.append(np.abs(a - s).mean())
    k = int(np.argmin(errors))
    return k, float(errors[k])


def lag_report(action: np.ndarray, state: np.ndarray, fps: int) -> list[str]:
    lines = []
    half = len(action) // 2
    for j, name in enumerate(JOINTS):
        a, s = action[:, j], state[:, j]
        motion = a.max() - a.min()
        if motion < 20:  # joint barely moved -> lag not measurable
            lines.append(f"    {name:<14} moved only {motion:5.0f} ticks, lag not measurable")
            continue
        k1, e1 = best_lag(a[:half], s[:half])
        k2, e2 = best_lag(a[half:], s[half:])
        lines.append(
            f"    {name:<14} lag 1st half {k1:2d} fr ({1000 * k1 / fps:4.0f} ms), "
            f"2nd half {k2:2d} fr ({1000 * k2 / fps:4.0f} ms) | rest offset {e1:5.1f} / {e2:5.1f} ticks "
            f"| motion {motion:5.0f} ticks"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="dataset folder name in local/ (default: newest with --prefix)")
    parser.add_argument("--prefix", default="nexarm_stack_boxes")
    parser.add_argument("--no-video", action="store_true", help="skip the (slower) duplicate-frame check")
    args = parser.parse_args()

    root = find_dataset(args.name, args.prefix)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    print(f"Dataset {root.name}: {info['total_episodes']} episodes, {info['total_frames']} frames, {fps} fps")

    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    episodes = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet")))
    cameras = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    for _, ep in episodes.sort_values("episode_index").iterrows():
        idx = int(ep["episode_index"])
        rows = data[data["episode_index"] == idx].sort_values("frame_index")
        action = np.stack(rows["action"].to_numpy()).astype(float)
        state = np.stack(rows["observation.state"].to_numpy()).astype(float)
        print(f"\nEpisode {idx}: {len(rows)} frames = {len(rows) / fps:.1f} s")

        if not args.no_video:
            for cam in cameras:
                chunk, file = int(ep[f"videos/{cam}/chunk_index"]), int(ep[f"videos/{cam}/file_index"])
                path = root / info["video_path"].format(video_key=cam, chunk_index=chunk, file_index=file)
                start, end = float(ep[f"videos/{cam}/from_timestamp"]), float(ep[f"videos/{cam}/to_timestamp"])
                frames = video_frames(path, start, end)
                share = duplicate_share(frames)
                print(
                    f"  {cam:<26} {len(frames)} video frames, {100 * share:4.1f} % duplicates "
                    f"(~{fps * (1 - share):.1f} real fps)"
                )

        print("  Follower lag behind leader (state vs. action):")
        for line in lag_report(action, state, fps):
            print(line)


if __name__ == "__main__":
    main()
