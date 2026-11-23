"""Live raw encoder positions of one NexArm arm, torque OFF, nothing is moved or saved.

Use it before calibrating to find a start pose and to see where a joint crosses the encoder
boundary 4095 <-> 0 (the calibration aborts when a joint crosses it).

Usage (through 0-show-raw-follower.cmd / 0-show-raw-leader.cmd):
    python tools/show_raw.py follower COM24
    python tools/show_raw.py leader COM25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot_robot_nexarm.bus import JOINT_NAMES
from lerobot_robot_nexarm.config_nexarm import NexArmFollowerConfig, NexArmLeaderConfig
from lerobot_robot_nexarm.nexarm_follower import NexArmFollower
from lerobot_robot_nexarm.nexarm_leader import NexArmLeader

NEAR_WRAP = 300  # ticks from 0 / 4095 that count as "close to the encoder boundary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=["follower", "leader"])
    parser.add_argument("port")
    args = parser.parse_args()

    if args.role == "follower":
        device = NexArmFollower(
            NexArmFollowerConfig(port=args.port, id="follower", calibration_dir=Path("calibration/follower"))
        )
    else:
        device = NexArmLeader(
            NexArmLeaderConfig(
                port=args.port,
                id="leader",
                calibration_dir=Path("calibration/leader"),
                follower_calibration_path=Path("calibration/follower/follower.json"),
            )
        )

    # calibrate=False: same connection path as the calibration, torque is switched off, no actions.
    device.connect(calibrate=False)
    print(f"{args.role} on {args.port}, torque OFF. Raw positions 0..4095, middle = 2048.")
    print(f"'!' = closer than {NEAR_WRAP} ticks to the encoder boundary 4095 <-> 0. Quit with Ctrl+C.\n")
    print("  ".join(f"{name:>13}" for name in JOINT_NAMES))
    try:
        while True:
            positions = device.bus.read_positions()
            cells = []
            for raw in positions:
                mark = "!" if raw < NEAR_WRAP or raw > 4095 - NEAR_WRAP else " "
                cells.append(f"{raw:>12d}{mark}")
            print("\r" + "  ".join(cells), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
