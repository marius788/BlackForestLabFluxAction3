"""Write calibration/follower/follower.json from the ranges measured on 2026-11-13 15:46.

That calibration run measured all six joints correctly but was rejected at the very end because
the gripper's physical travel (1094..2834) is slightly larger than the range the plugin accepts
for the follower gripper, raw [1195, 2833]. That range is exactly what the leader mapping can
command (bus.map_leader_to_follower clamps the gripper to it), so the gripper range is clamped
to it here. All other values are the measured ones. The file is validated and written by the
plugin's own save_calibration().

Usage (from the nexarm-f3a folder):  python tools/write_follower_calibration.py
"""

from pathlib import Path

from lerobot.motors import MotorCalibration
from lerobot_robot_nexarm.calibration import GRIPPER_MAX, GRIPPER_MIN, save_calibration

# name: (measured zero, measured min, measured max)
MEASURED = {
    "shoulder_pan": (2067, 262, 3861),
    "shoulder_lift": (2060, 964, 3279),
    "elbow_flex": (2065, 1085, 3969),
    "wrist_flex": (2041, 860, 3168),
    "wrist_roll": (2065, 1008, 3147),
    "gripper": (2830, 1094, 2834),
}

calibration = {}
for joint_id, (name, (zero, low, high)) in enumerate(MEASURED.items(), start=1):
    if name == "gripper":
        low, high = max(low, GRIPPER_MIN), min(high, GRIPPER_MAX)
        zero = min(high, max(low, zero))
    # drive_mode 0 = sign +. Directions are checked afterwards in teleoperation and set by hand.
    calibration[name] = MotorCalibration(
        id=joint_id, drive_mode=0, homing_offset=zero, range_min=low, range_max=high
    )

path = Path("calibration/follower/follower.json")
save_calibration(path, calibration, follower=True)
print(f"Written: {path.resolve()}")
print(path.read_text(encoding="utf-8"))
