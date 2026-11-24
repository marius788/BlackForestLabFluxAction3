"""A simulated NexArm follower in MuJoCo, as a stand-in robot for cloud/client.py.

Adopted from mariorch22/lerobot-nexarm (cloud/sim.py, Apache-2.0).

It tests the control loop with moving images and real physics. It does not show whether
a model solves the task on the real arm: the images are rendered.

The arm model is not part of this repo. Clone https://github.com/khanhthanhdev/lerobot-nexarm
into nexarm-f3a/third_party/khanhthanhdev-lerobot-nexarm (with
`git lfs pull --include "sim/assets/**"`) and pass its sim/description/mjcf/scene.xml.

Units follow the calibration used for the recordings of 13.11.2026: 0 degrees is Hiwonder's
home pose (upper arm vertical, forearm horizontal), gripper 0 is open and 100 is closed.
Checked by replaying a recorded episode next to its video: the poses agree once elbow_flex
runs opposite to the model's joint. The directions of shoulder_pan and wrist_roll were not
checked. The SO-101 checkpoint expects other values.
"""

import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lerobot_robot_nexarm.bus import JOINT_NAMES
from lerobot_robot_nexarm.calibration import DEGREES_PER_TICK, JointNormalizer, load_calibration

KEYS = [f"{name}.pos" for name in JOINT_NAMES]
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "inference.yaml"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "calibration" / "follower" / "follower.json"
SCHACH_DIR = Path(__file__).resolve().parents[2] / "Schach"
MJCF_DIR = (
    Path(__file__).resolve().parent.parent
    / "third_party"
    / "khanhthanhdev-lerobot-nexarm"
    / "sim"
    / "description"
    / "mjcf"
)
JOINTS = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
    "right_jaw_slide_joint",
]
JAW_OPEN = 0.0255  # metres of jaw travel when fully open
SIGN = [1, 1, -1, 1, -1]  # our degrees -> direction of the MuJoCo joint
OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0]  # our degrees in the CAD home pose
START = [0.0, -15.0, -15.0, 45.0, 0.0, 0.0]  # ready pose looking down at the workspace, gripper open
HEIGHT, WIDTH = 480, 640


def look_at(position, target) -> list[float]:
    """Camera x and y axes for a camera at `position` looking at `target`, z up."""
    forward = np.subtract(target, position)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    return [*right, *np.cross(right, forward)]


def resolve_scene_path(scene: Path) -> Path:
    """Sync Schach/*.xml into the MJCF folder when loading chess_scene.xml so includes resolve."""
    scene = Path(scene)
    if scene.name == "chess_scene.xml" and SCHACH_DIR.is_dir() and MJCF_DIR.is_dir():
        import shutil

        chess_sub = MJCF_DIR / "chess"
        chess_sub.mkdir(parents=True, exist_ok=True)
        if (SCHACH_DIR / "chess_scene.xml").is_file():
            shutil.copy2(SCHACH_DIR / "chess_scene.xml", MJCF_DIR / "chess_scene.xml")
        for sub_name in ("board.xml", "pieces.xml"):
            src = SCHACH_DIR / sub_name
            if src.is_file():
                shutil.copy2(src, chess_sub / sub_name)
        return MJCF_DIR / "chess_scene.xml"
    return scene


def build_model(scene: Path) -> mujoco.MjModel:
    """Load either the chess scene (chessboard + 32 pieces) or the box-stacking scene."""
    resolved = resolve_scene_path(scene)
    spec = mujoco.MjSpec.from_file(str(resolved))
    is_chess = spec.body("chessboard") is not None

    if is_chess:
        # Upgrade chess piece geoms with torsional friction & firm contact stiffness for grasping
        for geom in spec.geoms:
            if geom.name and (geom.name.startswith("w_") or geom.name.startswith("b_")):
                geom.condim = 4
                geom.solref = [0.004, 1.0]
                geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        position = [0.42, -0.68, 0.52]
        spec.worldbody.add_camera(
            name="scene", pos=position, xyaxes=look_at(position, [0.0, -0.25, 0.04]), fovy=46
        )
        return spec.compile()

    for key in list(spec.keys):  # they list the old objects' positions
        spec.delete(key)
    for name in ("cube", "target_zone"):
        body = spec.body(name)
        if body is not None:
            spec.delete(body)
    boxes = {
        "big_box": ([-0.07, -0.25, 0.02], [0.03, 0.03, 0.02], [0.92, 0.9, 0.85, 1]),
        "small_box": ([0.06, -0.25, 0.0125], [0.0125, 0.0125, 0.0125], [0.95, 0.75, 0.1, 1]),
    }
    for name, (position, half_size, colour) in boxes.items():
        body = spec.worldbody.add_body(name=name, pos=position)
        body.add_freejoint(name=f"{name}_joint")
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_size,
            rgba=colour,
            mass=0.02,
            friction=[1.4, 0.01, 0.001],
            condim=4,
            solref=[0.004, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            contype=2,
            conaffinity=3,
        )
    position = [0.45, -0.7, 0.55]  # fixed camera, roughly where the lab's scene camera stands
    spec.worldbody.add_camera(
        name="scene", pos=position, xyaxes=look_at(position, [0.0, -0.15, 0.1]), fovy=50
    )
    return spec.compile()


def load_normalizer(
    config_path: Path | None = None,
    calibration_path: Path | None = None,
) -> tuple[JointNormalizer | None, Path | None]:
    """Load the follower's JointNormalizer from config/inference.yaml or follower.json."""
    use_degrees = True
    cal_file = calibration_path
    cfg_file = config_path if config_path is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)

    if cfg_file is not None and cfg_file.is_file():
        from draccus.cfgparsing import load_config

        with cfg_file.open(encoding="utf-8") as stream:
            raw_cfg = load_config(stream, file=cfg_file)
        robot_cfg = raw_cfg.get("robot", raw_cfg) if isinstance(raw_cfg, dict) else {}
        use_degrees = bool(robot_cfg.get("use_degrees", True))
        if cal_file is None:
            cal_dir = Path(robot_cfg.get("calibration_dir", "calibration/follower"))
            robot_id = str(robot_cfg.get("id", "follower"))
            if not cal_dir.is_absolute():
                for base in (cfg_file.resolve().parent.parent, cfg_file.resolve().parent, Path.cwd()):
                    candidate = base / cal_dir / f"{robot_id}.json"
                    if candidate.is_file():
                        cal_file = candidate
                        break

    if cal_file is None and DEFAULT_CALIBRATION.is_file():
        cal_file = DEFAULT_CALIBRATION

    if cal_file is not None and cal_file.is_file():
        calibration = load_calibration(cal_file, follower=True)
        return JointNormalizer(calibration, use_degrees=use_degrees), cal_file
    return None, None


class MujocoRobot:
    """Same interface as the follower: degrees and gripper value in, images and state out."""

    def __init__(
        self,
        scene: Path,
        video: Path | None = None,
        fps: float = 30.0,
        config_path: Path | None = None,
        calibration_path: Path | None = None,
    ):
        self.model = build_model(scene)
        self.data = mujoco.MjData(self.model)
        self.video, self.fps = video, fps
        self.steps = max(1, round(1 / fps / self.model.opt.timestep))  # physics steps per control tick
        self.model.opt.timestep = (1.0 / fps) / self.steps
        self.qpos = [self.model.joint(name).qposadr[0] for name in JOINTS]
        self.limits = self.model.actuator_ctrlrange.copy()  # actuators are in joint order
        self.normalizer, self.calibration_path = load_normalizer(config_path, calibration_path)
        self.renderer = None
        self.frames = []

    def to_sim(self, values: list[float]) -> np.ndarray:
        """Model units -> radians and jaw travel, clipped to both follower calibration and MuJoCo limits."""
        if self.normalizer is not None:
            targets = []
            for name, v, s, o in zip(JOINT_NAMES[:5], values[:5], SIGN, OFFSET, strict=True):
                c = self.normalizer.calibration[name]
                raw = self.normalizer.unnormalize(name, float(v))
                physical_deg = (raw - c.homing_offset) * DEGREES_PER_TICK
                targets.append(math.radians(s * (physical_deg - o)))
            cg = self.normalizer.calibration["gripper"]
            raw_g = self.normalizer.unnormalize("gripper", float(values[5]))
            closed_ratio = (raw_g - cg.range_min) / (cg.range_max - cg.range_min)
            targets.append((1.0 - closed_ratio) * JAW_OPEN)
            return np.clip(targets, self.limits[:, 0], self.limits[:, 1])

        targets = [math.radians(s * (v - o)) for v, s, o in zip(values[:5], SIGN, OFFSET, strict=True)]
        targets.append((1 - values[5] / 100) * JAW_OPEN)
        return np.clip(targets, self.limits[:, 0], self.limits[:, 1])

    def from_sim(self, positions) -> list[float]:
        """Radians and jaw travel -> model units (via follower JointNormalizer when available)."""
        if self.normalizer is not None:
            values = []
            for name, p, s, o in zip(JOINT_NAMES[:5], positions[:5], SIGN, OFFSET, strict=True):
                c = self.normalizer.calibration[name]
                physical_deg = s * math.degrees(float(p)) + o
                raw = c.homing_offset + physical_deg / DEGREES_PER_TICK
                values.append(self.normalizer.normalize(name, raw))
            cg = self.normalizer.calibration["gripper"]
            closed_ratio = 1.0 - float(positions[5]) / JAW_OPEN
            raw_g = cg.range_min + closed_ratio * (cg.range_max - cg.range_min)
            values.append(self.normalizer.normalize("gripper", raw_g))
            return values

        values = [s * math.degrees(p) + o for p, s, o in zip(positions[:5], SIGN, OFFSET, strict=True)]
        return [*values, (1 - float(positions[5]) / JAW_OPEN) * 100]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        start = self.to_sim(START)
        self.data.qpos[self.qpos] = start
        self.data.qpos[self.model.joint("left_jaw_slide_joint").qposadr[0]] = start[5]
        self.data.ctrl[:] = start
        mujoco.mj_step(self.model, self.data, 500)  # let arm and boxes settle

    def connect(self):
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.reset()

    def disconnect(self):
        if self.renderer is not None:
            self.renderer.close()
        if self.video is not None and self.frames:
            self.video.parent.mkdir(parents=True, exist_ok=True)
            height, width = self.frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            for frame in self.frames:
                writer.write(frame[..., ::-1])  # OpenCV wants BGR
            writer.release()
            print(f"Video: {self.video} ({len(self.frames)} frames)")

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def get_observation(self) -> dict:
        state = self.from_sim(self.data.qpos[self.qpos])
        scene, wrist = self.render("scene"), self.render("wrist")
        if self.video is not None:
            self.frames.append(np.hstack([scene, wrist]))
        return {**dict(zip(KEYS, state, strict=True)), "scene": scene, "wrist": wrist}

    def send_action(self, action: dict) -> dict:
        targets = self.to_sim([float(action[key]) for key in KEYS])
        start_ctrl = self.data.ctrl.copy()
        for step in range(1, self.steps + 1):
            alpha = step / self.steps
            self.data.ctrl[:] = start_ctrl + (targets - start_ctrl) * alpha
            mujoco.mj_step(self.model, self.data)
        return dict(zip(KEYS, self.from_sim(targets), strict=True))  # really commanded, after clipping
