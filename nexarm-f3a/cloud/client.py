"""Control the NexArm with FLUX 3 Action running behind cloud/server.py. Runs on the laptop.

Adopted from mariorch22/lerobot-nexarm (cloud/client.py, Apache-2.0) and extended with
full-chunk step/excursion safety checks and --port override.

    python cloud/client.py --task "put the small box on top of the big box" --fake-robot
    python cloud/client.py --task "..." --config config/inference.yaml
    python cloud/client.py --task "..." --config config/inference.yaml --execute

Without --execute nothing is sent to the arm: plans are only requested, checked and logged.
--fake-robot needs no arm and no cameras; it tests the server and the tunnel end to end.
--sim SCENE.xml drives a simulated NexArm in MuJoCo instead (see cloud/sim.py).

Loop: observe -> ask the server for 32 commands -> check the full plan -> send them at 30 Hz
while recording every tick -> repeat with the last 8 ticks as history. While the server is
planning, no command is sent and the arm holds its position.
"""

import argparse
import base64
import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_nexarm.bus import JOINT_NAMES

KEYS = [f"{name}.pos" for name in JOINT_NAMES]  # arm joints in degrees, gripper 0..100
HISTORY = 8  # consecutive control ticks the SO-101 checkpoint is conditioned on (n_obs_steps)


def encode_image(rgb: np.ndarray, quality: int = 90) -> str:
    """Squash to 256x256 (what the model uses anyway) and encode as base64 JPEG."""
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])  # OpenCV wants BGR
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(jpeg.tobytes()).decode()


def jump(state: list[float], command: list[float], arm_limit: float, gripper_limit: float) -> str | None:
    """Describe the first joint whose command is too far from the measured state, if any."""
    for name, measured, target in zip(JOINT_NAMES, state, command, strict=True):
        limit = gripper_limit if name == "gripper" else arm_limit
        if abs(target - measured) > limit:
            return f"{name}: measured {measured:.1f}, commanded {target:.1f} (limit {limit:g})"
    return None


def check_plan(
    state: list[float],
    commands: list[list[float]],
    arm_jump: float,
    gripper_jump: float,
    step_arm_jump: float = 10.0,
    step_gripper_jump: float = 25.0,
    chunk_arm_span: float = 45.0,
    chunk_gripper_span: float = 60.0,
) -> str | None:
    """Validate the full 32-command chunk before sending any action to the arm."""
    if not commands:
        return "server returned empty commands"
    problem = jump(state, commands[0], arm_jump, gripper_jump)
    if problem:
        return problem
    if step_arm_jump > 0 or step_gripper_jump > 0:
        for idx, (prev, nxt) in enumerate(zip(commands, commands[1:], strict=False)):
            for name, a, b in zip(JOINT_NAMES, prev, nxt, strict=True):
                limit = step_gripper_jump if name == "gripper" else step_arm_jump
                if limit > 0 and abs(b - a) > limit:
                    return (
                        f"step {idx} -> {idx + 1} {name}: {a:.1f} -> {b:.1f} "
                        f"(step limit {limit:g})"
                    )
    if chunk_arm_span > 0 or chunk_gripper_span > 0:
        for idx, cmd in enumerate(commands):
            for name, measured, target in zip(JOINT_NAMES, state, cmd, strict=True):
                limit = chunk_gripper_span if name == "gripper" else chunk_arm_span
                if limit > 0 and abs(target - measured) > limit:
                    return (
                        f"step {idx} {name}: measured {measured:.1f}, target {target:.1f} "
                        f"(chunk span limit {limit:g})"
                    )
    return None


def http_predict(url: str, timeout: float):
    """Return predict(task, frames) -> response dict for a running server."""

    def predict(task: str, frames: list[dict]) -> dict:
        body = json.dumps({"task": task, "frames": frames}).encode()
        request = urllib.request.Request(f"{url}/predict", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    return predict


class FakeRobot:
    """Stands in for the follower: grey images, and it 'moves' exactly to every command."""

    def __init__(self):
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
        self.image = np.full((480, 640, 3), 128, dtype=np.uint8)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self) -> dict:
        return {**dict(zip(KEYS, self.state, strict=True)), "scene": self.image, "wrist": self.image}

    def send_action(self, action: dict) -> dict:
        self.state = [float(action[key]) for key in KEYS]
        return action


class Client:
    def __init__(
        self,
        robot,
        predict,
        task: str,
        *,
        execute: bool = False,
        fps: float = 30.0,
        arm_jump: float = 15.0,
        gripper_jump: float = 30.0,
        step_arm_jump: float = 10.0,
        step_gripper_jump: float = 25.0,
        chunk_arm_span: float = 45.0,
        chunk_gripper_span: float = 60.0,
        jpeg_quality: int = 90,
        log_path: Path | None = None,
    ):
        self.robot, self.predict, self.task, self.execute = robot, predict, task, execute
        self.fps, self.arm_jump, self.gripper_jump = fps, arm_jump, gripper_jump
        self.step_arm_jump, self.step_gripper_jump = step_arm_jump, step_gripper_jump
        self.chunk_arm_span, self.chunk_gripper_span = chunk_arm_span, chunk_gripper_span
        self.jpeg_quality = jpeg_quality
        self.log_path = log_path
        self.frames = collections.deque(maxlen=HISTORY)  # the newest ticks, oldest first
        self.last_command = None  # what was really sent to the arm; stays None in dry-run

    def observe(self) -> list[float]:
        """Read the arm and both cameras and remember this tick for the next plan."""
        observation = self.robot.get_observation()
        state = [float(observation[key]) for key in KEYS]
        self.frames.append(
            {
                "image_scene": encode_image(observation["scene"], self.jpeg_quality),
                "image_wrist": encode_image(observation["wrist"], self.jpeg_quality),
                "state": state,
                "last_command": self.last_command,
            }
        )
        return state

    def log(self, **entry):
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")

    def run(self, duration: float | None = None) -> str:
        """Plan and execute until something stops the loop; return the reason."""
        end = None if duration is None else time.perf_counter() + duration
        state = self.observe()
        plan = 0
        while end is None or time.perf_counter() < end:
            start = time.perf_counter()
            try:
                answer = self.predict(self.task, list(self.frames))
            except OSError as exc:  # timeout, connection refused, HTTP error
                return f"server: {exc}"
            latency = time.perf_counter() - start
            commands = answer["commands"]
            plan += 1
            self.log(
                time=time.time(),
                plan=plan,
                task=self.task,
                executed=self.execute,
                latency_s=latency,
                server_s=answer.get("seconds"),
                state=state,
                first=commands[0] if commands else None,
                last=commands[-1] if commands else None,
            )
            problem = check_plan(
                state,
                commands,
                self.arm_jump,
                self.gripper_jump,
                self.step_arm_jump,
                self.step_gripper_jump,
                self.chunk_arm_span,
                self.chunk_gripper_span,
            )
            if problem:
                return f"jump guard rejected plan {plan}: {problem}"

            chunk_start = time.perf_counter()
            done = 0
            for step, command in enumerate(commands):
                if end is not None and time.perf_counter() >= end:
                    break
                done += 1
                if self.execute:
                    # The plugin clips to the calibrated range and returns what it really sent.
                    sent = self.robot.send_action(dict(zip(KEYS, command, strict=True)))
                    self.last_command = [float(sent[key]) for key in KEYS]
                precise_sleep(max(0.0, chunk_start + (step + 1) / self.fps - time.perf_counter()))
                state = self.observe()
            rate = done / max(time.perf_counter() - chunk_start, 1e-9)
            mode = "sent" if self.execute else "dry-run"
            print(
                f"plan {plan}: {latency:.2f} s, {mode} at {rate:.1f} Hz, "
                f"first {[round(v, 1) for v in commands[0]]}",
                flush=True,
            )
        return "duration reached"


def load_follower(config_path: Path, port: str | None = None):
    """Build the follower from the `robot` section of a LeRobot YAML. Opens no hardware yet."""
    import draccus
    from draccus.cfgparsing import load_config
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 (registers the camera type)
    from lerobot.robots import RobotConfig

    from lerobot_robot_nexarm import NexArmFollower
    from lerobot_robot_nexarm.calibration import load_calibration

    with config_path.open(encoding="utf-8") as stream:
        config = draccus.decode(RobotConfig, load_config(stream, file=config_path)["robot"])
    if port:
        config.port = port
    if config.calibration_dir is not None and not Path(config.calibration_dir).is_absolute():
        rel = Path(config.calibration_dir)
        if not (rel / f"{config.id}.json").is_file():
            for base in (config_path.resolve().parent.parent, config_path.resolve().parent):
                candidate = base / rel
                if (candidate / f"{config.id}.json").is_file():
                    config.calibration_dir = candidate
                    break
    missing = {"scene", "wrist"} - set(config.cameras)
    if missing:
        raise ValueError(f"{config_path}: robot.cameras must define 'scene' and 'wrist', missing {missing}")
    robot = NexArmFollower(config)
    # Fail here with a clear message; connect() would otherwise start the calibration dialog.
    load_calibration(robot.calibration_fpath)
    return robot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--task", required=True, help="English instruction for the model")
    parser.add_argument("--config", type=Path, help="LeRobot YAML with the robot section")
    parser.add_argument("--port", help="override robot.port from --config, e.g. COM10")
    parser.add_argument("--fake-robot", action="store_true", help="no arm, no cameras")
    parser.add_argument("--sim", type=Path, help="MuJoCo scene.xml of the NexArm model (see cloud/sim.py)")
    parser.add_argument("--video", type=Path, help="with --sim: save scene and wrist view as MP4")
    parser.add_argument("--execute", action="store_true", help="really send commands to the arm")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds (default 30)")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a plan (default 5)")
    parser.add_argument("--arm-jump", type=float, default=15.0, help="first-command limit in degrees (default 15)")
    parser.add_argument(
        "--gripper-jump", type=float, default=30.0, help="first-command gripper limit in percent points (default 30)"
    )
    parser.add_argument(
        "--step-arm-jump",
        type=float,
        default=10.0,
        help="max degrees between consecutive steps in a chunk, 0 disables (default 10)",
    )
    parser.add_argument(
        "--step-gripper-jump",
        type=float,
        default=25.0,
        help="max gripper delta between consecutive steps in a chunk, 0 disables (default 25)",
    )
    parser.add_argument(
        "--chunk-arm-span",
        type=float,
        default=45.0,
        help="max arm excursion from measured state across a chunk, 0 disables (default 45)",
    )
    parser.add_argument(
        "--chunk-gripper-span",
        type=float,
        default=60.0,
        help="max gripper excursion from measured state across a chunk, 0 disables (default 60)",
    )
    parser.add_argument("--log", type=Path, default=Path("outputs/cloud_client.jsonl"))
    args = parser.parse_args()
    if args.fake_robot and (args.config is not None or args.sim is not None):
        parser.error("--fake-robot cannot be combined with --config or --sim")
    if not args.fake_robot and args.config is None and args.sim is None:
        parser.error("give one of --config, --fake-robot or --sim (optional: --sim with --config)")

    if args.sim:
        cloud_dir = str(Path(__file__).resolve().parent)
        if cloud_dir not in sys.path:
            sys.path.insert(0, cloud_dir)
        from sim import MujocoRobot  # needs mujoco: uv run --project plugin --with mujoco python cloud/client.py ...

        robot = MujocoRobot(args.sim, args.video, config_path=args.config)
    else:
        robot = FakeRobot() if args.fake_robot else load_follower(args.config, port=args.port)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    client = Client(
        robot,
        http_predict(args.url, args.timeout),
        args.task,
        execute=args.execute,
        arm_jump=args.arm_jump,
        gripper_jump=args.gripper_jump,
        step_arm_jump=args.step_arm_jump,
        step_gripper_jump=args.step_gripper_jump,
        chunk_arm_span=args.chunk_arm_span,
        chunk_gripper_span=args.chunk_gripper_span,
        log_path=args.log,
    )
    print(
        "EXECUTE: commands are sent to the arm." if args.execute else "Dry-run: nothing is sent to the arm."
    )
    robot.connect()  # the real follower switches torque on and holds its position, also in dry-run
    try:
        print("Stopped:", client.run(args.duration))
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C")
    finally:
        robot.disconnect()  # the arm keeps holding its position (disable_torque_on_disconnect: false)


if __name__ == "__main__":
    main()
