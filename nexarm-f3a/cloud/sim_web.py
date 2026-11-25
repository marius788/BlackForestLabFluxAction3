# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn", "mujoco", "opencv-python", "numpy"]
# ///
"""Interactive browser dashboard for the MuJoCo NexArm simulation + FLUX.3 Action.

Runs a persistent 30 Hz MuJoCo simulation loop on a dedicated thread (so the OpenGL
renderer context stays thread-local on Windows), streams the live scene + wrist cameras
as MJPEG, lets you send natural-language task prompts from the browser, and visualizes
the model's 32-step action chunk response, safety-guard checks, and joint trajectories.

Usage:
    uv run --project plugin --with mujoco --with fastapi --with uvicorn python cloud/sim_web.py
"""

from __future__ import annotations

import argparse
import collections
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

CLOUD_DIR = Path(__file__).resolve().parent
if str(CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUD_DIR))

from client import HISTORY, KEYS, JOINT_NAMES, check_plan, encode_image, http_predict  # noqa: E402

SCHACH_SCENE = CLOUD_DIR.parents[1] / "Schach" / "chess_scene.xml"
BOX_SCENE = (
    CLOUD_DIR.parent
    / "third_party"
    / "khanhthanhdev-lerobot-nexarm"
    / "sim"
    / "description"
    / "mjcf"
    / "scene.xml"
)
DEFAULT_SCENE = SCHACH_SCENE if SCHACH_SCENE.is_file() else BOX_SCENE
FILES = "abcdefgh"


def xy_to_square(x: float, y: float, camera_cfg: dict | None = None) -> str:
    """Convert MuJoCo world (x, y) on the NexArm chessboard to algebraic notation."""
    if camera_cfg and camera_cfg.get("real_board_layout"):
        b = camera_cfg.get("board", {})
        cx, cy = b.get("center_xy", [0.041, -0.2415])
        sq = float(b.get("square_size", 0.0355))
        file_idx = round(3.5 - (x - cx) / sq)
        rank_idx = round(3.5 - (y - cy) / sq)
    else:
        file_idx = round((x + 0.0875) / 0.025)
        rank_idx = round(((y + 0.25) + 0.0875) / 0.025)
    if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
        return f"{FILES[file_idx]}{rank_idx + 1}"
    return f"({x:+.2f},{y:+.2f})"


def square_to_xy(sq_name: str, camera_cfg: dict | None = None) -> tuple[float, float, float]:
    f_idx = FILES.index(sq_name[0])
    r_idx = int(sq_name[1]) - 1
    if camera_cfg and camera_cfg.get("real_board_layout"):
        b = camera_cfg.get("board", {})
        cx, cy = b.get("center_xy", [0.041, -0.2415])
        sq = float(b.get("square_size", 0.0355))
        return cx + (3.5 - f_idx) * sq, cy + (3.5 - r_idx) * sq, float(b.get("surface_z", 0.056))
    return -0.0875 + f_idx * 0.025, -0.3375 + r_idx * 0.025, 0.01


class ControlRequest(BaseModel):
    task: str | None = None
    mode: str | None = Field(
        default=None,
        description="One of: 'run', 'once', 'dryrun', 'pause'",
    )
    backend: str | None = Field(
        default=None,
        description="One of: 'server', 'demo_stack', 'demo_sweep'",
    )
    server_url: str | None = None


class ManualPoseRequest(BaseModel):
    targets: list[float] = Field(min_length=6, max_length=6)


class GuardsRequest(BaseModel):
    arm_jump: float = 15.0
    gripper_jump: float = 30.0
    step_arm_jump: float = 10.0
    step_gripper_jump: float = 25.0
    chunk_arm_span: float = 45.0
    chunk_gripper_span: float = 60.0
    bypass: bool = False


class ResetRequest(BaseModel):
    randomize_boxes: bool = False


def make_demo_sweep_chunk(state: list[float], tick_time: float, n_steps: int = 32) -> list[list[float]]:
    """Generate a smooth 32-step test motion over the front workspace without needing a GPU server."""
    poses = (
        [24.0, -24.0, -10.0, 50.0, 0.0, 75.0],
        [-24.0, -16.0, -15.0, 42.0, 0.0, 10.0],
    )
    target = poses[int(tick_time / 4.5) % 2]
    step_max = [0.45, 0.4, 0.45, 0.45, 0.45, 1.7]  # 32 * 1.7 = 54.4 < 60.0 chunk_gripper_span
    pos = list(state)
    commands: list[list[float]] = []
    for _ in range(n_steps):
        pos = [p + max(-s, min(s, t - p)) for p, t, s in zip(pos, target, step_max, strict=True)]
        commands.append([round(v, 3) for v in pos])
    return commands


def make_demo_stack_chunk(
    state: list[float], phase_idx: int, n_steps: int = 32, *, chess: bool = True
) -> tuple[list[list[float]], int]:
    """Multi-phase demonstration trajectory: Pawn e2->e4 on chess scene, or box stack on box scene."""
    if chess:
        waypoints = [
            # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
            [-2.5, -43.0, 19.0, 65.0, 0.0, 48.0],   # 0: pre-grasp above e2 (gripper 48% clears d2 & f2)
            [-2.5, -58.0, 24.0, 65.0, 0.0, 48.0],   # 1: descend onto w_pawn_e2
            [-2.5, -58.0, 24.0, 65.0, 0.0, 82.0],   # 2: close gripper on pawn
            [-2.5, -40.0, 18.0, 65.0, 0.0, 82.0],   # 3: lift pawn straight up
            [-2.5, -38.0, 3.0, 65.0, 0.0, 82.0],    # 4: carry over e4
            [-2.5, -52.5, 8.0, 65.0, 0.0, 82.0],    # 5: lower onto e4
            [-2.5, -52.5, 8.0, 65.0, 0.0, 48.0],    # 6: release gripper
            [0.0, -15.0, -15.0, 45.0, 0.0, 0.0],    # 7: retract home
        ]
    else:
        waypoints = [
            [-14.0, -34.0, -5.0, 52.0, 0.0, 0.0],   # 0: pre-grasp above small box
            [-14.0, -51.0, -5.0, 50.0, 0.0, 0.0],   # 1: descend around small box
            [-14.0, -51.0, -5.0, 50.0, 0.0, 68.0],  # 2: close gripper on small box
            [-14.0, -28.0, -5.0, 48.0, 0.0, 68.0],  # 3: lift small box
            [15.5, -28.0, -5.0, 48.0, 0.0, 68.0],   # 4: carry over big box
            [15.5, -41.5, -5.0, 48.0, 0.0, 68.0],   # 5: lower onto big box
            [15.5, -41.5, -5.0, 48.0, 0.0, 0.0],    # 6: release gripper
            [0.0, -15.0, -15.0, 45.0, 0.0, 0.0],    # 7: retract home
        ]
    target = waypoints[phase_idx % len(waypoints)]
    step_max = [0.45, 0.45, 0.45, 0.45, 0.45, 1.7]  # 32 * 1.7 = 54.4 < 60.0 chunk_gripper_span
    pos = list(state)
    commands: list[list[float]] = []
    for _ in range(n_steps):
        pos = [p + max(-s, min(s, t - p)) for p, t, s in zip(pos, target, step_max, strict=True)]
        commands.append([round(v, 3) for v in pos])
    dist = max(abs(a - b) for a, b in zip(pos[:5], target[:5], strict=True))
    grip_dist = abs(pos[5] - target[5])
    next_phase = (phase_idx + 1) % len(waypoints) if (dist < 1.2 and grip_dist < 3.0) else phase_idx
    return commands, next_phase


class SimWebController:
    """Manages the MuJoCo robot, 30 Hz control loop, MJPEG frames, and model predictions."""

    def __init__(
        self,
        robot,
        *,
        server_url: str = "http://127.0.0.1:8000",
        task: str = "put the small box on top of the big box",
        fps: float = 30.0,
        timeout: float = 25.0,
        connect_on_MainThread: bool = False,
    ):
        self.robot = robot
        self.server_url = server_url.rstrip("/")
        self.task = task
        self.fps = fps
        self.timeout = timeout

        self._lock = threading.Lock()
        self._cmd_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._frame_cond = threading.Condition()

        # Shared telemetry state (protected by self._lock)
        self.mode: str = "pause"  # 'pause', 'run', 'once', 'dryrun'
        self.backend: str = "server"  # 'server', 'demo_stack', 'demo_sweep'
        self.status: str = "idle"  # 'idle', 'planning', 'executing', 'dryrun_done', 'rejected', 'error'
        self.status_detail: str = "Ready — enter a prompt and press Run or Step 1 Chunk."
        self.state: list[float] = [0.0, 10.0, -15.0, 15.0, 0.0, 0.0]
        self.last_command: list[float] | None = None
        self.tick_count: int = 0
        self.sim_fps: float = 0.0
        self.plan_count: int = 0
        self.current_step: int = 0
        self.total_steps: int = 32
        self.box_positions: dict[str, list[float]] = {}

        self.guards: dict[str, Any] = {
            "arm_jump": 15.0,
            "gripper_jump": 30.0,
            "step_arm_jump": 10.0,
            "step_gripper_jump": 25.0,
            "chunk_arm_span": 45.0,
            "chunk_gripper_span": 60.0,
            "bypass": False,
        }

        self.server_health: dict[str, Any] = {
            "status": "unknown",
            "model": None,
            "error": None,
            "checked_at": 0.0,
        }
        self._health_failures: int = 0

        self.last_plan: dict[str, Any] | None = None
        self.plan_history: collections.deque[dict[str, Any]] = collections.deque(maxlen=20)
        self.state_history: collections.deque[list[float]] = collections.deque(maxlen=HISTORY)

        # Latest encoded JPEG images
        self._latest_jpeg_dual: bytes = b""
        self._latest_jpeg_scene: bytes = b""
        self._latest_jpeg_wrist: bytes = b""
        self._latest_jpeg_top: bytes = b""
        self._frame_seq: int = 0
        self.is_chess: bool = self._detect_chess()

        # Internal loop state (owned by worker thread)
        self._obs_frames: collections.deque[dict[str, Any]] = collections.deque(maxlen=HISTORY)
        self._pending_commands: list[list[float]] = []
        self._pending_step_idx: int = 0
        self._demo_phase: int = 0
        self._rng = np.random.default_rng(42)

        if connect_on_MainThread:
            self.robot.connect()
            self._capture_observation()

    def _detect_chess(self) -> bool:
        if hasattr(self.robot, "model"):
            try:
                self.robot.model.body("chessboard")
                return True
            except Exception:
                return False
        return False

    def start(self):
        self._worker = threading.Thread(target=self._run_loop, name="SimWebLoop", daemon=True)
        self._worker.start()
        self._health_worker = threading.Thread(target=self._health_loop, name="HealthPoll", daemon=True)
        self._health_worker.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def enqueue(self, action: str, payload: Any = None):
        self._cmd_queue.put((action, payload))

    def _health_loop(self):
        while not self._stop_event.is_set():
            self.check_server_health()
            self._stop_event.wait(2.5)

    def check_server_health(self) -> dict[str, Any]:
        with self._lock:
            url = self.server_url
            backend = self.backend
            is_planning = self.status == "planning"
            current_health = dict(self.server_health)
        if backend != "server":
            info = {
                "status": "ready",
                "model": "Built-in Demo Stacker" if backend == "demo_stack" else "Built-in Demo Sweep",
                "error": None,
                "checked_at": time.time(),
            }
            with self._lock:
                self.server_health = info
                self._health_failures = 0
            return info
        # Avoid competing with an in-flight /predict over the SSH tunnel
        if is_planning and current_health.get("status") == "ready":
            return current_health
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.load(resp)
            info = {
                "status": data.get("status", "ready"),
                "model": data.get("model") or "FLUX.3 Action",
                "error": data.get("error"),
                "checked_at": time.time(),
            }
            with self._lock:
                self._health_failures = 0
                self.server_health = info
            return info
        except Exception as exc:
            with self._lock:
                self._health_failures += 1
                # Only flip from ready -> offline after 2 consecutive timeouts
                if self.server_health.get("status") == "ready" and self._health_failures < 2:
                    return dict(self.server_health)
                info = {
                    "status": "offline",
                    "model": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "checked_at": time.time(),
                }
                self.server_health = info
            return info

    def _read_box_positions(self) -> dict[str, Any]:
        positions: dict[str, Any] = {}
        if hasattr(self.robot, "model") and hasattr(self.robot, "data"):
            if self.is_chess:
                squares: dict[str, str] = {}
                for piece_name in ("w_pawn_e2", "w_pawn_d2", "w_knight_g1", "b_pawn_e7", "b_pawn_d7"):
                    try:
                        body_id = self.robot.model.body(piece_name).id
                        pos = self.robot.data.xpos[body_id]
                        positions[piece_name] = [
                            round(float(pos[0]), 4),
                            round(float(pos[1]), 4),
                            round(float(pos[2]), 4),
                        ]
                        squares[piece_name] = xy_to_square(
                            float(pos[0]), float(pos[1]), getattr(self.robot, "camera_cfg", None)
                        )
                    except Exception:
                        pass
                positions["squares"] = squares
            else:
                for box_name in ("small_box", "big_box"):
                    try:
                        body_id = self.robot.model.body(box_name).id
                        pos = self.robot.data.xpos[body_id]
                        positions[box_name] = [
                            round(float(pos[0]), 4),
                            round(float(pos[1]), 4),
                            round(float(pos[2]), 4),
                        ]
                    except Exception:
                        pass
        return positions

    def _reset_sim(self, randomize_boxes: bool = False):
        if hasattr(self.robot, "model") and hasattr(self.robot, "data"):
            import mujoco
            from sim import START

            mujoco.mj_resetData(self.robot.model, self.robot.data)
            start = self.robot.to_sim(START)
            self.robot.data.qpos[self.robot.qpos] = start
            self.robot.data.qpos[self.robot.model.joint("left_jaw_slide_joint").qposadr[0]] = start[5]
            self.robot.data.ctrl[:] = start

            if randomize_boxes:
                if self.is_chess:
                    # Apply a randomized chess opening setup (advance 2 pawns to rank 4 or 5)
                    cam_cfg = getattr(self.robot, "camera_cfg", None)
                    candidates = [
                        ("w_pawn_e2", "e4"),
                        ("w_pawn_d2", "d4"),
                        ("w_pawn_c2", "c4"),
                        ("b_pawn_e7", "e5"),
                        ("b_pawn_c7", "c5"),
                        ("b_pawn_d7", "d5"),
                    ]
                    chosen_idx = self._rng.choice(len(candidates), size=2, replace=False)
                    for idx in chosen_idx:
                        pname, sq_target = candidates[int(idx)]
                        px, py, pz = square_to_xy(sq_target, cam_cfg)
                        try:
                            jnt_id = self.robot.model.body(pname).jntadr[0]
                            qadr = self.robot.model.jnt_qposadr[jnt_id]
                            self.robot.data.qpos[qadr : qadr + 3] = [px, py, pz]
                            self.robot.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                        except Exception:
                            pass
                else:
                    # Randomize box positions within reachable front workspace
                    big_x = float(self._rng.uniform(-0.10, -0.03))
                    big_y = float(self._rng.uniform(-0.28, -0.21))
                    small_x = float(self._rng.uniform(0.03, 0.10))
                    small_y = float(self._rng.uniform(-0.28, -0.21))
                    for name, (bx, by, bz) in (
                        ("big_box", (big_x, big_y, 0.02)),
                        ("small_box", (small_x, small_y, 0.0125)),
                    ):
                        try:
                            jnt_id = self.robot.model.body(name).jntadr[0]
                            qadr = self.robot.model.jnt_qposadr[jnt_id]
                            self.robot.data.qpos[qadr : qadr + 3] = [bx, by, bz]
                            self.robot.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                        except Exception:
                            pass

            mujoco.mj_step(self.robot.model, self.robot.data, 500)
        elif hasattr(self.robot, "reset"):
            self.robot.reset()

        self._obs_frames.clear()
        self._pending_commands.clear()
        self._pending_step_idx = 0
        self._demo_phase = 0
        with self._lock:
            self.last_command = None
            self.current_step = 0
            self.mode = "pause"
            self.status = "idle"
            self.status_detail = (
                "Board randomized." if (randomize_boxes and self.is_chess)
                else ("Scene reset with randomized boxes." if randomize_boxes else "Scene reset to home pose.")
            )
            self.state_history.clear()
        self._capture_observation()

    def _capture_observation(self) -> list[float]:
        obs = self.robot.get_observation()
        state = [round(float(obs[k]), 3) for k in KEYS]
        scene_rgb = obs["scene"]
        wrist_rgb = obs["wrist"]

        top_rgb = None
        if hasattr(self.robot, "render") and hasattr(self.robot, "model"):
            try:
                self.robot.model.camera("top")
                top_rgb = self.robot.render("top")
            except Exception:
                top_rgb = None

        with self._lock:
            last_cmd = list(self.last_command) if self.last_command is not None else None

        self._obs_frames.append(
            {
                "image_scene": encode_image(scene_rgb, 90),
                "image_wrist": encode_image(wrist_rgb, 90),
                "state": state,
                "last_command": last_cmd,
            }
        )

        dual_rgb = np.hstack([scene_rgb, wrist_rgb])
        ok_d, buf_d = cv2.imencode(".jpg", dual_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok_s, buf_s = cv2.imencode(".jpg", scene_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok_w, buf_w = cv2.imencode(".jpg", wrist_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok_t, buf_t = (
            cv2.imencode(".jpg", top_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if top_rgb is not None
            else (False, None)
        )

        box_pos = self._read_box_positions()

        with self._lock:
            self.state = state
            self.state_history.append(state)
            self.tick_count += 1
            self.box_positions = box_pos
            if ok_d:
                self._latest_jpeg_dual = buf_d.tobytes()
            if ok_s:
                self._latest_jpeg_scene = buf_s.tobytes()
            if ok_w:
                self._latest_jpeg_wrist = buf_w.tobytes()
            if ok_t and buf_t is not None:
                self._latest_jpeg_top = buf_t.tobytes()
            self._frame_seq += 1

        with self._frame_cond:
            self._frame_cond.notify_all()

        return state

    def _request_plan(self, state: list[float]) -> tuple[list[list[float]] | None, float, float | None, str | None]:
        """Request 32 action steps from the active backend. Returns (commands, latency_s, server_s, error)."""
        with self._lock:
            task = self.task
            backend = self.backend
            url = self.server_url
            frames = list(self._obs_frames)
            self.status = "planning"
            self.status_detail = f"Querying {backend} for task: \"{task}\"..."

        start = time.perf_counter()
        try:
            if backend == "demo_sweep":
                time.sleep(0.03)
                commands = make_demo_sweep_chunk(state, time.monotonic())
                server_s = 0.03
            elif backend == "demo_stack":
                time.sleep(0.03)
                commands, self._demo_phase = make_demo_stack_chunk(
                    state, self._demo_phase, chess=self.is_chess
                )
                server_s = 0.03
            else:
                predict_fn = http_predict(url, self.timeout)
                answer = predict_fn(task, frames)
                commands = [[round(float(v), 3) for v in cmd] for cmd in answer["commands"]]
                server_s = answer.get("seconds")
        except Exception as exc:
            return None, time.perf_counter() - start, None, f"{type(exc).__name__}: {exc}"

        latency = time.perf_counter() - start
        return commands, latency, server_s, None

    def step_once_in_thread(self):
        """Process queued commands and advance one control tick. Useful for unit testing."""
        self._process_commands()
        self._step_control()

    def _process_commands(self):
        while True:
            try:
                action, payload = self._cmd_queue.get_nowait()
            except queue.Empty:
                break

            if action == "reset":
                self._reset_sim(randomize_boxes=bool(payload))
            elif action == "manual_pose":
                targets = [float(v) for v in payload]
                self._pending_commands.clear()
                self._pending_step_idx = 0
                sent = self.robot.send_action(dict(zip(KEYS, targets, strict=True)))
                with self._lock:
                    self.mode = "pause"
                    self.status = "idle"
                    self.status_detail = "Manual joint pose applied."
                    self.last_command = [round(float(sent[k]), 3) for k in KEYS]
                self._capture_observation()
            elif action == "control":
                req: ControlRequest = payload
                with self._lock:
                    if req.task is not None:
                        self.task = req.task.strip() or self.task
                    if req.backend is not None:
                        self.backend = req.backend
                    if req.server_url is not None:
                        self.server_url = req.server_url.rstrip("/")
                    if req.mode is not None:
                        self.mode = req.mode
                        if req.mode == "pause":
                            self._pending_commands.clear()
                            self._pending_step_idx = 0
                            self.status = "idle"
                            self.status_detail = "Paused."
                        elif req.mode in ("once", "dryrun"):
                            # Trigger an immediate fresh plan
                            self._pending_commands.clear()
                            self._pending_step_idx = 0

    def _step_control(self):
        with self._lock:
            mode = self.mode
            guards = dict(self.guards)
            task = self.task
            backend = self.backend
            history_snapshot = list(self.state_history)

        if mode == "pause":
            # Keep camera stream fresh without stepping actions
            if not self._obs_frames:
                self._capture_observation()
            return

        # If we have an active chunk being executed, send the next step
        if self._pending_commands and self._pending_step_idx < len(self._pending_commands):
            cmd = self._pending_commands[self._pending_step_idx]
            self._pending_step_idx += 1
            sent = self.robot.send_action(dict(zip(KEYS, cmd, strict=True)))
            with self._lock:
                self.last_command = [round(float(sent[k]), 3) for k in KEYS]
                self.current_step = self._pending_step_idx
                self.total_steps = len(self._pending_commands)
                self.status = "executing"
                self.status_detail = (
                    f"Executing step {self._pending_step_idx}/{len(self._pending_commands)} "
                    f"for \"{task}\""
                )
            self._capture_observation()

            # Check if chunk finished
            if self._pending_step_idx >= len(self._pending_commands):
                self._pending_commands.clear()
                self._pending_step_idx = 0
                if mode == "once":
                    with self._lock:
                        self.mode = "pause"
                        self.status = "idle"
                        self.status_detail = f"Completed 1 chunk ({self.total_steps} steps)."
            return

        # Otherwise we need a new plan from the model
        if not self._obs_frames:
            state = self._capture_observation()
        else:
            state = list(self.state)

        commands, latency, server_s, err = self._request_plan(state)
        if err is not None or commands is None:
            with self._lock:
                self.mode = "pause"
                self.status = "error"
                self.status_detail = f"Model request failed: {err}"
            return

        problem = None
        if not guards.get("bypass", False):
            problem = check_plan(
                state,
                commands,
                guards["arm_jump"],
                guards["gripper_jump"],
                guards["step_arm_jump"],
                guards["step_gripper_jump"],
                guards["chunk_arm_span"],
                guards["chunk_gripper_span"],
            )

        with self._lock:
            self.plan_count += 1
            plan_num = self.plan_count

        # Compute per-joint deltas & stats for the UI inspector
        joint_deltas = [round(commands[-1][i] - state[i], 2) for i in range(6)]
        first_jumps = [round(commands[0][i] - state[i], 2) for i in range(6)]
        max_excursions = [
            round(max(abs(cmd[i] - state[i]) for cmd in commands), 2) for i in range(6)
        ]

        plan_record = {
            "plan": plan_num,
            "timestamp": time.time(),
            "task": task,
            "backend": backend,
            "mode": mode,
            "latency_s": round(latency, 3),
            "server_s": round(server_s, 3) if server_s is not None else None,
            "state_at_plan": state,
            "history_states": history_snapshot,
            "commands": commands,
            "first_command": commands[0] if commands else None,
            "last_command": commands[-1] if commands else None,
            "joint_deltas": joint_deltas,
            "first_jumps": first_jumps,
            "max_excursions": max_excursions,
            "guard_ok": problem is None,
            "guard_problem": problem,
            "executed": mode in ("run", "once") and problem is None,
        }

        with self._lock:
            self.last_plan = plan_record
            self.plan_history.appendleft(plan_record)
            self.total_steps = len(commands)
            self.current_step = 0

        if problem is not None:
            with self._lock:
                self.mode = "pause"
                self.status = "rejected"
                self.status_detail = f"Guard rejected plan #{plan_num}: {problem}"
            return

        if mode == "dryrun":
            with self._lock:
                self.mode = "pause"
                self.status = "dryrun_done"
                self.status_detail = (
                    f"Dry-Run Plan #{plan_num} received ({len(commands)} steps in {latency*1000:.0f} ms). "
                    "Arm held in place."
                )
            return

        # Queue commands for 30 Hz execution
        self._pending_commands = commands
        self._pending_step_idx = 0

    def _run_loop(self):
        self.robot.connect()
        try:
            self._capture_observation()
            ema_fps = self.fps
            last_tick = time.perf_counter()
            while not self._stop_event.is_set():
                tick_start = time.perf_counter()
                dt = tick_start - last_tick
                last_tick = tick_start
                if dt > 1e-4:
                    ema_fps = 0.85 * ema_fps + 0.15 * (1.0 / dt)
                    with self._lock:
                        self.sim_fps = round(ema_fps, 1)

                self._process_commands()
                self._step_control()

                elapsed = time.perf_counter() - tick_start
                sleep_s = max(0.002, (1.0 / self.fps) - elapsed)
                self._stop_event.wait(sleep_s)
        finally:
            self.robot.disconnect()

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task": self.task,
                "mode": self.mode,
                "backend": self.backend,
                "server_url": self.server_url,
                "status": self.status,
                "status_detail": self.status_detail,
                "state": list(self.state),
                "last_command": list(self.last_command) if self.last_command else None,
                "joint_names": list(JOINT_NAMES),
                "tick_count": self.tick_count,
                "sim_fps": self.sim_fps,
                "plan_count": self.plan_count,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "is_chess": self.is_chess,
                "box_positions": dict(self.box_positions),
                "guards": dict(self.guards),
                "server_health": dict(self.server_health),
                "last_plan": self.last_plan,
                "plan_history": [
                    {
                        "plan": p["plan"],
                        "timestamp": p["timestamp"],
                        "task": p["task"],
                        "backend": p["backend"],
                        "mode": p["mode"],
                        "latency_s": p["latency_s"],
                        "server_s": p["server_s"],
                        "guard_ok": p["guard_ok"],
                        "guard_problem": p["guard_problem"],
                        "executed": p["executed"],
                        "first_command": p["first_command"],
                        "last_command": p["last_command"],
                        "joint_deltas": p["joint_deltas"],
                    }
                    for p in self.plan_history
                ],
            }

    def get_jpeg(self, view: str = "dual") -> bytes:
        with self._lock:
            if view == "scene":
                return self._latest_jpeg_scene
            if view == "wrist":
                return self._latest_jpeg_wrist
            if view == "top" and self._latest_jpeg_top:
                return self._latest_jpeg_top
            return self._latest_jpeg_dual


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NexArm — FLUX.3 Action</title>
<style>
  :root {
    --bg: #09090b;
    --surface: #0f0f12;
    --surface-2: #16161a;
    --surface-3: #1c1c22;
    --border: #232329;
    --border-subtle: #1a1a1f;
    --text: #f4f4f5;
    --text-secondary: #a1a1aa;
    --text-muted: #63636e;
    --primary: #fafafa;
    --primary-fg: #09090b;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --mono: 'JetBrains Mono', 'Geist Mono', 'SFMono-Regular', Consolas, monospace;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 12.5px;
    line-height: 1.4;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    font-variant-numeric: tabular-nums;
  }

  /* Minimal Top Bar */
  header {
    height: 44px;
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    position: sticky;
    top: 0;
    z-index: 20;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
  }
  .brand-sep {
    color: var(--border);
  }
  .brand-sub {
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--mono);
  }

  .telemetry-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text-secondary);
  }
  .telemetry-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .telemetry-divider {
    width: 1px;
    height: 14px;
    background: var(--border);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
  }
  .dot.green { background: var(--green); }
  .dot.amber { background: var(--amber); }
  .dot.red { background: var(--red); }
  .dot.active { background: var(--text); }

  /* Workspace Grid */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    border-bottom: 1px solid var(--border);
  }
  @media (max-width: 1180px) {
    main { grid-template-columns: 1fr; }
  }

  .pane-left {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
  }
  .pane-right {
    display: flex;
    flex-direction: column;
  }

  .section {
    border-bottom: 1px solid var(--border);
  }
  .section:last-child {
    border-bottom: none;
  }

  .section-bar {
    height: 36px;
    padding: 0 14px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }
  .section-label {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
  }

  /* Segmented Control */
  .segmented {
    display: inline-flex;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1px;
  }
  .segmented button {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 3px;
    cursor: pointer;
    height: 22px;
  }
  .segmented button:hover {
    color: var(--text-secondary);
  }
  .segmented button.active {
    background: var(--surface-3);
    color: var(--text);
  }

  /* Viewport */
  .viewport {
    position: relative;
    background: #050506;
    width: 100%;
    aspect-ratio: 8 / 3;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .viewport.single-cam {
    aspect-ratio: 4 / 3;
    max-height: 430px;
  }
  .viewport img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .cam-overlay {
    position: absolute;
    top: 8px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
  }
  .cam-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.05em;
    color: rgba(244, 244, 245, 0.75);
    background: rgba(9, 9, 11, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2px 7px;
    border-radius: 3px;
  }

  .progress-line {
    height: 2px;
    background: var(--surface-2);
    width: 100%;
  }
  .progress-fill {
    height: 100%;
    background: var(--text);
    width: 0%;
    transition: width 0.08s linear;
  }

  .status-strip {
    height: 32px;
    padding: 0 14px;
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 11.5px;
    border-top: 1px solid var(--border-subtle);
  }
  .status-left {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .status-state {
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text);
  }
  .status-msg {
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Controls & Buttons */
  button, select, input[type="text"], input[type="number"] {
    font-family: inherit;
    font-size: 12px;
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 5px;
    height: 30px;
    padding: 0 11px;
    outline: none;
    transition: border-color 0.12s ease, background 0.12s ease;
  }
  button {
    cursor: pointer;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    white-space: nowrap;
  }
  button:hover {
    background: var(--surface-3);
    border-color: #32323b;
  }
  button.btn-primary {
    background: var(--primary);
    color: var(--primary-fg);
    border-color: var(--primary);
    font-weight: 600;
  }
  button.btn-primary:hover {
    background: #e4e4e7;
    border-color: #e4e4e7;
  }
  button.btn-ghost {
    background: transparent;
    border-color: var(--border);
    color: var(--text-secondary);
  }
  button.btn-ghost:hover {
    color: var(--text);
    background: var(--surface-2);
  }

  select.compact-select {
    height: 24px;
    padding: 0 8px;
    font-family: var(--mono);
    font-size: 11px;
    background: var(--surface);
    color: var(--text-secondary);
  }

  /* Command Dock */
  .command-dock {
    padding: 14px;
    background: var(--bg);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .prompt-bar {
    display: flex;
    gap: 6px;
  }
  .prompt-input {
    flex: 1;
    height: 32px;
    padding: 0 11px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 12.5px;
    color: var(--text);
  }
  .prompt-input:focus {
    border-color: var(--text-secondary);
  }

  .presets-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .preset-tag {
    height: 22px;
    padding: 0 8px;
    font-size: 11px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
    border-radius: 4px;
  }
  .preset-tag:hover {
    color: var(--text-secondary);
    border-color: var(--border);
    background: var(--surface-2);
  }

  .action-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding-top: 6px;
    border-top: 1px solid var(--border-subtle);
  }
  .btn-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* Sliders & Guards Drawer */
  .drawer-body {
    padding: 12px 14px;
    background: var(--bg);
  }
  .slider-row {
    display: grid;
    grid-template-columns: 110px 1fr 56px;
    align-items: center;
    gap: 12px;
    height: 26px;
    font-family: var(--mono);
    font-size: 11.5px;
  }
  .slider-name {
    color: var(--text-secondary);
  }
  .slider-val {
    text-align: right;
    color: var(--text);
  }
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 3px;
    background: var(--surface-3);
    border-radius: 2px;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--text);
    cursor: pointer;
  }

  .guards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }
  .guard-field label {
    display: block;
    font-size: 10.5px;
    color: var(--text-muted);
    margin-bottom: 3px;
    font-family: var(--mono);
  }
  .guard-field input {
    width: 100%;
    height: 28px;
    font-family: var(--mono);
    background: var(--surface);
  }

  /* Right Pane: KPI Strip */
  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .kpi-cell {
    padding: 10px 14px;
    border-right: 1px solid var(--border);
  }
  .kpi-cell:last-child {
    border-right: none;
  }
  .kpi-label {
    font-size: 10.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .kpi-value {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 600;
    margin-top: 2px;
    color: var(--text);
  }

  /* Verdict Strip */
  .verdict-strip {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11.5px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    color: var(--text-secondary);
  }
  .verdict-strip.ok {
    color: var(--text);
  }
  .verdict-strip.err {
    background: rgba(239, 68, 68, 0.08);
    color: #fca5a5;
    border-bottom-color: rgba(239, 68, 68, 0.25);
  }

  /* 6-DOF Sparkline Matrix */
  .joints-matrix {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    background: var(--border);
    gap: 1px;
    border-bottom: 1px solid var(--border);
  }
  .joint-cell {
    background: var(--bg);
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .joint-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--mono);
    font-size: 11.5px;
  }
  .joint-title {
    color: var(--text-secondary);
    font-weight: 500;
  }
  .joint-readout {
    display: flex;
    gap: 6px;
    font-size: 11px;
  }
  .ro-curr { color: var(--text-secondary); }
  .ro-targ { color: var(--text); font-weight: 600; }
  .ro-delta { color: var(--text-muted); }

  canvas.spark {
    width: 100%;
    height: 48px;
    display: block;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
  }

  /* Data Tables */
  .table-wrap {
    flex: 1;
    max-height: 240px;
    overflow-y: auto;
    background: var(--bg);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11px;
  }
  th, td {
    padding: 5px 10px;
    text-align: right;
    border-bottom: 1px solid var(--border-subtle);
  }
  th {
    position: sticky;
    top: 0;
    background: var(--surface);
    color: var(--text-muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    z-index: 2;
  }
  th:first-child, td:first-child {
    text-align: left;
    padding-left: 14px;
  }
  th:last-child, td:last-child {
    padding-right: 14px;
  }
  tr.active-step {
    background: var(--surface-2);
    color: var(--text);
  }
  tr:hover {
    background: var(--surface);
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-title">NexArm / FLUX.3 Action</span>
    <span class="brand-sep">/</span>
    <span class="brand-sub">MuJoCo Control Loop</span>
  </div>

  <div class="telemetry-bar">
    <div class="telemetry-item">
      <span>Backend</span>
      <select id="backendSelect" class="compact-select" onchange="changeBackend()">
        <option value="server">Port 8000 (FLUX.3)</option>
        <option value="demo_stack">Demo: Pawn e2 → e4</option>
        <option value="demo_sweep">Demo: Joint Sweep</option>
      </select>
    </div>
    <div class="telemetry-divider"></div>
    <div class="telemetry-item">
      <span id="healthDot" class="dot"></span>
      <span id="healthText">Connecting</span>
    </div>
    <div class="telemetry-divider"></div>
    <div class="telemetry-item">
      <span id="fpsText">30.0 Hz</span>
    </div>
    <div class="telemetry-divider"></div>
    <div class="telemetry-item" id="boxPosText">
      Board: --
    </div>
  </div>
</header>

<main>
  <!-- LEFT PANE -->
  <div class="pane-left">
    <div class="section">
      <div class="section-bar">
        <span class="section-label">Viewport (640×480)</span>
        <div class="segmented">
          <button id="viewDualBtn" class="active" onclick="setCamView('dual')">Dual</button>
          <button id="viewSceneBtn" onclick="setCamView('scene')">Scene</button>
          <button id="viewWristBtn" onclick="setCamView('wrist')">Wrist</button>
          <button id="viewTopBtn" onclick="setCamView('top')">Top</button>
        </div>
      </div>

      <div id="viewportWrap" class="viewport">
        <img id="camStream" src="/stream?view=dual" alt="MuJoCo Stream" />
        <div id="camLabels" class="cam-overlay">
          <span class="cam-label">SCENE</span>
          <span class="cam-label">WRIST</span>
        </div>
      </div>

      <div class="progress-line">
        <div id="chunkProgress" class="progress-fill"></div>
      </div>

      <div class="status-strip">
        <div class="status-left">
          <span id="statusDot" class="dot"></span>
          <span id="statusBadge" class="status-state">IDLE</span>
          <span id="statusDetail" class="status-msg">Ready</span>
        </div>
        <span id="stepCounter" style="color:var(--text-secondary);">0 / 32</span>
      </div>
    </div>

    <!-- Command Input & Execution Bar -->
    <div class="section command-dock">
      <div class="prompt-bar">
        <input id="taskInput" type="text" class="prompt-input"
               value="move white pawn e2 to e4"
               placeholder="Task instruction..."
               onkeydown="if(event.key==='Enter') sendControl('once')" />
        <button class="btn-primary" onclick="sendControl('once')" style="height:32px;">
          Step 32
        </button>
        <button onclick="sendControl('run')" style="height:32px;">
          Run Loop
        </button>
      </div>

      <div class="presets-bar">
        <button class="preset-tag" onclick="setPrompt('move white pawn e2 to e4')">move white pawn e2 to e4</button>
        <button class="preset-tag" onclick="setPrompt('pick up the white pawn')">pick up the white pawn</button>
        <button class="preset-tag" onclick="setPrompt('move white pawn d2 to d4')">move white pawn d2 to d4</button>
        <button class="preset-tag" onclick="setPrompt('move white knight g1 to f3')">move white knight g1 to f3</button>
      </div>

      <div class="action-bar">
        <div class="btn-row">
          <button class="btn-ghost" onclick="sendControl('dryrun')">Dry-Run (Plan Only)</button>
          <button class="btn-ghost" onclick="sendControl('pause')">Pause</button>
        </div>
        <div class="btn-row">
          <button class="btn-ghost" onclick="resetScene(false)">Reset Board</button>
          <button class="btn-ghost" onclick="resetScene(true)">Randomize Opening</button>
        </div>
      </div>
    </div>

    <!-- Manual Pose & Safety Guards -->
    <div class="section" style="flex:1;">
      <div class="section-bar">
        <span class="section-label">Configuration</span>
        <div class="segmented">
          <button id="tabSlidersBtn" class="active" onclick="switchBottomTab('sliders')">Joints</button>
          <button id="tabGuardsBtn" onclick="switchBottomTab('guards')">Guards</button>
        </div>
      </div>

      <div class="drawer-body">
        <div id="slidersPane">
          <div id="slidersContainer"></div>
        </div>

        <div id="guardsPane" style="display:none;">
          <div class="guards-grid">
            <div class="guard-field">
              <label>arm_jump (deg)</label>
              <input id="g_arm_jump" type="number" step="1" value="15" />
            </div>
            <div class="guard-field">
              <label>gripper_jump (%)</label>
              <input id="g_gripper_jump" type="number" step="1" value="30" />
            </div>
            <div class="guard-field">
              <label>step_arm_jump (deg)</label>
              <input id="g_step_arm_jump" type="number" step="1" value="10" />
            </div>
            <div class="guard-field">
              <label>step_gripper_jump (%)</label>
              <input id="g_step_gripper_jump" type="number" step="1" value="25" />
            </div>
            <div class="guard-field">
              <label>chunk_arm_span (deg)</label>
              <input id="g_chunk_arm_span" type="number" step="1" value="45" />
            </div>
            <div class="guard-field">
              <label>chunk_gripper_span (%)</label>
              <input id="g_chunk_gripper_span" type="number" step="1" value="60" />
            </div>
          </div>
          <div style="margin-top:10px; display:flex; align-items:center; justify-content:space-between;">
            <label style="display:flex; align-items:center; gap:7px; cursor:pointer; font-family:var(--mono); font-size:11.5px; color:var(--text-secondary);">
              <input id="g_bypass" type="checkbox" />
              Bypass safety guards
            </label>
            <button onclick="saveGuards()">Apply</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANE: Telemetry & Trajectory Inspector -->
  <div class="pane-right">
    <div class="section-bar">
      <span class="section-label">Action Chunk Telemetry (n=32)</span>
      <span id="planTimestamp" style="font-family:var(--mono); font-size:11px; color:var(--text-muted);">—</span>
    </div>

    <div class="kpi-strip">
      <div class="kpi-cell">
        <div class="kpi-label">Plan</div>
        <div id="mPlanNum" class="kpi-value">—</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Inference</div>
        <div id="mServerTime" class="kpi-value">—</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Roundtrip</div>
        <div id="mLatency" class="kpi-value">—</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Peak Excursion</div>
        <div id="mMaxExcursion" class="kpi-value">—</div>
      </div>
    </div>

    <div id="verdictBox" class="verdict-strip">
      <span id="verdictDot" class="dot"></span>
      <span id="verdictText">Awaiting model query</span>
    </div>

    <!-- 6-DOF Sparkline Matrix -->
    <div id="jointsGrid" class="joints-matrix"></div>

    <!-- Bottom Data Inspector -->
    <div class="section-bar">
      <span class="section-label">Chunk Data</span>
      <div class="segmented">
        <button id="tabChunkBtn" class="active" onclick="switchInspectorTab('chunk')">Steps (32)</button>
        <button id="tabHistoryBtn" onclick="switchInspectorTab('history')">History</button>
        <button id="tabJsonBtn" onclick="switchInspectorTab('json')">JSON</button>
      </div>
    </div>

    <div id="chunkTablePane" class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>shoulder_pan</th>
            <th>shoulder_lift</th>
            <th>elbow_flex</th>
            <th>wrist_flex</th>
            <th>wrist_roll</th>
            <th>gripper</th>
          </tr>
        </thead>
        <tbody id="chunkTableBody">
          <tr><td colspan="7" style="text-align:center; color:var(--text-muted); padding:24px;">No action chunk recorded</td></tr>
        </tbody>
      </table>
    </div>

    <div id="historyPane" class="table-wrap" style="display:none;">
      <table>
        <thead>
          <tr>
            <th>Plan</th>
            <th>Prompt</th>
            <th>Backend</th>
            <th>Roundtrip</th>
            <th>Guard</th>
            <th>Δ (Pan / Lift / Elbow / Grip)</th>
          </tr>
        </thead>
        <tbody id="historyTableBody">
          <tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:24px;">No history</td></tr>
        </tbody>
      </table>
    </div>

    <div id="jsonPane" class="table-wrap" style="display:none; padding:12px 14px;">
      <pre id="rawJsonPre" style="font-family:var(--mono); font-size:11px; color:var(--text-secondary); white-space:pre-wrap;">{}</pre>
    </div>
  </div>
</main>

<script>
const JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
const JOINT_UNITS = ["°", "°", "°", "°", "°", "%"];
const JOINT_RANGES = [[-110, 110], [-100, 100], [-110, 110], [-100, 100], [-150, 150], [0, 100]];

let latestSnapshot = null;
let sliderDragging = false;
let lastRenderedPlanId = -1;

function initUI() {
  const grid = document.getElementById("jointsGrid");
  grid.innerHTML = JOINT_NAMES.map((name, i) => `
    <div class="joint-cell">
      <div class="joint-header">
        <span class="joint-title">${name}</span>
        <div class="joint-readout">
          <span class="ro-curr" id="jCurr_${i}">0.0${JOINT_UNITS[i]}</span>
          <span style="color:var(--text-muted)">→</span>
          <span class="ro-targ" id="jTarg_${i}">—</span>
          <span class="ro-delta" id="jDelta_${i}"></span>
        </div>
      </div>
      <canvas class="spark" id="spark_${i}" width="320" height="52"></canvas>
    </div>
  `).join("");

  const sliders = document.getElementById("slidersContainer");
  sliders.innerHTML = JOINT_NAMES.map((name, i) => {
    const [min, max] = JOINT_RANGES[i];
    return `
      <div class="slider-row">
        <span class="slider-name">${name}</span>
        <input type="range" id="slider_${i}" min="${min}" max="${max}" step="0.5" value="0"
               onmousedown="sliderDragging=true" onmouseup="sliderDragging=false; sendManualPose()"
               ontouchstart="sliderDragging=true" ontouchend="sliderDragging=false; sendManualPose()"
               oninput="onSliderInput(${i})" />
        <span id="sliderVal_${i}" class="slider-val">0.0${JOINT_UNITS[i]}</span>
      </div>
    `;
  }).join("");
}

function setCamView(view) {
  document.getElementById("camStream").src = `/stream?view=${view}&t=${Date.now()}`;
  const wrap = document.getElementById("viewportWrap");
  const labels = document.getElementById("camLabels");
  document.getElementById("viewDualBtn").classList.toggle("active", view === "dual");
  document.getElementById("viewSceneBtn").classList.toggle("active", view === "scene");
  document.getElementById("viewWristBtn").classList.toggle("active", view === "wrist");
  const topBtn = document.getElementById("viewTopBtn");
  if (topBtn) topBtn.classList.toggle("active", view === "top");
  if (view === "dual") {
    wrap.classList.remove("single-cam");
    labels.innerHTML = `<span class="cam-label">SCENE</span><span class="cam-label">WRIST</span>`;
  } else {
    wrap.classList.add("single-cam");
    labels.innerHTML = `<span class="cam-label">${view.toUpperCase()}</span>`;
  }
}

function switchBottomTab(tab) {
  document.getElementById("slidersPane").style.display = tab === "sliders" ? "block" : "none";
  document.getElementById("guardsPane").style.display = tab === "guards" ? "block" : "none";
  document.getElementById("tabSlidersBtn").classList.toggle("active", tab === "sliders");
  document.getElementById("tabGuardsBtn").classList.toggle("active", tab === "guards");
}

function switchInspectorTab(tab) {
  document.getElementById("chunkTablePane").style.display = tab === "chunk" ? "block" : "none";
  document.getElementById("historyPane").style.display = tab === "history" ? "block" : "none";
  document.getElementById("jsonPane").style.display = tab === "json" ? "block" : "none";
  document.getElementById("tabChunkBtn").classList.toggle("active", tab === "chunk");
  document.getElementById("tabHistoryBtn").classList.toggle("active", tab === "history");
  document.getElementById("tabJsonBtn").classList.toggle("active", tab === "json");
}

function setPrompt(text) {
  document.getElementById("taskInput").value = text;
  sendControl("once");
}

async function sendControl(mode) {
  const task = document.getElementById("taskInput").value;
  const backend = document.getElementById("backendSelect").value;
  await fetch("/api/control", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task, mode, backend})
  });
  pollState();
}

async function changeBackend() {
  const backend = document.getElementById("backendSelect").value;
  await fetch("/api/control", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({backend})
  });
  pollState();
}

async function resetScene(randomize_boxes) {
  await fetch("/api/reset", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({randomize_boxes})
  });
  pollState();
}

function onSliderInput(idx) {
  const val = parseFloat(document.getElementById(`slider_${idx}`).value);
  document.getElementById(`sliderVal_${idx}`).textContent = `${val.toFixed(1)}${JOINT_UNITS[idx]}`;
  sendManualPose();
}

async function sendManualPose() {
  const targets = JOINT_NAMES.map((_, i) => parseFloat(document.getElementById(`slider_${i}`).value));
  await fetch("/api/manual_pose", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({targets})
  });
}

async function saveGuards() {
  const payload = {
    arm_jump: parseFloat(document.getElementById("g_arm_jump").value),
    gripper_jump: parseFloat(document.getElementById("g_gripper_jump").value),
    step_arm_jump: parseFloat(document.getElementById("g_step_arm_jump").value),
    step_gripper_jump: parseFloat(document.getElementById("g_step_gripper_jump").value),
    chunk_arm_span: parseFloat(document.getElementById("g_chunk_arm_span").value),
    chunk_gripper_span: parseFloat(document.getElementById("g_chunk_gripper_span").value),
    bypass: document.getElementById("g_bypass").checked
  };
  await fetch("/api/guards", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
}

function drawSparkline(idx, state, plan, currentStep) {
  const canvas = document.getElementById(`spark_${idx}`);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const hist = (plan && plan.history_states && plan.history_states.length)
    ? plan.history_states.map(s => s[idx])
    : [state[idx]];
  const cmds = (plan && plan.commands) ? plan.commands.map(c => c[idx]) : [];
  const allVals = [...hist, ...cmds, state[idx]];
  let minV = Math.min(...allVals);
  let maxV = Math.max(...allVals);
  if (maxV - minV < 6) {
    const mid = 0.5 * (maxV + minV);
    minV = mid - 3;
    maxV = mid + 3;
  }
  const pad = 6;
  const toY = v => H - pad - ((v - minV) / (maxV - minV)) * (H - 2 * pad);

  const baseVal = plan ? plan.state_at_plan[idx] : state[idx];
  const baseY = toY(baseVal);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(0, baseY);
  ctx.lineTo(W, baseY);
  ctx.stroke();
  ctx.setLineDash([]);

  const nHist = hist.length;
  const nCmd = cmds.length || 32;
  const totalPts = nHist + nCmd;
  const toX = stepIdx => (stepIdx / Math.max(1, totalPts - 1)) * (W - 12) + 6;

  if (nHist > 1) {
    ctx.strokeStyle = "rgba(161, 161, 170, 0.35)";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = toX(i), y = toY(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (cmds.length > 0) {
    const ok = plan.guard_ok;
    ctx.strokeStyle = ok ? "#f4f4f5" : "#ef4444";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(toX(nHist - 1), toY(hist[nHist - 1]));
    cmds.forEach((v, i) => {
      ctx.lineTo(toX(nHist + i), toY(v));
    });
    ctx.stroke();

    if (currentStep > 0 && currentStep <= cmds.length) {
      const cx = toX(nHist + currentStep - 1);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, H);
      ctx.stroke();
    }
  }
}

function updateUI(snap) {
  latestSnapshot = snap;

  document.getElementById("fpsText").textContent = `${snap.sim_fps.toFixed(1)} Hz`;
  const hDot = document.getElementById("healthDot");
  const hText = document.getElementById("healthText");
  const sh = snap.server_health || {};
  hDot.className = "dot " + (sh.status === "ready" ? "green" : (sh.status === "loading" ? "amber" : "red"));
  hText.textContent = sh.status === "ready"
    ? `${sh.model || "Ready"}`
    : (sh.status === "loading" ? "Loading" : "Port 8000 Offline");

  if (snap.box_positions && snap.box_positions.squares) {
    const sq = snap.box_positions.squares;
    document.getElementById("boxPosText").textContent =
      `w_e2:${sq.w_pawn_e2 || '--'} · w_d2:${sq.w_pawn_d2 || '--'} · w_g1:${sq.w_knight_g1 || '--'} · b_e7:${sq.b_pawn_e7 || '--'}`;
  } else if (snap.box_positions && snap.box_positions.small_box && snap.box_positions.big_box) {
    const sb = snap.box_positions.small_box;
    const bb = snap.box_positions.big_box;
    document.getElementById("boxPosText").textContent =
      `S(${sb[0].toFixed(2)}, ${sb[1].toFixed(2)}) B(${bb[0].toFixed(2)}, ${bb[1].toFixed(2)})`;
  }

  const sDot = document.getElementById("statusDot");
  document.getElementById("statusBadge").textContent = snap.status;
  document.getElementById("statusDetail").textContent = snap.status_detail;
  sDot.className = "dot " + (
    snap.status === "executing" ? "green" :
    snap.status === "planning" ? "amber" :
    (snap.status === "rejected" || snap.status === "error") ? "red" : "active"
  );

  const pct = snap.total_steps > 0 ? Math.min(100, (snap.current_step / snap.total_steps) * 100) : 0;
  document.getElementById("chunkProgress").style.width = `${pct}%`;
  document.getElementById("stepCounter").textContent = `${snap.current_step} / ${snap.total_steps}`;

  if (!sliderDragging) {
    snap.state.forEach((v, i) => {
      const sl = document.getElementById(`slider_${i}`);
      const sv = document.getElementById(`sliderVal_${i}`);
      if (sl && sv) {
        sl.value = v;
        sv.textContent = `${v.toFixed(1)}${JOINT_UNITS[i]}`;
      }
    });
  }

  const plan = snap.last_plan;
  snap.state.forEach((v, i) => {
    document.getElementById(`jCurr_${i}`).textContent = `${v.toFixed(1)}${JOINT_UNITS[i]}`;
    if (plan && plan.last_command) {
      const targ = plan.last_command[i];
      const delta = plan.joint_deltas[i];
      const sign = delta >= 0 ? "+" : "";
      document.getElementById(`jTarg_${i}`).textContent = `${targ.toFixed(1)}${JOINT_UNITS[i]}`;
      document.getElementById(`jDelta_${i}`).textContent = `${sign}${delta.toFixed(1)}`;
    }
    drawSparkline(i, snap.state, plan, snap.current_step);
  });

  if (plan) {
    document.getElementById("mPlanNum").textContent = `#${plan.plan}`;
    document.getElementById("mServerTime").textContent =
      plan.server_s !== null ? `${(plan.server_s * 1000).toFixed(0)} ms` : "—";
    document.getElementById("mLatency").textContent = `${(plan.latency_s * 1000).toFixed(0)} ms`;
    const maxArmExc = Math.max(...plan.max_excursions.slice(0, 5));
    document.getElementById("mMaxExcursion").textContent = `${maxArmExc.toFixed(1)}°`;
    document.getElementById("planTimestamp").textContent =
      new Date(plan.timestamp * 1000).toLocaleTimeString();

    const vbox = document.getElementById("verdictBox");
    const vdot = document.getElementById("verdictDot");
    const vtxt = document.getElementById("verdictText");
    if (plan.guard_ok) {
      vbox.className = "verdict-strip ok";
      vdot.className = "dot green";
      vtxt.textContent = `Guards passed · "${plan.task}"`;
    } else {
      vbox.className = "verdict-strip err";
      vdot.className = "dot red";
      vtxt.textContent = `Rejected · ${plan.guard_problem}`;
    }

    const tbody = document.getElementById("chunkTableBody");
    tbody.innerHTML = plan.commands.map((cmd, idx) => {
      const isCurr = (idx + 1) === snap.current_step;
      return `<tr class="${isCurr ? 'active-step' : ''}">
        <td>${idx + 1}</td>
        ${cmd.map(v => `<td>${v.toFixed(2)}</td>`).join("")}
      </tr>`;
    }).join("");

    if (plan.plan !== lastRenderedPlanId) {
      lastRenderedPlanId = plan.plan;
      document.getElementById("rawJsonPre").textContent = JSON.stringify(plan, null, 2);
    }
  }

  if (snap.plan_history && snap.plan_history.length > 0) {
    const hbody = document.getElementById("historyTableBody");
    hbody.innerHTML = snap.plan_history.map(h => `
      <tr>
        <td>#${h.plan}</td>
        <td style="max-width:190px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${h.task}">${h.task}</td>
        <td>${h.backend}</td>
        <td>${(h.latency_s * 1000).toFixed(0)} ms</td>
        <td style="color:${h.guard_ok ? 'var(--green)' : 'var(--red)'};">${h.guard_ok ? 'OK' : 'REJECT'}</td>
        <td>${h.joint_deltas ? `${h.joint_deltas[0]} / ${h.joint_deltas[1]} / ${h.joint_deltas[2]} / ${h.joint_deltas[5]}` : '—'}</td>
      </tr>
    `).join("");
  }
}

async function pollState() {
  try {
    const resp = await fetch("/api/state");
    if (resp.ok) {
      const data = await resp.json();
      updateUI(data);
    }
  } catch (e) {}
}

initUI();
pollState();
setInterval(pollState, 120);
</script>
</body>
</html>
"""


def create_app(controller: SimWebController) -> FastAPI:
    app = FastAPI(title="NexArm × FLUX.3 Action — MuJoCo Live Studio")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/state")
    def get_state():
        return controller.get_snapshot()

    @app.post("/api/control")
    def post_control(req: ControlRequest):
        if req.mode is not None and req.mode not in ("run", "once", "dryrun", "pause"):
            raise HTTPException(400, f"Invalid mode: {req.mode}")
        if req.backend is not None and req.backend not in ("server", "demo_stack", "demo_sweep"):
            raise HTTPException(400, f"Invalid backend: {req.backend}")
        controller.enqueue("control", req)
        return {"ok": True}

    @app.post("/api/reset")
    def post_reset(req: ResetRequest):
        controller.enqueue("reset", req.randomize_boxes)
        return {"ok": True}

    @app.post("/api/manual_pose")
    def post_manual_pose(req: ManualPoseRequest):
        controller.enqueue("manual_pose", req.targets)
        return {"ok": True}

    @app.post("/api/guards")
    def post_guards(req: GuardsRequest):
        with controller._lock:
            controller.guards = req.model_dump()
        return {"ok": True, "guards": controller.guards}

    @app.get("/stream")
    def stream(view: str = "dual"):
        def generate():
            last_seq = -1
            while not controller._stop_event.is_set():
                with controller._frame_cond:
                    if controller._frame_seq == last_seq:
                        controller._frame_cond.wait(timeout=0.5)
                    seq = controller._frame_seq
                last_seq = seq
                jpeg = controller.get_jpeg(view)
                if not jpeg:
                    time.sleep(0.03)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SCENE, help="Path to MuJoCo scene.xml")
    parser.add_argument(
        "--config",
        type=Path,
        default=CLOUD_DIR.parent / "config" / "inference.yaml",
        help="LeRobot YAML with robot calibration_dir and use_degrees (default: config/inference.yaml)",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FLUX.3 Action server URL")
    parser.add_argument("--task", default="put the small box on top of the big box", help="Initial English task")
    parser.add_argument(
        "--backend",
        choices=("server", "demo_stack", "demo_sweep"),
        default="server",
        help="Initial backend ('server' queries --url, 'demo_stack' runs built-in demo)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    if not args.sim.is_file():
        raise SystemExit(
            f"MuJoCo scene not found: {args.sim}\n"
            "Clone third_party/khanhthanhdev-lerobot-nexarm first (see sim-test.cmd)."
        )

    from sim import MujocoRobot
    import uvicorn

    robot = MujocoRobot(args.sim, config_path=args.config)
    controller = SimWebController(robot, server_url=args.url, task=args.task)
    controller.backend = args.backend
    controller.start()

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    try:
        uvicorn.run(create_app(controller), host=args.host, port=args.port, log_level="info")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
