"""Tests for the selectively integrated FLUX-3-Action cloud client, AI Galaxy helper and server."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

NEXARM_F3A = Path(__file__).resolve().parents[1]
CLOUD_DIR = NEXARM_F3A / "cloud"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"f3a_cloud_{name}", CLOUD_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


client = load_module("client")
galaxy = load_module("galaxy")


class DummyRobot:
    def __init__(self, state: list[float] | None = None):
        self.state = list(state or [0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self.image = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.sent: list[list[float]] = []

    def get_observation(self) -> dict:
        return {
            **dict(zip(client.KEYS, self.state, strict=True)),
            "scene": self.image,
            "wrist": self.image,
        }

    def send_action(self, action: dict) -> dict:
        cmd = [float(action[key]) for key in client.KEYS]
        self.sent.append(cmd)
        self.state = list(cmd)
        return action


def test_check_plan_accepts_smooth_trajectory():
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    commands = [[float(i), 0.0, 0.0, 0.0, 0.0, 50.0] for i in range(1, 33)]
    assert client.check_plan(state, commands, 15.0, 30.0) is None


def test_check_plan_rejects_first_step_jump():
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    commands = [[20.0, 0.0, 0.0, 0.0, 0.0, 50.0]] + [[20.0, 0.0, 0.0, 0.0, 0.0, 50.0]] * 31
    problem = client.check_plan(state, commands, 15.0, 30.0)
    assert problem is not None and "shoulder_pan" in problem

    gripper_bad = [[0.0, 0.0, 0.0, 0.0, 0.0, 90.0]] * 32
    problem_g = client.check_plan(state, gripper_bad, 15.0, 30.0)
    assert problem_g is not None and "gripper" in problem_g


def test_check_plan_rejects_mid_chunk_step_jump():
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    commands = [list(state) for _ in range(32)]
    commands[11] = [0.0, 20.0, 0.0, 0.0, 0.0, 50.0]
    problem = client.check_plan(state, commands, 15.0, 30.0, step_arm_jump=10.0)
    assert problem is not None
    assert "step 10 -> 11" in problem and "shoulder_lift" in problem


def test_check_plan_rejects_excessive_chunk_excursion():
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    commands = [[2.0 * (i + 1), 0.0, 0.0, 0.0, 0.0, 50.0] for i in range(32)]
    problem = client.check_plan(state, commands, 15.0, 30.0, step_arm_jump=10.0, chunk_arm_span=45.0)
    assert problem is not None
    assert "chunk span limit" in problem and "shoulder_pan" in problem


def test_client_dry_run_and_execute_history(tmp_path):
    # Dry-run: sends zero actions and keeps last_command as None
    dry_robot = DummyRobot()
    seen_dry: list[list[dict]] = []

    def predict_dry(task, frames):
        seen_dry.append(frames)
        if len(seen_dry) == 2:
            raise OSError("stop")
        return {"commands": [[1.0, 0.0, 0.0, 0.0, 0.0, 50.0]] * 32, "seconds": 0.1}

    c_dry = client.Client(dry_robot, predict_dry, "stack boxes", execute=False, fps=1000.0)
    assert c_dry.run() == "server: stop"
    assert dry_robot.sent == []
    assert all(frame["last_command"] is None for frame in seen_dry[1])

    # Execute: sends 32 actions and stores last 8 ticks as history
    exec_robot = DummyRobot()
    seen_exec: list[list[dict]] = []

    def predict_exec(task, frames):
        seen_exec.append(frames)
        if len(seen_exec) == 2:
            raise OSError("stop")
        return {"commands": [[float(i) * 0.5, 0.0, 0.0, 0.0, 0.0, 50.0] for i in range(1, 33)], "seconds": 0.1}

    log_file = tmp_path / "client.jsonl"
    c_exec = client.Client(exec_robot, predict_exec, "stack boxes", execute=True, fps=1000.0, log_path=log_file)
    assert c_exec.run() == "server: stop"
    assert len(exec_robot.sent) == 32
    assert len(seen_exec[1]) == client.HISTORY
    assert all(frame["last_command"] is not None for frame in seen_exec[1])
    assert log_file.is_file()


def test_load_follower_uses_user_config_and_port_override():
    robot = client.load_follower(NEXARM_F3A / "config" / "inference.yaml", port="COM99")
    assert robot.config.port == "COM99"
    assert set(robot.cameras) == {"scene", "wrist"}
    assert not robot.is_connected


def test_load_env_file_parses_quotes_comments_and_keeps_os_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "export SSH_HOST='ai.blockelite.cn'\n"
        'SSH_PORT="12345"\n'
        "REMOTE_DIR=/root/data/nexarm # inline comment\n"
        "ALREADY_SET=from_file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALREADY_SET", "from_os")
    for key in ("SSH_HOST", "SSH_PORT", "REMOTE_DIR"):
        monkeypatch.delenv(key, raising=False)

    galaxy.load_env_file(env_file)
    assert galaxy.os.environ["SSH_HOST"] == "ai.blockelite.cn"
    assert galaxy.os.environ["SSH_PORT"] == "12345"
    assert galaxy.os.environ["REMOTE_DIR"] == "/root/data/nexarm"
    assert galaxy.os.environ["ALREADY_SET"] == "from_os"


def test_galaxy_train_script_batch_sizes_and_custom_remote_dir():
    s1 = galaxy.train_script("my_ds", "run1", steps=None, resume=False, batch_size=1, remote_dir="/root/data/nexarm")
    assert "/workspace" not in s1
    assert "cd /root/data/nexarm/lerobot" in s1
    assert "--dataset.root=/root/data/nexarm/data/my_ds" in s1
    assert "--batch_size=1" in s1
    assert "--accelerator.gradient_accumulation.steps=8" in s1
    assert "--save_freq=1000" in s1
    assert "--eval_steps=1000" in s1
    assert "--steps=20000" in s1

    s2 = galaxy.train_script("my_ds", "run2", steps=None, resume=False, batch_size=2, remote_dir="/root/data/nexarm")
    assert "/workspace" not in s2
    assert "cd /root/data/nexarm/lerobot" in s2
    assert "--batch_size=1" not in s2
    assert "--steps=" not in s2


def test_galaxy_training_table_and_validate_remote_dir():
    log = (
        "INFO step:100 loss:0.542 action_mse:0.120 video_mse:0.422\n"
        "INFO step:200 loss:0.310 action_mse:0.080 video_mse:0.230\n"
        "INFO Eval loss: 0.295\n"
    )
    table = galaxy.training_table(log)
    assert "100" in table and "200" in table and "Eval loss: 0.295" in table

    assert galaxy.validate_remote_dir("/root/data/nexarm/") == "/root/data/nexarm"
    for bad in ("/root/data; rm -rf /", "relative/path", "/root/../etc"):
        with pytest.raises(SystemExit):
            galaxy.validate_remote_dir(bad)


def test_server_models_and_pad_frames():
    pytest.importorskip("fastapi")
    server = load_module("server")

    f1 = server.Frame(image_scene="a", image_wrist="b", state=[0, 0, 0, 0, 0, 10])
    f2 = server.Frame(image_scene="c", image_wrist="d", state=[1, 0, 0, 0, 0, 20])
    padded = server.pad_frames([f1, f2], 4)
    assert len(padded) == 4 and padded[:2] == [f1, f1] and padded[2:] == [f1, f2]
    assert server.pad_frames([f1, f2], 1) == [f2]

    req = server.PredictRequest(task="stack boxes", frames=[f2])
    mock_cmds = server.MockModel().predict(req)
    sweep_cmds = server.SweepModel().predict(req)
    assert len(mock_cmds) == 32 and all(len(c) == 6 for c in mock_cmds)
    assert len(sweep_cmds) == 32 and all(len(c) == 6 for c in sweep_cmds)
