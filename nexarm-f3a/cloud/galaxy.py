"""Manage FLUX 3 Action training & inference on an AI Galaxy (gpu.ai-galaxy.com) instance via SSH.

Adapted from mariorch22/lerobot-nexarm (cloud/deploy.py, Apache-2.0). Pure Python stdlib:
no vastai SDK or python-dotenv required. Reads connection and mirror settings from cloud/.env.

    python cloud/galaxy.py setup     upload server.py + setup_remote.sh, install env & download weights
    python cloud/galaxy.py up        start server.py in tmux and open local tunnel http://127.0.0.1:8000
    python cloud/galaxy.py status    show SSH/GPU status, disk space, tmux sessions and /health
    python cloud/galaxy.py train --dataset PATH     upload dataset and start LoRA training
    python cloud/galaxy.py logs      follow live GPU memory & /train.log
    python cloud/galaxy.py curves    save train.log locally and print condensed loss table
    python cloud/galaxy.py fetch     download trained LoRA adapter + processors to outputs/lora/<run>
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWN_HOSTS = HERE / ".known_hosts"

PASSTHROUGH_ENV_VARS = (
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_XET",
    "UV_DEFAULT_INDEX",
    "UV_PYTHON_INSTALL_MIRROR",
    "PYTORCH_INDEX_URL",
    "LEROBOT_GIT_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
)


def fail(message: str):
    sys.exit(f"ERROR: {message}")


def load_env_file(path: Path) -> None:
    """Parse KEY=VALUE lines from a .env file without overriding already-set OS env vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def validate_remote_dir(remote_dir: str) -> str:
    value = remote_dir.strip().rstrip("/") or "/"
    if not re.fullmatch(r"/[\w./-]+", value) or ".." in Path(value).parts:
        fail(f"REMOTE_DIR '{remote_dir}' must be a clean absolute POSIX path (only letters, digits, '.', '_', '-', '/')")
    return value


def train_script(
    dataset: str,
    run: str,
    steps: int | None,
    resume: bool,
    batch_size: int = 2,
    remote_dir: str = "/workspace",
) -> str:
    """Shell script for the instance: BFL's LoRA recipe on a local dataset, or resuming a run.

    batch_size=2 with accumulation 4 (10000 microsteps) fits 40+ GB GPUs such as A800-40GB.
    batch_size=1 with accumulation 8 (20000 microsteps) fits 32 GB GPUs such as RTX 5090.
    """
    remote_dir = validate_remote_dir(remote_dir)
    if batch_size not in (1, 2):
        fail("--batch-size must be 1 or 2")
    for name in (dataset, run):
        if not re.fullmatch(r"[\w.-]+", name):
            fail(f"'{name}' may only contain letters, digits, '.', '_' and '-'")
    if resume:
        arguments = [
            f"--config_path=outputs/{run}/checkpoints/last/pretrained_model/train_config.json",
            "--resume=true",
        ]
    else:
        arguments = [
            "--config_path=examples/flux3/lora.json",
            "--policy.path=models/flux-3-action-so101",
            "--policy.device=cuda",
            "--policy.video_vae_id=models/flux-3-action-base/video_vae.safetensors",
            "--policy.text_encoder_id=models/flux-3-action-base/text_encoder",
            f"--dataset.repo_id=local/{dataset}",
            f"--dataset.root={remote_dir}/data/{dataset}",
            f"--output_dir=outputs/{run}",
        ]
        if batch_size == 1:
            # save_freq and eval_steps must stay divisible by the accumulation.
            arguments += [
                "--batch_size=1",
                "--accelerator.gradient_accumulation.steps=8",
                "--save_freq=1000",
                "--eval_steps=1000",
            ]
            steps = 20000 if steps is None else steps
        if steps is not None:
            arguments.append(f"--steps={steps}")
    lines = [
        "set -euo pipefail",
        f"cd {remote_dir}/lerobot",
        "source .venv/bin/activate",
        "export HF_HUB_OFFLINE=1  # everything needed is already on the disk",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory lost to fragmentation",
        "python -m lerobot.scripts.lerobot_train \\",
        *[f"  {argument} \\" for argument in arguments[:-1]],
        f"  {arguments[-1]}",
    ]
    return "\n".join(lines) + "\n"


def training_table(log: str, rows: int = 20) -> str:
    """Condense a LeRobot training log: evenly spaced training lines, then every evaluation line."""
    train = [dict(re.findall(r"(\w+):(\S+)", line)) for line in log.splitlines() if " step:" in line]
    train = [entry for entry in train if "loss" in entry]
    picked = train[:: max(1, len(train) // rows)]
    if train and picked[-1] is not train[-1]:
        picked.append(train[-1])  # always show where the training ended
    lines = ["  step      loss  action_mse  video_mse"]
    lines += [
        f"{e['step']:>6}  {e['loss']:>8}  {e.get('action_mse', '-'):>10}  {e.get('video_mse', '-'):>9}"
        for e in picked
    ]
    evaluations = [line.strip() for line in log.splitlines() if "eval" in line.lower() and "loss" in line]
    return "\n".join(
        [*lines, "", "Held-out episodes:", *(evaluations or ["  no evaluation line in the log yet"])]
    )


def checkpoint(step: int | None) -> str:
    """Directory name of a saved training state: the newest one, or the one after `step` microsteps."""
    return "last" if step is None else f"{step:06d}"


def api_health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


class GalaxyDeployment:
    def __init__(self, env_path: Path | None = None):
        load_env_file(env_path or (HERE / ".env"))
        self.host = (os.environ.get("SSH_HOST") or "").strip()
        port_raw = (os.environ.get("SSH_PORT") or "").strip()
        self.user = (os.environ.get("SSH_USER") or "root").strip()
        self.remote_dir = validate_remote_dir(os.environ.get("REMOTE_DIR") or "/workspace")
        if not self.host or not port_raw:
            fail("SSH_HOST and SSH_PORT are missing; copy cloud/.env.example to cloud/.env and fill them in.")
        try:
            self.port = int(port_raw)
        except ValueError:
            fail(f"SSH_PORT must be an integer, got {port_raw!r}")
        if not 1 <= self.port <= 65535:
            fail(f"SSH_PORT out of range: {self.port}")
        key_env = (os.environ.get("SSH_KEY_PATH") or "").strip()
        if key_env:
            self.key: Path | None = Path(key_env).expanduser()
            if not self.key.is_file():
                fail(f"SSH_KEY_PATH does not exist: {self.key}")
        else:
            default_key = Path.home() / ".ssh" / "id_ed25519"
            self.key = default_key if default_key.is_file() else None

    def ssh_command(self, *extra: str) -> list[str]:
        cmd = ["ssh"]
        if self.key is not None:
            cmd += ["-i", str(self.key), "-o", "BatchMode=yes"]
        cmd += [
            "-p",
            str(self.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}",
            "-o",
            "ServerAliveInterval=15",
            *extra,
            f"{self.user}@{self.host}",
        ]
        return cmd

    def ssh(self, command: str, stdin: bytes | None = None, check: bool = True, quiet: bool = False) -> int:
        """Run a command on the instance; its output goes straight to this terminal."""
        result = subprocess.run(
            [*self.ssh_command(), command],
            input=stdin,
            stdin=subprocess.DEVNULL if stdin is None else None,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
        )
        if check and result.returncode:
            fail(f"remote command failed (exit {result.returncode}): {command}")
        return result.returncode

    def connect(self) -> None:
        if self.ssh("true", check=False, quiet=True) == 0:
            return
        fail(
            f"SSH connection to {self.user}@{self.host}:{self.port} failed.\n"
            "If you have not installed your SSH public key on this AI Galaxy instance yet, run once in PowerShell:\n"
            f'  Get-Content "$env:USERPROFILE\\.ssh\\id_ed25519.pub" | ssh -p {self.port} {self.user}@{self.host} '
            '"mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"'
        )

    def upload(self, name: str) -> None:
        # Through stdin with Unix line endings, so a Windows checkout cannot break the shell script.
        content = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
        self.ssh(f"mkdir -p {self.remote_dir} && cat > {self.remote_dir}/{name}", stdin=content)

    def upload_dir(self, local: Path, remote_parent: str) -> None:
        """Copy a directory to remote_parent/<its name> as a tar stream, replacing an older copy."""
        self.ssh(f"mkdir -p {remote_parent} && rm -rf {remote_parent}/{local.name}")
        process = subprocess.Popen(
            [*self.ssh_command(), f"tar -xf - -C {remote_parent}"], stdin=subprocess.PIPE
        )
        with tarfile.open(fileobj=process.stdin, mode="w|") as tar:
            tar.add(local, arcname=local.name)
        process.stdin.close()
        if process.wait():
            fail(f"upload of {local} failed")

    def download(self, remote_dir: str, names: str, local: Path) -> None:
        """Copy entries of a remote directory as a tar stream, following symlinks."""
        process = subprocess.Popen(
            [*self.ssh_command(), f"cd {remote_dir} && tar -chf - {names}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        local.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            tar.extractall(local, filter="data")
        if process.wait():
            fail(f"download from {remote_dir} failed")

    def _remote_env_prefix(self, extra: dict[str, str] | None = None) -> str:
        pairs = {"REMOTE_DIR": self.remote_dir}
        for key in PASSTHROUGH_ENV_VARS:
            val = (os.environ.get(key) or "").strip()
            if val:
                pairs[key] = val
        if extra:
            pairs.update(extra)
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in pairs.items())

    def setup(self, *, start_server: bool = False, skip_gpu_check: bool = False, server_args: str = "") -> None:
        hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN") or "").strip()
        if not hf_token:
            fail("HF_TOKEN is missing; put it into cloud/.env")
        self.connect()
        self.upload("server.py")
        self.upload("setup_remote.sh")
        extra = {
            "START_SERVER": "1" if start_server else "0",
            "SKIP_GPU_CHECK": "1" if skip_gpu_check else "0",
        }
        if server_args:
            extra["SERVER_ARGS"] = server_args
        env_prefix = self._remote_env_prefix(extra)
        # The token is piped through stdin so it never shows up in a command line.
        self.ssh(
            f"read -r HF_TOKEN && export HF_TOKEN && {env_prefix} bash {self.remote_dir}/setup_remote.sh",
            stdin=hf_token.encode() + b"\n",
        )

    def _server_cmd(self, server_args: str) -> str:
        return (
            f"env -u HF_TOKEN HF_HUB_OFFLINE=1 {self.remote_dir}/lerobot/.venv/bin/python "
            f"{self.remote_dir}/server.py {server_args} > {self.remote_dir}/server.log 2>&1"
        )

    def up(
        self,
        local_port: int,
        redeploy: bool,
        adapter: Path | None,
        run: str | None,
        step: int | None = None,
        text_encoder_on_cpu: bool = False,
    ) -> None:
        if run is not None and not re.fullmatch(r"[\w.-]+", run):
            fail(f"run name '{run}' may only contain letters, digits, '.', '_', '-'")
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            fail(f"{adapter} is not an adapter directory (no adapter_config.json)")
        if adapter is not None and not re.fullmatch(r"[\w.-]+", adapter.name):
            fail(f"adapter directory name '{adapter.name}' may only contain letters, digits, '.', '_', '-'")
        self.connect()
        if self.ssh("tmux has-session -t train", check=False, quiet=True) == 0:
            fail("a training runs on the instance; the server would take its GPU memory. See `galaxy.py logs`.")

        server_flags: list[str] = []
        if text_encoder_on_cpu:
            server_flags.append("--text-encoder-on-cpu")

        if run is not None:
            remote = f"{self.remote_dir}/lerobot/outputs/{run}/checkpoints/{checkpoint(step)}/pretrained_model"
            if self.ssh(f"test -d {remote}", check=False, quiet=True) != 0:
                fail(f"run '{run}' has no checkpoint '{checkpoint(step)}' on the instance")
            server_flags += ["--adapter", remote]
        elif adapter is not None:
            self.upload_dir(adapter, f"{self.remote_dir}/adapters")
            server_flags += ["--adapter", f"{self.remote_dir}/adapters/{adapter.name}"]

        server_args = " ".join(shlex.quote(arg) for arg in server_flags)
        installed = (
            self.ssh(
                f"test -f {self.remote_dir}/lerobot/models/flux-3-action-so101/config.json",
                check=False,
                quiet=True,
            )
            == 0
        )
        running = self.ssh("curl -fs http://127.0.0.1:8000/health", check=False, quiet=True) == 0

        if not installed or redeploy:
            self.setup(start_server=True, server_args=server_args)
        elif not running or server_args:
            self.upload("server.py")
            self.ssh("tmux kill-session -t f3a 2>/dev/null || true")
            self.ssh(f"tmux new-session -d -s f3a {shlex.quote(self._server_cmd(server_args))}")
        else:
            print("Server already runs on the instance; opening tunnel (use --redeploy to redo setup).")

        print(f"Opening tunnel 127.0.0.1:{local_port} -> {self.host}:{self.port}, waiting for model ...")
        tunnel = subprocess.Popen(
            [*self.ssh_command("-N", "-L", f"{local_port}:127.0.0.1:8000")], stdin=subprocess.DEVNULL
        )
        try:
            deadline = time.monotonic() + 15 * 60
            while (health := api_health(local_port) or {}).get("status") != "ready":
                if health.get("status") == "error":
                    fail(f"model failed to load: {health['error']} (see {self.remote_dir}/server.log)")
                if tunnel.poll() is not None:
                    fail(f"SSH tunnel exited; is local port {local_port} already in use?")
                if time.monotonic() > deadline:
                    fail(f"model did not become ready; see {self.remote_dir}/server.log on the instance")
                time.sleep(5)
            print(f"\nREADY on {health['model']}")
            print(f"API: http://127.0.0.1:{local_port}  (docs: /docs, health: /health)")
            print(
                "IMPORTANT: Ctrl+C closes ONLY the SSH tunnel. "
                "To stop billing on AI Galaxy, click End (结束) in the web console — Shut Down (关机) is NOT enough!"
            )
            tunnel.wait()
            print("SSH tunnel ended. The instance is STILL RUNNING on AI Galaxy.")
        except KeyboardInterrupt:
            print(
                "\nTunnel closed. The instance is STILL RUNNING on AI Galaxy — "
                "click End (结束) in the console when finished!"
            )
        finally:
            tunnel.terminate()

    def train(self, dataset: Path, run: str, steps: int | None, resume: bool, batch_size: int) -> None:
        if not (dataset / "meta" / "info.json").is_file():
            fail(f"{dataset} is not a LeRobot dataset (no meta/info.json)")
        script = train_script(dataset.name, run, steps, resume, batch_size, remote_dir=self.remote_dir)
        self.connect()
        if self.ssh("tmux has-session -t train", check=False, quiet=True) == 0:
            fail("a training already runs on the instance; watch it with `galaxy.py logs`")
        installed = (
            self.ssh(
                f"test -f {self.remote_dir}/lerobot/models/flux-3-action-so101/config.json",
                check=False,
                quiet=True,
            )
            == 0
        )
        if installed:
            self.ssh("tmux kill-session -t f3a 2>/dev/null || true")
        else:
            self.setup(start_server=False)
        exists = self.ssh(f"test -e {self.remote_dir}/lerobot/outputs/{run}", check=False, quiet=True) == 0
        if exists != resume:
            fail(
                f"run '{run}' already exists on the instance; add --resume or choose another --run"
                if exists
                else f"run '{run}' does not exist on the instance, nothing to resume"
            )
        print(f"Uploading {dataset} -> {self.remote_dir}/data ...")
        self.upload_dir(dataset, f"{self.remote_dir}/data")
        self.ssh(f"cat > {self.remote_dir}/train.sh", stdin=script.encode())
        self.ssh(
            f"tmux new-session -d -s train 'bash {self.remote_dir}/train.sh > {self.remote_dir}/train.log 2>&1'"
        )
        print(f"\nTraining '{run}' started in tmux session 'train'.")
        print("Watch progress: python cloud/galaxy.py logs")
        print("When done: python cloud/galaxy.py fetch --run", run)

    def logs(self) -> None:
        self.connect()
        running = self.ssh("tmux has-session -t train", check=False, quiet=True) == 0
        print("Training is running." if running else "No training is running; showing last log.")
        self.ssh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv", check=False)
        try:
            self.ssh(f"tail -n 40 {'-f' if running else ''} {self.remote_dir}/train.log", check=False)
        except KeyboardInterrupt:
            print("\nStopped watching. The training continues on the instance.")

    def fetch(self, run: str, out: Path, step: int | None = None) -> None:
        self.connect()
        last = f"{self.remote_dir}/lerobot/outputs/{run}/checkpoints/{checkpoint(step)}"
        if self.ssh(f"test -d {last}/pretrained_model", check=False, quiet=True) != 0:
            self.ssh(f"ls {self.remote_dir}/lerobot/outputs/{run}/checkpoints", check=False)
            fail(f"run '{run}' has no checkpoint '{checkpoint(step)}' (saved ones are listed above)")
        self.ssh(f"du -shL {last}/pretrained_model*")
        self.download(last, "pretrained_model*", out)
        print(
            f"Saved to {out}. Serve it with: python cloud/galaxy.py up --adapter {out / 'pretrained_model'}"
        )

    def curves(self, out: Path) -> None:
        """Save the training log locally and print how the errors developed."""
        self.connect()
        result = subprocess.run(
            [*self.ssh_command(), f"cat {self.remote_dir}/train.log"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
        if result.returncode:
            fail(f"no training log at {self.remote_dir}/train.log")
        log = result.stdout.decode(errors="replace").replace("\r", "\n")
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.log").write_text(log, encoding="utf-8")
        print(training_table(log))
        print(f"\nFull log: {out / 'train.log'}")

    def status(self, local_port: int) -> None:
        self.connect()
        print(f"SSH OK: {self.user}@{self.host}:{self.port} (REMOTE_DIR={self.remote_dir})")
        self.ssh("nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv", check=False)
        self.ssh(f"df -h {self.remote_dir}", check=False)
        self.ssh("tmux ls 2>/dev/null || echo 'No active tmux sessions'", check=False)
        print(f"Local API on 127.0.0.1:{local_port}: {api_health(local_port) or 'not reachable'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="install LeRobot + weights without starting the server")
    p_setup.add_argument("--cpu", action="store_true", help="skip nvidia-smi check (cardless CPU host)")

    p_up = sub.add_parser("up", help="start server.py in tmux and open SSH tunnel")
    p_up.add_argument("--port", type=int, default=8000, help="local tunnel port (default 8000)")
    p_up.add_argument("--redeploy", action="store_true", help="re-upload server.py and rerun setup_remote.sh")
    p_up.add_argument("--adapter", type=Path, help="local LoRA adapter directory to upload and serve")
    p_up.add_argument("--adapter-run", help="serve the adapter of an existing run on the instance")
    p_up.add_argument("--step", type=int, help="with --adapter-run: checkpoint step to serve")
    p_up.add_argument("--text-encoder-on-cpu", action="store_true", help="save ~8 GiB VRAM on 32 GB GPUs")

    p_train = sub.add_parser("train", help="upload dataset and start LoRA training in tmux")
    p_train.add_argument("--dataset", type=Path, required=True, help="local LeRobot dataset directory")
    p_train.add_argument("--run", default="nexarm_lora", help="training run name (default nexarm_lora)")
    p_train.add_argument("--steps", type=int, help="microsteps instead of recipe default")
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="2 works on 40+ GB (A800-40G); use 1 on 32 GB (RTX 5090)",
    )
    p_train.add_argument("--resume", action="store_true", help="continue an existing run")

    sub.add_parser("logs", help="show GPU memory and follow train.log")

    p_curves = sub.add_parser("curves", help="download train.log and print loss table")
    p_curves.add_argument("--run", default="nexarm_lora", help="run name for local output directory")
    p_curves.add_argument("--out", type=Path, help="local output directory (default outputs/lora/<run>)")

    p_fetch = sub.add_parser("fetch", help="download trained LoRA adapter + processors")
    p_fetch.add_argument("--run", default="nexarm_lora", help="training run name (default nexarm_lora)")
    p_fetch.add_argument("--step", type=int, help="saved state after N steps (default last)")
    p_fetch.add_argument("--out", type=Path, help="local target (default outputs/lora/<run>)")

    p_status = sub.add_parser("status", help="check SSH, GPU, disk, tmux and local /health")
    p_status.add_argument("--port", type=int, default=8000, help="local tunnel port (default 8000)")

    args = parser.parse_args()
    deployment = GalaxyDeployment()
    if args.command == "setup":
        deployment.setup(start_server=False, skip_gpu_check=args.cpu)
    elif args.command == "up":
        deployment.up(
            args.port,
            args.redeploy,
            args.adapter,
            args.adapter_run,
            args.step,
            text_encoder_on_cpu=args.text_encoder_on_cpu,
        )
    elif args.command == "train":
        deployment.train(args.dataset, args.run, args.steps, args.resume, args.batch_size)
    elif args.command == "logs":
        deployment.logs()
    elif args.command == "curves":
        deployment.curves(args.out or Path("outputs") / "lora" / args.run)
    elif args.command == "fetch":
        deployment.fetch(args.run, args.out or Path("outputs") / "lora" / args.run, args.step)
    elif args.command == "status":
        deployment.status(args.port)


if __name__ == "__main__":
    main()
