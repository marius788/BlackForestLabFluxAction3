#!/usr/bin/env bash
# Runs on the rented AI Galaxy GPU (or CPU) instance. Installs LeRobot with FLUX 3 Action
# at the exact project pin, downloads the model and optionally starts server.py.
# Adapted from mariorch22/lerobot-nexarm (cloud/setup_remote.sh, Apache-2.0) for AI Galaxy & China mirrors.
# Safe to run again: finished steps are skipped or are quick no-ops.
# Expects HF_TOKEN in the environment (read access to both gated F3A repos).
set -euo pipefail

LEROBOT_COMMIT=2f05d8f64d62a865c368a9fdf0ba3d8423e28035   # same pin as the NexArm plugin
LEROBOT_GIT_URL="${LEROBOT_GIT_URL:-https://github.com/huggingface/lerobot.git}"
POLICY_REVISION=c9e13b2aca6a0a472b3ea03fd90cff32d3e85849
ENCODER_REVISION=62878e2925e59b7a89ec14463ce89932624c490d

REMOTE_DIR="${REMOTE_DIR:-/workspace}"
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export GIT_LFS_SKIP_SMUDGE=1   # upstream test images are not needed
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CACHE_DIR="$REMOTE_DIR/.cache/uv"
export HF_HOME="$REMOTE_DIR/.cache/huggingface"
export TMPDIR="$REMOTE_DIR/.tmp"

mkdir -p "$REMOTE_DIR/lerobot" "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"

echo "== disk check on $REMOTE_DIR"
df -h "$REMOTE_DIR"
AVAIL_KB=$(df -Pk "$REMOTE_DIR" | awk 'NR==2 {print $4}')
if [ "${AVAIL_KB:-0}" -lt $((60 * 1024 * 1024)) ]; then
  echo "WARNING: less than 60 GB free on $REMOTE_DIR (environment + weights need ~45-60 GB)." >&2
fi

if [ "${SKIP_GPU_CHECK:-0}" != 1 ]; then
  echo "== GPU / CUDA driver check"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Pass SKIP_GPU_CHECK=1 on a CPU-only host." >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)
  echo "Reported driver CUDA Version: ${CUDA_VER:-unknown}"
  if [ -n "${CUDA_VER:-}" ]; then
    python3 - "$CUDA_VER" <<'EOF'
import sys
parts = tuple(int(p) for p in sys.argv[1].split(".")[:2])
if parts < (12, 8):
    sys.exit(
        f"ERROR: Host driver reports CUDA {sys.argv[1]} (< 12.8). "
        "torch==2.11.0+cu128 + natten require driver >= 570 (CUDA >= 12.8). "
        "End (结束) this instance on AI Galaxy and choose RTX 5090 or another >=12.8 node."
    )
EOF
  fi
fi

echo "== system packages"
missing_pkgs=()
for cmd in git ffmpeg curl tmux; do
  command -v "$cmd" >/dev/null 2>&1 || missing_pkgs+=("$cmd")
done
if [ "${#missing_pkgs[@]}" -gt 0 ]; then
  apt-get update -qq
  apt-get install -y -qq "${missing_pkgs[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  if [ -n "${UV_DEFAULT_INDEX:-}" ]; then
    python3 -m pip install -q -U uv -i "$UV_DEFAULT_INDEX" || curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    python3 -m pip install -q -U uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi

echo "== LeRobot at $LEROBOT_COMMIT"
cd "$REMOTE_DIR/lerobot"
# Fetch only the pinned commit: quick, and it repairs a checkout that was interrupted earlier.
if [ "$(git rev-parse HEAD 2>/dev/null || true)" != "$LEROBOT_COMMIT" ]; then
  git init -q
  git fetch --progress --depth 1 "$LEROBOT_GIT_URL" "$LEROBOT_COMMIT"
  git checkout -q FETCH_HEAD
fi
[ -d .venv ] || uv venv --python 3.12
# shellcheck source=/dev/null
source .venv/bin/activate

echo "== PyTorch 2.11.0+cu128"
if [ -n "${PYTORCH_INDEX_URL:-}" ]; then
  uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url "$PYTORCH_INDEX_URL" \
    || uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
else
  uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
fi

echo "== LeRobot + flux3 + peft + server deps"
uv pip install -e '.[training,flux3,peft,diffusion]'
uv pip install fastapi uvicorn pillow

echo "== natten==0.21.6+torch2110cu128"
local_natten=$(find "$REMOTE_DIR/wheels" -maxdepth 1 -name 'natten-0.21.6+torch2110cu128*.whl' 2>/dev/null | head -n 1 || true)
if [ -n "$local_natten" ]; then
  uv pip install "$local_natten"
else
  natten_ok=0
  for attempt in 1 2 3; do
    if uv pip install 'natten==0.21.6+torch2110cu128' --find-links "${NATTEN_FIND_LINKS:-https://whl.natten.org/}"; then
      natten_ok=1
      break
    fi
    echo "natten install attempt $attempt failed; retrying in 3s..."
    sleep 3
  done
  [ "$natten_ok" = 1 ] || { echo "ERROR: natten install failed after 3 attempts." >&2; exit 1; }
fi

ffmpeg -version | head -n 1
python -c 'import torch; ok = torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else "CPU-ONLY")'

if [ -n "${HF_ENDPOINT:-}" ] && [ "$HF_ENDPOINT" != "https://huggingface.co" ]; then
  # huggingface-hub==2.4.0 bundles hf-xet; disable Xet when using hf-mirror/proxies.
  export HF_HUB_DISABLE_XET=1
fi

echo "== checking gated Hugging Face access"
python - <<'EOF'
import os
import sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("ERROR: HF_TOKEN is missing.")
api = HfApi(token=token)
for repo_id, rev in [
    ("black-forest-labs/flux-3-action-so101", "c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"),
    ("black-forest-labs/flux-3-action-base", "62878e2925e59b7a89ec14463ce89932624c490d"),
]:
    try:
        api.model_info(repo_id, revision=rev)
        print(f"HF access OK: {repo_id}")
    except Exception as exc:
        sys.exit(
            f"ERROR: cannot access {repo_id} ({type(exc).__name__}: {exc}). "
            f"Accept the license on https://huggingface.co/{repo_id} and check HF_TOKEN."
        )
EOF

echo "== model download (about 25 GB on the first run)"
hf download black-forest-labs/flux-3-action-so101 \
  --revision "$POLICY_REVISION" \
  --local-dir models/flux-3-action-so101
hf download black-forest-labs/flux-3-action-base \
  --revision "$ENCODER_REVISION" \
  --include 'video_vae.safetensors' --include 'text_encoder/*' \
  --local-dir models/flux-3-action-base
python - <<'EOF'
import json
config = json.load(open("models/flux-3-action-so101/config.json"))
for key in ("camera_keys", "camera_layout", "fps", "chunk_size", "n_action_steps", "n_obs_steps"):
    print(key, config.get(key))
assert config["fps"] == 30
EOF

tmux kill-session -t f3a 2>/dev/null || true
if [ "${START_SERVER:-1}" != 1 ]; then
  echo "== setup done (server not started)"
  exit 0
fi
echo "== start server in tmux session 'f3a' (log: $REMOTE_DIR/server.log)"
# The server only reads local files; it does not get the HF token.
tmux new-session -d -s f3a \
  "env -u HF_TOKEN HF_HUB_OFFLINE=1 $REMOTE_DIR/lerobot/.venv/bin/python $REMOTE_DIR/server.py ${SERVER_ARGS:-} > $REMOTE_DIR/server.log 2>&1"
echo "== setup done"
