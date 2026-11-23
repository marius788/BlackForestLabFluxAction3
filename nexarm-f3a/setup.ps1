# Reproducible install of the FLUX-3-Action recording environment:
# upstream LeRobot (pinned by the plugin's uv.lock) + lerobot_robot_nexarm plugin in plugin\.venv.
# Usage (PowerShell in this folder):  Set-ExecutionPolicy -Scope Process Bypass; .\setup.ps1
[CmdletBinding()]
param(
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$plugin = Join-Path $root "plugin"
$repoUrl = "https://github.com/mariorch22/lerobot-nexarm.git"
# x64 Python on purpose: on Windows ARM64 it runs in emulation, but all wheels (torch, torchcodec,
# rerun, opencv) exist for it. The old Hiwonder fork must NOT be installed into this environment.
$python = "cpython-3.12-windows-x86_64-none"

function Assert-Exit {
    param([string]$Action)
    if ($LASTEXITCODE -ne 0) { throw "$Action ist mit Exitcode $LASTEXITCODE fehlgeschlagen." }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv fehlt. Installation: https://docs.astral.sh/uv/getting-started/installation/"
}

if (-not (Test-Path -LiteralPath (Join-Path $plugin "pyproject.toml"))) {
    Write-Host "==> Plugin klonen" -ForegroundColor Cyan
    git clone $repoUrl $plugin
    Assert-Exit "git clone"
}

Write-Host "==> Python 3.12 (x64) bereitstellen" -ForegroundColor Cyan
uv python install $python
Assert-Exit "uv python install"

Write-Host "==> Umgebung nach uv.lock installieren" -ForegroundColor Cyan
Push-Location $plugin
try {
    # The upstream LFS test images are not needed to build the package.
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    uv sync --locked --python $python
    Assert-Exit "uv sync"

    if (-not $SkipTests) {
        Write-Host "==> Plugin-Tests (nur Mocks, keine Hardware)" -ForegroundColor Cyan
        uv run --locked pytest -q
        Assert-Exit "pytest"
    }

    Write-Host "==> Importtest" -ForegroundColor Cyan
    & (Join-Path $plugin ".venv\Scripts\python.exe") -c "import lerobot, torch, cv2, av, rerun, lerobot_robot_nexarm; from lerobot.robots import RobotConfig; assert 'nexarm_follower' in RobotConfig.get_known_choices(); print('NEXARM_F3A_OK lerobot', lerobot.__version__, 'torch', torch.__version__)"
    Assert-Exit "Importtest"
} finally {
    Pop-Location
}

Write-Host "Fertig. Weiter mit 1-calibrate-follower.cmd (siehe README.md)." -ForegroundColor Green
