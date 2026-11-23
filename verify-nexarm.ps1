[CmdletBinding()]
param(
    [string]$CondaRoot = (Join-Path $env:LOCALAPPDATA "NexArm\Miniconda3-x64")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $CondaRoot "envs\nexarm\python.exe"
$ffmpegExe = Join-Path $CondaRoot "envs\nexarm\Library\bin\ffmpeg.exe"
$nexarmEnvRoot = Join-Path $CondaRoot "envs\nexarm"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "NexArm-Python fehlt: $pythonExe"
}

$env:CONDA_PREFIX = $nexarmEnvRoot
$env:CONDA_DEFAULT_ENV = "nexarm"
$env:PATH = "$nexarmEnvRoot;$nexarmEnvRoot\Scripts;$nexarmEnvRoot\Library\bin;$env:PATH"

& $pythonExe -c "import platform, sys; import torch, cv2, serial, av, rerun; from lerobot.robots.nexarm_follower import NexArmFollower; print('NEXARM_OK'); print('Python:', sys.version); print('Architektur:', platform.machine()); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python/LeRobot-Pruefung fehlgeschlagen." }

if (-not (Test-Path -LiteralPath $ffmpegExe)) {
    throw "FFmpeg fehlt: $ffmpegExe"
}
& $ffmpegExe -version | Select-Object -First 1

$driverOutput = pnputil.exe /enum-drivers 2>&1 | Out-String
if ($driverOutput -match 'CH340|CH341|wchusbserial|wch') {
    Write-Host "CH340/CH341-Treiber: gefunden" -ForegroundColor Green
} else {
    Write-Warning "CH340/CH341-Treiber wurde nicht gefunden. Morgen zuerst beide Arme anschliessen und Windows Update abwarten. Falls kein COM-Port erscheint: https://www.wch-ic.com/downloads/CH341SER_ZIP.html"
}

Write-Host "Softwarepruefung abgeschlossen." -ForegroundColor Green

