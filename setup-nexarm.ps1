[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [string]$CondaRoot = (Join-Path $env:LOCALAPPDATA "NexArm\Miniconda3-x64"),
    [string]$RepoRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = (Get-Location).Path
}
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Join-Path $scriptRoot "lerobot-nexarm"
}

$minicondaName = "Miniconda3-py311_25.7.0-2-Windows-x86_64.exe"
$minicondaUrl = "https://repo.anaconda.com/miniconda/$minicondaName"
$minicondaSha256 = "95030b3bbdbabf23dea1f7332dc1b03d9860fd240887c15b10f3ad469cbce9dc"
$repoUrl = "https://github.com/Hiwonder-official/lerobot-nexarm.git"
$downloadRoot = Join-Path $scriptRoot ".downloads"
$installerPath = Join-Path $downloadRoot $minicondaName
$condaExe = Join-Path $CondaRoot "Scripts\conda.exe"
$pythonExe = Join-Path $CondaRoot "envs\nexarm\python.exe"
$ffmpegExe = Join-Path $CondaRoot "envs\nexarm\Library\bin\ffmpeg.exe"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-LastExitCode {
    param([string]$Action)
    $exitCodeVariable = Get-Variable -Name LASTEXITCODE -Scope 1 -ErrorAction SilentlyContinue
    if ($null -eq $exitCodeVariable) {
        return
    }
    if ([int]$exitCodeVariable.Value -ne 0) {
        throw "$Action ist mit Exitcode $($exitCodeVariable.Value) fehlgeschlagen."
    }
}

function Get-VerifiedFile {
    param(
        [string]$Uri,
        [string]$Destination,
        [string]$Sha256
    )

    if (Test-Path -LiteralPath $Destination) {
        $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $Sha256) {
            Write-Host "Bereits vorhanden und geprueft: $Destination"
            return
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) {
        Remove-Item -LiteralPath $Destination -Force
        throw "SHA256-Pruefung fehlgeschlagen fuer $Destination."
    }
}

Write-Step "System und vorhandenes Python pruefen"
Write-Host "Windows-Prozessorarchitektur: $env:PROCESSOR_ARCHITECTURE"
$existingPython = Get-Command python -ErrorAction SilentlyContinue
if ($existingPython) {
    Write-Host "Vorhandenes Python bleibt unangetastet: $($existingPython.Source)"
} else {
    Write-Host "Vorhandenes Python ist nicht im PATH. Es wird nicht veraendert."
}
if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    Write-Host "ARM64 erkannt: Fuer NexArm wird absichtlich die vom Hersteller geforderte x64-Umgebung verwendet."
} elseif ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    throw "Nicht unterstuetzte Windows-Architektur: $env:PROCESSOR_ARCHITECTURE"
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $condaExe)) {
    Write-Step "Offizielles Miniconda x64 aus der Hiwonder-Anleitung herunterladen"
    Get-VerifiedFile -Uri $minicondaUrl -Destination $installerPath -Sha256 $minicondaSha256

    Write-Step "Miniconda nur fuer diesen Benutzer installieren"
    New-Item -ItemType Directory -Path (Split-Path -Parent $CondaRoot) -Force | Out-Null
    $installerArguments = @(
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/S",
        "/D=$CondaRoot"
    )
    $installerProcess = Start-Process -FilePath $installerPath -ArgumentList $installerArguments -Wait -PassThru -WindowStyle Hidden
    if ($installerProcess.ExitCode -ne 0) {
        throw "Miniconda-Installation ist mit Exitcode $($installerProcess.ExitCode) fehlgeschlagen."
    }
}

if (-not (Test-Path -LiteralPath $condaExe)) {
    throw "Conda wurde nicht unter '$CondaRoot' gefunden."
}

Write-Step "Isolierte NexArm-Umgebung mit Python 3.12 erstellen"
$envExists = (& $condaExe env list --json | ConvertFrom-Json).envs -contains (Join-Path $CondaRoot "envs\nexarm")
if (-not $envExists) {
    & $condaExe create --override-channels -c conda-forge -n nexarm python=3.12 pip -y
    Assert-LastExitCode "Erstellen der Conda-Umgebung"
}

# Aktivierung nur fuer diesen Setup-Prozess; die globale Python-Installation bleibt unveraendert.
$nexarmEnvRoot = Join-Path $CondaRoot "envs\nexarm"
$env:CONDA_PREFIX = $nexarmEnvRoot
$env:CONDA_DEFAULT_ENV = "nexarm"
$env:PATH = "$nexarmEnvRoot;$nexarmEnvRoot\Scripts;$nexarmEnvRoot\Library\bin;$env:PATH"

Write-Step "FFmpeg fuer LeRobot-Datensaetze installieren"
& $condaExe install --override-channels -c conda-forge -n nexarm ffmpeg=7.1.1 -y
Assert-LastExitCode "FFmpeg-Installation"

Write-Step "Offiziellen Hiwonder-NexArm-Quellcode bereitstellen"
if (-not (Test-Path -LiteralPath $RepoRoot)) {
    & git clone --depth 1 $repoUrl $RepoRoot
    Assert-LastExitCode "Klonen des NexArm-Repositories"
} elseif (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "pyproject.toml"))) {
    throw "'$RepoRoot' existiert, ist aber kein NexArm-Repository."
} else {
    Write-Host "Vorhandenes Repository wird verwendet und nicht ueberschrieben: $RepoRoot"
}

Write-Step "pip-Werkzeuge aktualisieren"
& $pythonExe -m pip install --upgrade pip setuptools wheel
Assert-LastExitCode "Aktualisieren der pip-Werkzeuge"

Write-Step "NexArm, Datensatz-, Hardware- und Visualisierungs-Abhaengigkeiten installieren"
Push-Location $RepoRoot
try {
    $pyproject = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot "pyproject.toml")
    if ($pyproject -match '(?m)^nexarm\s*=') {
        & $pythonExe -m pip install -e ".[nexarm,viz]"
        Assert-LastExitCode "Installation der NexArm- und Visualisierungs-Abhaengigkeiten"
        & $pythonExe -m pip install "lerobot[dataset]"
        Assert-LastExitCode "Installation der Datensatz-Abhaengigkeiten"
        if (-not $SkipTraining) {
            & $pythonExe -m pip install -e ".[nexarm,training]"
            Assert-LastExitCode "Installation der Trainings-Abhaengigkeiten"
        }
    } else {
        # Neuere Versionen des offiziellen Forks haben die NexArm-Treiber bereits im Paket.
        $extras = if ($SkipTraining) { ".[core_scripts]" } else { ".[core_scripts,training]" }
        & $pythonExe -m pip install -e $extras
        Assert-LastExitCode "Installation der LeRobot-Abhaengigkeiten"
    }
} finally {
    Pop-Location
}

Write-Step "Installation pruefen"
& $pythonExe -c "import platform, sys; import torch, cv2, serial, av, rerun; from lerobot.robots.nexarm_follower import NexArmFollower; print('NEXARM_OK'); print('Python:', sys.version); print('Python-Architektur:', platform.machine()); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"
Assert-LastExitCode "NexArm-Importtest"
& $ffmpegExe -version | Select-Object -First 1

Write-Step "Fertig"
Write-Host "NexArm-Repository: $RepoRoot"
Write-Host "NexArm-Python:     $pythonExe"
Write-Host "Start: Doppelklick auf open-nexarm-shell.cmd"
Write-Host "Hardwaretest morgen: python -m lerobot.scripts.lerobot_find_port"

