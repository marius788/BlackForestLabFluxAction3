@echo off
set "NEXARM_CONDA=%LOCALAPPDATA%\NexArm\Miniconda3-x64"
rem Datasets (local\...) and calibration live on Google Drive instead of %USERPROFILE%\.cache\huggingface\lerobot.
set "HF_LEROBOT_HOME=H:\Meine Ablage\02_ARBEIT_&_PROJEKTE\FluxAction\lerobot"
if not exist "%NEXARM_CONDA%\Scripts\activate.bat" (
  echo NexArm-Miniconda wurde nicht gefunden.
  echo Bitte zuerst setup-nexarm.ps1 ausfuehren.
  pause
  exit /b 1
)
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -match 'lerobot_(teleoperate|record|rollout)' }) { exit 1 }" || (
  echo Es laeuft bereits eine Teleoperation oder Aufnahme und belegt die Arme.
  echo Bitte dort zuerst mit Strg+C beenden und dann neu starten.
  pause
  exit /b 1
)
call "%~dp0ensure-nexarm-viewer.cmd" || (pause & exit /b 1)
call "%NEXARM_CONDA%\Scripts\activate.bat" nexarm
cd /d "%~dp0finsh"
title NexArm - Teleoperation
echo Teleoperation startet. Follower faehrt sofort in die Haltung des Leaders. Beenden mit Strg+C.
python -m lerobot.scripts.lerobot_teleoperate --config_path=examples/nexarm/teleoperate.yaml
pause
