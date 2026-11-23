@echo off
set "NEXARM_CONDA=%LOCALAPPDATA%\NexArm\Miniconda3-x64"
rem Datasets (local\...) and calibration live on Google Drive instead of %USERPROFILE%\.cache\huggingface\lerobot.
set "HF_LEROBOT_HOME=H:\Meine Ablage\02_ARBEIT_&_PROJEKTE\FluxAction\lerobot"
set "NEXARM_DATASET=nexarm_stack_boxes"
set "NEXARM_TARGET_EPISODES=50"
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
call "%NEXARM_CONDA%\Scripts\activate.bat" nexarm
set "RESUME_NAME="
for /f "tokens=1,2*" %%a in ('python "%~dp0nexarm_resume_info.py" %NEXARM_DATASET% %NEXARM_TARGET_EPISODES%') do (
  set "RESUME_NAME=%%a"
  set "RESUME_MISSING=%%b"
  set "RESUME_ROOT=%%c"
)
if not defined RESUME_NAME (
  echo Keine Aufnahme local/%NEXARM_DATASET%_* gefunden. Bitte zuerst nexarm-record.cmd ausfuehren.
  pause
  exit /b 1
)
if "%RESUME_MISSING%"=="0" (
  echo local/%RESUME_NAME% hat bereits %NEXARM_TARGET_EPISODES% Episoden.
  pause
  exit /b 0
)
call "%~dp0ensure-nexarm-viewer.cmd" || (pause & exit /b 1)
cd /d "%~dp0finsh"
title NexArm - Datenaufnahme fortsetzen
echo Setze local/%RESUME_NAME% fort, es fehlen noch %RESUME_MISSING% Episoden.
echo Pfeil rechts = Episode beenden, Pfeil links = Episode neu aufnehmen, Esc = Aufnahme beenden.
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm/record.yaml --resume=true --dataset.repo_id=local/%RESUME_NAME% "--dataset.root=%RESUME_ROOT%" --dataset.num_episodes=%RESUME_MISSING%
pause
