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
call "%NEXARM_CONDA%\Scripts\activate.bat" nexarm
cd /d "%~dp0finsh"
title NexArm - Python 3.12 - LeRobot (finsh)
echo NexArm-Umgebung ist aktiv. Python und pip zeigen jetzt auf die isolierte Umgebung.
cmd /k

