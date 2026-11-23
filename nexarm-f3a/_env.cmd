@echo off
rem Common environment for all nexarm-f3a launchers.
rem Usage inside a launcher:  call "%~dp0_env.cmd" [hw] || (pause & exit /b 1)
rem   hw = additionally refuse to start while another LeRobot process holds the arms.
rem   calib (second argument) = additionally require both calibration files.

rem ---- Edit here when Windows assigns other COM ports (Device Manager -> USB-SERIAL CH340) ----
set "F3A_FOLLOWER_PORT=COM10"
set "F3A_LEADER_PORT=COM15"
rem ---- Datasets live on Google Drive, same place as before. Calibration stays in this folder. ----
set "HF_LEROBOT_HOME=H:\Meine Ablage\02_ARBEIT_&_PROJEKTE\FluxAction\lerobot"
rem ---- Dataset name prefix (a timestamp is appended per recording session) and target size ----
set "F3A_DATASET=nexarm_f3a_stack_boxes"
set "F3A_TARGET_EPISODES=20"
rem ---- Default English task instruction and inference server URL for FLUX 3 Action ----
set "F3A_TASK=put the small box on top of the big box"
set "F3A_SERVER_URL=http://127.0.0.1:8000"
rem ---- Live camera viewer in the browser during teleoperation / recording: 0 = off, 1 = on ----
rem Off by default: a stalled viewer blocks the control loop and the arm stops following.
rem Check the camera picture with check-cameras.cmd instead.
set "F3A_VIEWER=0"
rem Own ports, so an old browser tab of the previous setup (9876 / 9090) cannot attach.
set "F3A_VIEWER_PORT=9886"
set "F3A_VIEWER_WEB_PORT=9096"

set "F3A_ROOT=%~dp0"
set "F3A_VENV=%~dp0plugin\.venv"

if not exist "%F3A_VENV%\Scripts\lerobot-record.exe" (
  echo Die Plugin-Umgebung fehlt: %F3A_VENV%
  echo Bitte zuerst setup.ps1 in diesem Ordner ausfuehren.
  exit /b 1
)
if not exist "%HF_LEROBOT_HOME%" (
  echo Der Datensatz-Ordner ist nicht erreichbar:
  rem Quoted on purpose: the path contains an ampersand.
  echo   "%HF_LEROBOT_HOME%"
  echo Google Drive starten oder HF_LEROBOT_HOME in _env.cmd anpassen.
  exit /b 1
)
if /i "%~1"=="hw" (
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|lerobot-)' -and $_.CommandLine -match 'lerobot[-_](teleoperate|record|rollout|calibrate)' }) { exit 1 }"
  if errorlevel 1 (
    echo Es laeuft bereits eine Teleoperation, Aufnahme oder Kalibrierung und belegt die Arme.
    echo Bitte dort zuerst mit Strg+C beenden und dann neu starten.
    exit /b 1
  )
)
if /i "%~2"=="calib" (
  if not exist "%~dp0calibration\follower\follower.json" (
    echo Die Follower-Kalibrierung fehlt noch.
    echo Bitte zuerst 1-calibrate-follower.cmd und danach 2-calibrate-leader.cmd ausfuehren.
    exit /b 1
  )
  if not exist "%~dp0calibration\leader\leader.json" (
    echo Die Leader-Kalibrierung fehlt noch.
    echo Bitte zuerst 2-calibrate-leader.cmd ausfuehren.
    exit /b 1
  )
)
set "PATH=%F3A_VENV%\Scripts;%PATH%"
rem All configs use paths relative to this folder (config\..., calibration\...).
cd /d "%~dp0"
exit /b 0

