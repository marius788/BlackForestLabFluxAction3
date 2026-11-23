@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
set "RESUME_NAME="
for /f "tokens=1,2*" %%a in ('python "%~dp0tools\resume_info.py" %F3A_DATASET% %F3A_TARGET_EPISODES%') do (
  set "RESUME_NAME=%%a"
  set "RESUME_MISSING=%%b"
  set "RESUME_ROOT=%%c"
)
if not defined RESUME_NAME (
  echo Keine Aufnahme local/%F3A_DATASET%_* gefunden. Bitte zuerst 5-record.cmd ausfuehren.
  pause
  exit /b 1
)
if "%RESUME_MISSING%"=="0" (
  echo local/%RESUME_NAME% hat bereits %F3A_TARGET_EPISODES% Episoden.
  pause
  exit /b 0
)
call "%~dp0_viewer.cmd" || (pause & exit /b 1)
title NexArm F3A - Datenaufnahme fortsetzen
echo Setze local/%RESUME_NAME% fort, es fehlen noch %RESUME_MISSING% Episoden.
echo Pfeil rechts = Episode beenden, Pfeil links = Episode neu aufnehmen, Esc = Aufnahme beenden.
echo.
lerobot-record --config_path=config/record.yaml --robot.port=%F3A_FOLLOWER_PORT% --teleop.port=%F3A_LEADER_PORT% --resume=true --dataset.repo_id=local/%RESUME_NAME% "--dataset.root=%RESUME_ROOT%" --dataset.num_episodes=%RESUME_MISSING% %F3A_DISPLAY%
pause
