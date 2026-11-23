@echo off
call "%~dp0_env.cmd" || (pause & exit /b 1)
title NexArm F3A - MuJoCo Simulation
set "SIM_SCENE=%~dp0third_party\khanhthanhdev-lerobot-nexarm\sim\description\mjcf\scene.xml"
if not exist "%SIM_SCENE%" (
  echo MuJoCo Arm-Modell fehlt unter:
  echo   %SIM_SCENE%
  echo.
  echo Einmalig im Ordner nexarm-f3a ausfuehren:
  echo   git clone --depth 1 https://github.com/khanhthanhdev/lerobot-nexarm.git third_party\khanhthanhdev-lerobot-nexarm
  echo   git -C third_party\khanhthanhdev-lerobot-nexarm lfs pull --include "sim/assets/**"
  pause
  exit /b 1
)
where uv >nul 2>&1 || (
  echo uv wurde nicht im PATH gefunden. Bitte installieren: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
echo Starte MuJoCo-Simulation gegen %F3A_SERVER_URL% ...
echo Video wird gespeichert unter: outputs\sim.mp4
uv run --project plugin --with mujoco python cloud\client.py --sim "%SIM_SCENE%" --video outputs\sim.mp4 --url %F3A_SERVER_URL% --task "%F3A_TASK%" --execute %*
pause
