@echo off
call "%~dp0_env.cmd" || (pause & exit /b 1)
title NexArm F3A - MuJoCo Chess Live Studio
set "SIM_SCENE=%~dp0..\Schach\chess_scene.xml"
if not exist "%SIM_SCENE%" (
  set "SIM_SCENE=%~dp0third_party\khanhthanhdev-lerobot-nexarm\sim\description\mjcf\scene.xml"
)
if not exist "%~dp0third_party\khanhthanhdev-lerobot-nexarm\sim\description\mjcf\nexarm.xml" (
  echo MuJoCo Arm-Modell fehlt unter:
  echo   %~dp0third_party\khanhthanhdev-lerobot-nexarm\sim\description\mjcf\nexarm.xml
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
echo Starte NexArm MuJoCo Chess Studio im Browser (http://127.0.0.1:8080) ...
echo Szene: %SIM_SCENE%
echo FLUX.3 Action Server URL: %F3A_SERVER_URL%
uv run --project plugin --with mujoco --with fastapi --with uvicorn python cloud\sim_web.py --sim "%SIM_SCENE%" --config config\inference.yaml --url %F3A_SERVER_URL% --task "move white pawn e2 to e4" %*
pause
