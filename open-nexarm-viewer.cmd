@echo off
set "NEXARM_CONDA=%LOCALAPPDATA%\NexArm\Miniconda3-x64"
if not exist "%NEXARM_CONDA%\Scripts\activate.bat" (
  echo NexArm-Miniconda wurde nicht gefunden.
  echo Bitte zuerst setup-nexarm.ps1 ausfuehren.
  pause
  exit /b 1
)
call "%NEXARM_CONDA%\Scripts\activate.bat" nexarm
title NexArm - Rerun Web-Viewer
echo Rerun-Server fuer die Live-Anzeige laeuft, der Viewer oeffnet sich im Browser.
echo Falls nicht: http://localhost:9090/?url=rerun%%2Bhttp://localhost:9876/proxy
echo Dieses Fenster offen lassen, solange teleoperiert wird. Beenden mit Strg+C.
rem --serve-web instead of --web-viewer: rerun must not spawn the browser itself, otherwise the browser
rem inherits rerun's listening socket and keeps port 9876 blocked after this window is closed.
rerun --serve-web --port 9876 --web-viewer-port 9090 --server-memory-limit 1GB
