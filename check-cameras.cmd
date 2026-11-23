@echo off
set "NEXARM_CONDA=%LOCALAPPDATA%\NexArm\Miniconda3-x64"
if not exist "%NEXARM_CONDA%\Scripts\activate.bat" (
  echo NexArm-Miniconda wurde nicht gefunden.
  echo Bitte zuerst setup-nexarm.ps1 ausfuehren.
  pause
  exit /b 1
)
call "%NEXARM_CONDA%\Scripts\activate.bat" nexarm
title NexArm - Kameratest
echo Kameratest: zeigt die echte Bildrate von Kamera 0 und 2.
echo Im Fenster: q = Beenden, 1/2 = Treiber-Einstellungen der Kamera oeffnen.
python "%~dp0check_cameras.py" %*
pause
