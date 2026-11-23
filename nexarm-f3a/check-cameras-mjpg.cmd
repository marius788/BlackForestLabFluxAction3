@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Kameratest MJPG
echo Kameratest mit Pixelformat MJPG statt YUY2 (weniger USB-Bandbreite pro Kamera).
echo Vergleichen mit check-cameras.cmd: Erreichen beide Kameras hier 28-30 fps?
echo Im Fenster: q = Beenden, 1/2 = Treiber-Einstellungen der Kamera oeffnen.
python "%~dp0..\check_cameras.py" --fourcc MJPG %*
pause
