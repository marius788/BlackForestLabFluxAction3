@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Kameratest
echo Kameratest: Live-Bild und echte Bildrate von Kamera 0 (scene) und 2 (wrist).
echo Im Fenster: q = Beenden, 1/2 = Treiber-Einstellungen der Kamera oeffnen.
echo Die Arme werden nicht angesprochen. Nicht gleichzeitig mit Teleoperation oder Aufnahme starten.
python "%~dp0..\check_cameras.py" %*
pause
