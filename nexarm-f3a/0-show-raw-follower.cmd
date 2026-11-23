@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Rohwerte Follower
echo Zeigt die Rohwerte des FOLLOWERS an %F3A_FOLLOWER_PORT%. Das Drehmoment wird abgeschaltet: Arm vorher abstuetzen!
echo Es wird nichts bewegt und nichts gespeichert. Beenden mit Strg+C.
echo.
python "%~dp0tools\show_raw.py" follower %F3A_FOLLOWER_PORT%
pause
