@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Rohwerte Leader
echo Zeigt die Rohwerte des LEADERS an %F3A_LEADER_PORT%. Es wird nichts bewegt und nichts gespeichert.
echo Beenden mit Strg+C.
echo.
python "%~dp0tools\show_raw.py" leader %F3A_LEADER_PORT%
pause
