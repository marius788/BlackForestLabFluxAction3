@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
title NexArm F3A - Teleoperation ohne Kameras
echo Erster Test nach der Kalibrierung, ohne Kameras.
echo Der Follower faehrt sofort in die Haltung des Leaders: Arbeitsbereich frei halten!
echo Pruefen: Jedes Gelenk folgt in die richtige Richtung, der Greifer oeffnet und schliesst richtig.
echo Beenden mit Strg+C.
echo.
lerobot-teleoperate --config_path=config/teleop_no_cameras.yaml --robot.port=%F3A_FOLLOWER_PORT% --teleop.port=%F3A_LEADER_PORT%
pause
