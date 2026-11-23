@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
call "%~dp0_viewer.cmd" || (pause & exit /b 1)
title NexArm F3A - Teleoperation
echo Teleoperation mit Kameras. Der Follower faehrt sofort in die Haltung des Leaders. Beenden mit Strg+C.
echo.
lerobot-teleoperate --config_path=config/teleop.yaml --robot.port=%F3A_FOLLOWER_PORT% --teleop.port=%F3A_LEADER_PORT% %F3A_DISPLAY%
pause
