@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
call "%~dp0_viewer.cmd" || (pause & exit /b 1)
title NexArm F3A - Datenaufnahme
echo Neue Aufnahme mit %F3A_TARGET_EPISODES% Episoden: local/%F3A_DATASET%_[Zeitstempel] in Grad / Greifer 0-100 (FLUX-3-Action-Format).
echo Pfeil rechts = Episode beenden, Pfeil links = Episode neu aufnehmen, Esc = Aufnahme beenden.
echo.
lerobot-record --config_path=config/record.yaml --robot.port=%F3A_FOLLOWER_PORT% --teleop.port=%F3A_LEADER_PORT% --dataset.repo_id=local/%F3A_DATASET% --dataset.num_episodes=%F3A_TARGET_EPISODES% %F3A_DISPLAY%
echo.
echo Jetzt pruefen: check-dataset.cmd
pause
