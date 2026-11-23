@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
title NexArm F3A - Inferenz (Dry-Run, KEINE Bewegung)
echo Starte FLUX-3-Action Client im DRY-RUN Modus (keine Befehle an den Arm).
echo Server: %F3A_SERVER_URL%  |  Follower: %F3A_FOLLOWER_PORT%
echo Task:   "%F3A_TASK%"
echo Log:    outputs\cloud_client.jsonl
echo.
python cloud\client.py --config config\inference.yaml --port %F3A_FOLLOWER_PORT% --url %F3A_SERVER_URL% --task "%F3A_TASK%" %*
pause
