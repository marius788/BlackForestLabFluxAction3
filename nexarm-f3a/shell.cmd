@echo off
call "%~dp0_env.cmd" || (pause & exit /b 1)
title NexArm F3A - Shell (Upstream-LeRobot + NexArm-Plugin)
echo Plugin-Umgebung ist aktiv: python, lerobot-record, lerobot-teleoperate, lerobot-find-port ...
rem The path is quoted on purpose: it contains an ampersand.
echo Follower %F3A_FOLLOWER_PORT%, Leader %F3A_LEADER_PORT%, Datensaetze in "%HF_LEROBOT_HOME%"
cmd /k
