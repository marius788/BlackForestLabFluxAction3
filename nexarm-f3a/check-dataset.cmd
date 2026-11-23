@echo off
call "%~dp0_env.cmd" || (pause & exit /b 1)
title NexArm F3A - Datensatz pruefen
rem Without arguments: newest local/%F3A_DATASET%_* dataset. Optional: dataset folder name or full path.
python "%~dp0tools\check_f3a_dataset.py" --prefix %F3A_DATASET% %*
pause
