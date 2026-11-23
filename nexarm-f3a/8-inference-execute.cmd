@echo off
call "%~dp0_env.cmd" hw calib || (pause & exit /b 1)
title NexArm F3A - Inferenz (EXECUTE - Arm bewegt sich!)
echo ====================================================================
echo ACHTUNG: Dieser Lauf SENDET BEFEHLE an den NexArm (%F3A_FOLLOWER_PORT%)!
echo Bitte vorher pruefen:
echo   1. Arbeitsbereich frei und eine Person am Netzschalter?
echo   2. Koordinatenlimits in der Hiwonder-Software gesetzt?
echo   3. 7-inference-dryrun.cmd ohne Auffaelligkeiten gelaufen?
echo ====================================================================
echo Server: %F3A_SERVER_URL%
echo Task:   "%F3A_TASK%"
echo.
set "CONFIRM="
set /p "CONFIRM=Wirklich mit Bewegung starten? (tippe JA): "
if /i not "%CONFIRM%"=="JA" (
  echo Abgebrochen. Es wurden keine Befehle gesendet.
  pause
  exit /b 0
)
python cloud\client.py --config config\inference.yaml --port %F3A_FOLLOWER_PORT% --url %F3A_SERVER_URL% --task "%F3A_TASK%" --execute %*
pause
