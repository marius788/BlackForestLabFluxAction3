@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Kalibrierung Leader
if not exist "%~dp0calibration\follower\follower.json" (
  echo Die Follower-Kalibrierung fehlt. Bitte zuerst 1-calibrate-follower.cmd ausfuehren.
  pause
  exit /b 1
)
echo Kalibrierung LEADER an %F3A_LEADER_PORT%. Ablauf wie beim Follower:
echo.
echo  1. Leader in die SO-101-Mittelpose bringen, ENTER = Nullpunkt speichern.
echo  2. Jedes Gelenk nacheinander langsam durch seinen ganzen Bereich bewegen, ENTER = naechstes Gelenk.
echo  In diesem Dialog nur ENTER druecken, nichts eintippen. Strg+C bricht ab, ohne zu speichern.
echo.
lerobot-calibrate --teleop.type=nexarm_leader --teleop.port=%F3A_LEADER_PORT% --teleop.id=leader --teleop.calibration_dir=calibration/leader --teleop.follower_calibration_path=calibration/follower/follower.json
if errorlevel 1 (
  echo.
  echo Die Kalibrierung wurde ABGEBROCHEN, es wurde nichts gespeichert. Bitte erneut starten.
  echo Bei "encoder transition 4095 - 0": Das Gelenk wurde ueber die Encodergrenze gedreht.
  echo Mit 0-show-raw-leader.cmd die Rohwerte ansehen und unter 4095 / ueber 0 bleiben.
  pause
  exit /b 1
)
echo.
echo Ergebnis: %~dp0calibration\leader\leader.json
echo Danach: 3-teleop-no-cameras.cmd (Richtungen pruefen)
pause
