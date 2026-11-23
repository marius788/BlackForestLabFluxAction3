@echo off
call "%~dp0_env.cmd" hw || (pause & exit /b 1)
title NexArm F3A - Kalibrierung Follower
echo Kalibrierung FOLLOWER an %F3A_FOLLOWER_PORT%. Das Drehmoment wird abgeschaltet: Arm vorher abstuetzen!
echo.
echo  1. Arm in die SO-101-Mittelpose bringen, ENTER = Nullpunkt speichern.
echo  2. Jedes Gelenk nacheinander langsam durch seinen ganzen Bereich bewegen, ENTER = naechstes Gelenk.
echo  In diesem Dialog nur ENTER druecken, nichts eintippen. Strg+C bricht ab, ohne zu speichern.
echo.
lerobot-calibrate --robot.type=nexarm_follower --robot.port=%F3A_FOLLOWER_PORT% --robot.id=follower --robot.calibration_dir=calibration/follower
if errorlevel 1 (
  echo.
  echo Die Kalibrierung wurde ABGEBROCHEN, es wurde nichts gespeichert. Bitte erneut starten.
  echo Bei "encoder transition 4095 - 0": Das Gelenk wurde ueber die Encodergrenze gedreht.
  echo Mit 0-show-raw-follower.cmd die Rohwerte ansehen und unter 4095 / ueber 0 bleiben.
  pause
  exit /b 1
)
echo.
echo Ergebnis: %~dp0calibration\follower\follower.json
echo Danach: 2-calibrate-leader.cmd
pause
