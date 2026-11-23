@echo off
rem Helper for 4-teleop.cmd / 5-record.cmd / 6-record-continue.cmd: sets F3A_DISPLAY.
rem The live viewer is OFF by default (F3A_VIEWER in _env.cmd). LeRobot writes to the viewer from
rem inside the 30 Hz control loop; when the browser viewer does not take the data fast enough,
rem the rerun server blocks and the arm stops following (seen on 2026-11-13).
set "F3A_DISPLAY=--display_data=false"
if not "%F3A_VIEWER%"=="1" exit /b 0
call "%~dp0ensure-viewer.cmd" || exit /b 1
rem With display_ip/display_port LeRobot sends JPEG instead of raw frames: far less data for the viewer.
set "F3A_DISPLAY=--display_data=true --display_ip=127.0.0.1 --display_port=%F3A_VIEWER_PORT%"
exit /b 0
