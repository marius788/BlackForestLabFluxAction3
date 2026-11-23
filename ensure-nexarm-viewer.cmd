@echo off
rem Helper for nexarm-teleop.cmd / nexarm-record.cmd / nexarm-record-continue.cmd: starts open-nexarm-viewer.cmd in its own window
rem unless the rerun server (gRPC 9876 + web viewer 9090) is already listening, then waits until it accepts connections
rem and opens the web viewer in the browser.
call :is_listening
if not errorlevel 1 (
  echo Kamera-Viewer laeuft bereits. Falls der Browser-Tab geschlossen ist:
  echo   http://localhost:9090/?url=rerun%%2Bhttp://localhost:9876/proxy
  exit /b 0
)
call :port_blocked
if not errorlevel 1 (
  echo Port 9876 ist belegt, aber der Viewer auf Port 9090 laeuft nicht.
  echo Meist haelt ein Browserfenster aus einem frueheren Viewer-Start den Port fest.
  echo Bitte den Browser komplett schliessen und neu starten.
  exit /b 1
)
echo Starte Kamera-Viewer ...
start "NexArm - Rerun Web-Viewer" "%~dp0open-nexarm-viewer.cmd"
for /L %%i in (1,1,30) do (
  ping -n 2 127.0.0.1 >nul
  call :is_listening && goto :open_browser
)
echo Kamera-Viewer ist nach 30 s nicht erreichbar.
exit /b 1

:open_browser
start "" "http://localhost:9090/?url=rerun%%2Bhttp://localhost:9876/proxy"
exit /b 0

:is_listening
powershell -NoProfile -Command "if ((Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue) -and (Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue)) { exit 0 } else { exit 1 }"
exit /b %errorlevel%

:port_blocked
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
exit /b %errorlevel%
