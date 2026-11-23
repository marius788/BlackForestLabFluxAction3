@echo off
rem Helper for _viewer.cmd: starts viewer.cmd in its own window unless the rerun server (gRPC port
rem F3A_VIEWER_PORT + web viewer port F3A_VIEWER_WEB_PORT, see _env.cmd) is already listening, then
rem waits until it accepts connections and opens the web viewer in the browser.
call :is_listening
if not errorlevel 1 (
  echo Kamera-Viewer laeuft bereits. Falls der Browser-Tab geschlossen ist:
  echo   http://localhost:%F3A_VIEWER_WEB_PORT%/?url=rerun%%2Bhttp://localhost:%F3A_VIEWER_PORT%/proxy
  exit /b 0
)
call :port_blocked
if not errorlevel 1 (
  echo Port %F3A_VIEWER_PORT% ist belegt, aber der Viewer auf Port %F3A_VIEWER_WEB_PORT% laeuft nicht.
  echo Meist haelt ein Browserfenster aus einem frueheren Viewer-Start den Port fest.
  echo Bitte den Browser komplett schliessen und neu starten.
  exit /b 1
)
echo Starte Kamera-Viewer ...
start "NexArm F3A - Rerun Web-Viewer" "%~dp0viewer.cmd"
for /L %%i in (1,1,30) do (
  ping -n 2 127.0.0.1 >nul
  call :is_listening && goto :open_browser
)
echo Kamera-Viewer ist nach 30 s nicht erreichbar.
exit /b 1

:open_browser
start "" "http://localhost:%F3A_VIEWER_WEB_PORT%/?url=rerun%%2Bhttp://localhost:%F3A_VIEWER_PORT%/proxy"
exit /b 0

:is_listening
powershell -NoProfile -Command "if ((Get-NetTCPConnection -LocalPort %F3A_VIEWER_PORT% -State Listen -ErrorAction SilentlyContinue) -and (Get-NetTCPConnection -LocalPort %F3A_VIEWER_WEB_PORT% -State Listen -ErrorAction SilentlyContinue)) { exit 0 } else { exit 1 }"
exit /b %errorlevel%

:port_blocked
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %F3A_VIEWER_PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
exit /b %errorlevel%
