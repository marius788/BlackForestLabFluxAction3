@echo off
call "%~dp0_env.cmd" || (pause & exit /b 1)
title NexArm F3A - Rerun Web-Viewer
echo Rerun-Server fuer die Live-Anzeige laeuft, der Viewer oeffnet sich im Browser.
echo Falls nicht: http://localhost:%F3A_VIEWER_WEB_PORT%/?url=rerun%%2Bhttp://localhost:%F3A_VIEWER_PORT%/proxy
echo Dieses Fenster offen lassen, solange teleoperiert oder aufgenommen wird. Beenden mit Strg+C.
rem --serve-web instead of --web-viewer: rerun must not spawn the browser itself, otherwise the browser
rem inherits rerun's listening socket and keeps the port blocked after this window is closed.
rerun --serve-web --port %F3A_VIEWER_PORT% --web-viewer-port %F3A_VIEWER_WEB_PORT% --server-memory-limit 1GB
