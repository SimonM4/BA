@echo off
title BA - Kennzeichenerkennung: Live-Stream (RTSP)
cd /d C:\BA\Code
echo RTSP-Adresse des Drohnen-Streams eingeben (z.B. rtsp://192.168.1.20:8554/live)
set /p URL=URL: 
if "%URL%"=="" goto ende
echo Live-Fenster schliessen mit Taste q. Ergebnisse: C:\BA\Ergebnisse\stream_^<zeit^>\
python analyze_video.py --source "%URL%" --show --stride 2 --open-result
:ende
pause
