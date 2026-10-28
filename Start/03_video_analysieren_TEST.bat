@echo off
title Pipeline-Test mit Testvideo
cd /d C:\BA\Code
python -u analyze_video.py --source C:\BA\Videos\testvideo_kennzeichen.mp4 --stride 2 > C:\BA\logs\pipeline_test.txt 2>&1
echo FERTIG (Exitcode %errorlevel%) >> C:\BA\logs\pipeline_test.txt
