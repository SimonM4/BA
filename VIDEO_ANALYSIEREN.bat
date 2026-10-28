@echo off
title BA - Kennzeichenerkennung: Video analysieren
cd /d C:\BA\Code
echo ================================================================
echo  Kennzeichenerkennung (Stufe 1-3)
echo  - Video(s) per Drag ^& Drop auf diese Datei ziehen, ODER
echo  - Videos nach C:\BA\Videos\Eingang legen und doppelklicken.
echo  Ergebnisse: C:\BA\Ergebnisse\^<video^>_^<zeit^>\
echo ================================================================
echo.
if "%~1"=="" goto ordner

:naechstes
if "%~1"=="" goto ende
echo Analysiere: %~1
python analyze_video.py --source "%~1" --open-result
shift
goto naechstes

:ordner
if not exist C:\BA\Videos\Eingang mkdir C:\BA\Videos\Eingang
echo Kein Video uebergeben - analysiere alle Videos in C:\BA\Videos\Eingang
python analyze_video.py --source C:\BA\Videos\Eingang --open-result

:ende
echo.
pause
