@echo off
title Pipeline testen (Stufe 1-3, Simulation, MRTMD)
cd /d C:\BA\Code
echo Pipeline-Test laeuft ... Fortschritt: C:\BA\logs\pipeline_tests.txt
echo Ergebnisse: C:\BA\Tests\pipeline_test_^<datum_zeit^>\bericht.md
python -u eval_pipeline.py %* > C:\BA\logs\pipeline_tests.txt 2>&1
echo FERTIG (Exitcode %errorlevel%) >> C:\BA\logs\pipeline_tests.txt
echo FERTIG (Exitcode %errorlevel%)
