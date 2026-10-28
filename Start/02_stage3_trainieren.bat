@echo off
title Stufe 3 - Zeichenerkennung trainieren
cd /d C:\BA\Code
echo Training Stufe 3 laeuft ... Fortschritt: C:\BA\logs\stage3_training.txt
python -u train_stage3_char_ocr.py --data "C:\BA\Daten\archive (4).zip" "C:\BA\Daten\DeutschEinzelneZahlen.zip" --negatives-from C:\BA\Daten\kennzeichen_detection_dataset > C:\BA\logs\stage3_training.txt 2>&1
echo FERTIG (Exitcode %errorlevel%) >> C:\BA\logs\stage3_training.txt
