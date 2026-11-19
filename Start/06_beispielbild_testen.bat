@echo off
title Beispielbild durch die Pipeline (altes und neues Stufe-3-Modell)
cd /d C:\BA\Code
set BILD=%~1
if "%BILD%"=="" set BILD=C:\BA\Daten\kennzeichen_detection_dataset\images\test\roboflow_001121.jpg
set NEU=C:\BA\runs_stage3_kaggle_gpu\char_ocr_model.pt
set ALT=C:\BA\runs_stage3\char_ocr_model.pt
set LOG=C:\BA\logs\beispielbild.txt
echo Laeuft ... Fortschritt: %LOG%
echo ==== %date% %time% Bild: %BILD% > %LOG%
echo ==== 1) Pipeline mit NEUEM Stufe-3-Modell (Kaggle GPU, 60 Epochen) >> %LOG%
python -u analyze_video.py --source "%BILD%" --min-track-frames 1 --no-vehicles --ocr-model %NEU% >> %LOG% 2>&1
echo ==== 2) Pipeline mit ALTEM Stufe-3-Modell (CPU, 15 Epochen) >> %LOG%
python -u analyze_video.py --source "%BILD%" --min-track-frames 1 --no-vehicles --ocr-model %ALT% >> %LOG% 2>&1
echo ==== 3) Stufe 3 auf den 13 echten Kennzeichen: NEU >> %LOG%
python -u eval_stage3_plates.py --ocr-model %NEU% --out-dir C:\BA\runs_stage3_kaggle_gpu\eval_kennzeichen >> %LOG% 2>&1
echo ==== 4) Stufe 3 auf den 13 echten Kennzeichen: ALT >> %LOG%
python -u eval_stage3_plates.py --ocr-model %ALT% --out-dir C:\BA\runs_stage3\eval_kennzeichen >> %LOG% 2>&1
echo FERTIG (Exitcode %errorlevel%) >> %LOG%
echo FERTIG
