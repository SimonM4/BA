@echo off
title Stufe 3 auf echten Kennzeichen-Ausschnitten bewerten
cd /d C:\BA\Code
copy /y C:\BA\runs_stage3\char_ocr_model.pt C:\BA\runs_stage3\char_ocr_model_eval.pt > nul
python -u eval_stage3_plates.py --ocr-model C:\BA\runs_stage3\char_ocr_model_eval.pt > C:\BA\logs\stage3_eval.txt 2>&1
echo FERTIG (Exitcode %errorlevel%) >> C:\BA\logs\stage3_eval.txt
