@echo off
cd /d C:\BA\logs
python export_npz.py C:\BA\runs_stage3\char_ocr_model_eval.pt C:\BA\logs\char_model_debug.npz > C:\BA\logs\export.txt 2>&1
