@echo off
cd /d C:\BA
if not exist kaggle_output\batraining mkdir kaggle_output\batraining
echo ==== %date% %time% > logs\kaggle_download.txt
kaggle kernels status simon1543/batraining >> logs\kaggle_download.txt 2>&1
kaggle kernels output simon1543/batraining -p C:\BA\kaggle_output\batraining >> logs\kaggle_download.txt 2>&1
echo FERTIG >> logs\kaggle_download.txt
