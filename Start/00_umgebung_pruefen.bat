@echo off
cd /d C:\BA
echo ==== %date% %time% > logs\umgebung.txt
where py >> logs\umgebung.txt 2>&1
where python >> logs\umgebung.txt 2>&1
where kaggle >> logs\umgebung.txt 2>&1
py -0p >> logs\umgebung.txt 2>&1
python -c "import sys,torch,torchvision,ultralytics,cv2; print(sys.executable, sys.version); print('torch',torch.__version__,'tv',torchvision.__version__,'ul',ultralytics.__version__,'cv2',cv2.__version__,'cuda',torch.cuda.is_available()); import os; print('cpus',os.cpu_count())" >> logs\umgebung.txt 2>&1
python -c "import lap" >> logs\umgebung.txt 2>&1 && echo lap ok >> logs\umgebung.txt
wmic cpu get name,NumberOfLogicalProcessors >> logs\umgebung.txt 2>&1
wmic computersystem get TotalPhysicalMemory >> logs\umgebung.txt 2>&1
echo FERTIG >> logs\umgebung.txt
