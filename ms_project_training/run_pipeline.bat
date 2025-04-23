@echo off
echo ==========================================
echo Microsoft Project Detection Training Pipeline
echo ==========================================
echo.

REM Activate the YOLOv5 environment
call ..\..\yolov5env\Scripts\activate.bat

REM Run the pipeline script
python run_pipeline.py %*

echo.
echo Pipeline execution complete.
echo ==========================================
pause
