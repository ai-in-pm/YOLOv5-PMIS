@echo off
setlocal enabledelayedexpansion

rem Activate the Python virtual environment
call "%~dp0..\yolov5env\Scripts\activate.bat"

echo ===============================================
echo YOLOv5 Enhanced Detection System
echo ===============================================
echo.

:menu
echo Select operation mode:
echo 1 - Standard Detection (Original YOLOv5)
echo 2 - Enhanced Detection (with Database)
echo 3 - Analyze Previous Detections
echo 4 - Exit
echo.

set /p choice=Enter your choice (1-4): 

if "%choice%"=="1" goto standard_detect
if "%choice%"=="2" goto enhanced_detect
if "%choice%"=="3" goto analyze
if "%choice%"=="4" goto end

echo Invalid choice! Please try again.
echo.
goto menu

:standard_detect
echo.
echo === Standard YOLOv5 Detection ===
echo.
set /p source=Enter source (path to image/video/folder or webcam number): 
set /p model=Enter model (yolov5s.pt, yolov5m.pt, etc.) [yolov5s.pt]: 
if "!model!"=="" set model=yolov5s.pt
set /p conf=Enter confidence threshold (0.1-0.9) [0.25]: 
if "!conf!"=="" set conf=0.25

set cmd_args=--mode detect --source "!source!" --weights !model! --conf !conf!

echo.
set /p save_txt=Save detection results as text files? (y/n) [n]: 
if /i "!save_txt!"=="y" set cmd_args=!cmd_args! --save-txt

echo.
set /p view=View detection results in real-time? (y/n) [n]: 
if /i "!view!"=="y" set cmd_args=!cmd_args! --view

echo.
python "%~dp0run_yolov5.py" !cmd_args!

echo.
echo Press any key to return to the menu...
pause >nul
goto menu

:enhanced_detect
echo.
echo === Enhanced YOLOv5 Detection ===
echo.
set /p source=Enter source (path to image/video/folder or webcam number): 
set /p model=Enter model (yolov5s.pt, yolov5m.pt, etc.) [yolov5s.pt]: 
if "!model!"=="" set model=yolov5s.pt
set /p conf=Enter confidence threshold (0.1-0.9) [0.25]: 
if "!conf!"=="" set conf=0.25

set cmd_args=--mode enhanced --source "!source!" --weights !model! --conf !conf!

echo.
set /p save_txt=Save detection results as text files? (y/n) [n]: 
if /i "!save_txt!"=="y" set cmd_args=!cmd_args! --save-txt

echo.
set /p save_db=Save detection results to database? (y/n) [y]: 
if not /i "!save_db!"=="n" set cmd_args=!cmd_args! --save-db

echo.
set /p view=View detection results in real-time? (y/n) [n]: 
if /i "!view!"=="y" set cmd_args=!cmd_args! --view

echo.
python "%~dp0run_yolov5.py" !cmd_args!

echo.
echo Press any key to return to the menu...
pause >nul
goto menu

:analyze
echo.
echo === Analyze Previous Detections ===
echo.
set /p model=Enter model to filter by (blank for all): 

set cmd_args=--mode analyze
if not "!model!"=="" set cmd_args=!cmd_args! --weights !model!

echo.
python "%~dp0run_yolov5.py" !cmd_args!

echo.
echo Press any key to return to the menu...
pause >nul
goto menu

:end
echo.
echo Thank you for using YOLOv5 Enhanced Detection System!
echo Deactivating virtual environment...
call deactivate
echo.
endlocal
