@echo off
setlocal enabledelayedexpansion

rem Activate the Python virtual environment
call "%~dp0..\..\yolov5env\Scripts\activate.bat"

echo ===============================================
echo YOLO5 Project Management Integration System
echo ===============================================
echo.

:menu
echo Select application to analyze:
echo 1 - Microsoft Project
echo 2 - Primavera P6 Professional
echo 3 - Exit
echo.

set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" goto msproject
if "%choice%"=="2" goto primavera
if "%choice%"=="3" goto end

echo Invalid choice! Please try again.
echo.
goto menu

:msproject
echo.
echo === Microsoft Project Detection Options ===
echo.
echo 1 - Start Microsoft Project
echo 2 - Detect Project Elements
echo 3 - Generate Project Report
echo 4 - Return to Main Menu
echo.

set /p ms_choice=Enter your choice (1-4): 

if "%ms_choice%"=="1" goto start_msproject
if "%ms_choice%"=="2" goto detect_msproject
if "%ms_choice%"=="3" goto report_msproject
if "%ms_choice%"=="4" goto menu

echo Invalid choice! Please try again.
echo.
goto msproject

:start_msproject
echo.
echo Starting Microsoft Project...
echo.

python -c "from integrations.msproject.msproject_detector import MSProjectDetector; detector = MSProjectDetector(); detector.start_application('msproject')"

echo.
echo Press any key to return to Microsoft Project menu...
pause >nul
goto msproject

:detect_msproject
echo.
echo Detecting Microsoft Project elements...
echo.

python "%~dp0msproject\msproject_detector.py"

echo.
echo Press any key to return to Microsoft Project menu...
pause >nul
goto msproject

:report_msproject
echo.
echo Generating Microsoft Project report...
echo.

python -c "from integrations.msproject.msproject_detector import MSProjectDetector; detector = MSProjectDetector(); print(detector.generate_report('msproject'))"

echo.
echo Press any key to return to Microsoft Project menu...
pause >nul
goto msproject

:primavera
echo.
echo === Primavera P6 Detection Options ===
echo.
echo 1 - Start Primavera P6
echo 2 - Detect Project Elements
echo 3 - Generate Project Report
echo 4 - Return to Main Menu
echo.

set /p p6_choice=Enter your choice (1-4): 

if "%p6_choice%"=="1" goto start_primavera
if "%p6_choice%"=="2" goto detect_primavera
if "%p6_choice%"=="3" goto report_primavera
if "%p6_choice%"=="4" goto menu

echo Invalid choice! Please try again.
echo.
goto primavera

:start_primavera
echo.
echo Starting Primavera P6 Professional...
echo.

python -c "from integrations.primavera.primavera_detector import PrimaveraDetector; detector = PrimaveraDetector(); detector.start_application('primavera')"

echo.
echo Press any key to return to Primavera P6 menu...
pause >nul
goto primavera

:detect_primavera
echo.
echo Detecting Primavera P6 elements...
echo.

python "%~dp0primavera\primavera_detector.py"

echo.
echo Press any key to return to Primavera P6 menu...
pause >nul
goto primavera

:report_primavera
echo.
echo Generating Primavera P6 report...
echo.

python -c "from integrations.primavera.primavera_detector import PrimaveraDetector; detector = PrimaveraDetector(); print(detector.generate_report('primavera'))"

echo.
echo Press any key to return to Primavera P6 menu...
pause >nul
goto primavera

:end
echo.
echo Thank you for using the YOLO5 Project Management Integration System!
echo Deactivating virtual environment...
call deactivate
echo.
endlocal
