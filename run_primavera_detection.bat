@echo off
rem ===============================================
rem Primavera P6 Detection Tool
rem ===============================================

rem Set the Python executable
set PYTHON_EXE=python

rem Check if Python is available
%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

echo Primavera P6 Detection Tool
echo ===============================================
echo.

rem Parse command line arguments
set START=
set REPORT=
set OUTPUT=
set DEBUG=
set MODEL=primavera_model.pt
set CONF=0.25
set IOU=0.45
set TESSERACT=
set NO_OCR=

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--start" set START=--start
if /i "%~1"=="--report" set REPORT=--report
if /i "%~1"=="--debug" set DEBUG=--debug
if /i "%~1"=="--no-ocr" set NO_OCR=--no-ocr
if /i "%~1"=="--model" set MODEL=%~2 & shift
if /i "%~1"=="--conf" set CONF=%~2 & shift
if /i "%~1"=="--iou" set IOU=%~2 & shift
if /i "%~1"=="--output" set OUTPUT=--output "%~2" & shift
if /i "%~1"=="--tesseract" set TESSERACT=--tesseract "%~2" & shift
shift
goto parse_args

:run
echo Running Primavera P6 Detection...
echo.

python "%~dp0run_primavera_detection.py" %START% %REPORT% %DEBUG% %NO_OCR% --model "%MODEL%" --conf %CONF% --iou %IOU% %OUTPUT% %TESSERACT%

echo.
echo Detection completed with exit code %ERRORLEVEL%.
echo ===============================================

pause
