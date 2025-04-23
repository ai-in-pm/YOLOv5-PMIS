@echo off
rem ===============================================
rem Real-time Direct Detection in Microsoft Project
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

echo Real-time Direct Detection in Microsoft Project
echo ===============================================
echo.

rem Check for help flag first
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="/?" goto :show_help

rem Parse command line arguments
set MODEL=
set CONF=0.25
set IOU=0.45
set TESSERACT=
set NO_OCR=
set INTERVAL=0.5
set REPORT=
set REPORT_INTERVAL=10
set DEBUG=
set START=

:parse_args
if "%~1"=="" goto :run_detection
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--conf" (
    set CONF=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--iou" (
    set IOU=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--tesseract" (
    set TESSERACT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--no-ocr" (
    set NO_OCR=--no-ocr
    shift
    goto :parse_args
)
if /i "%~1"=="--interval" (
    set INTERVAL=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--report" (
    set REPORT=--report
    shift
    goto :parse_args
)
if /i "%~1"=="--report-interval" (
    set REPORT_INTERVAL=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--debug" (
    set DEBUG=--debug
    shift
    goto :parse_args
)
if /i "%~1"=="--start" (
    set START=--start
    shift
    goto :parse_args
)
echo Unknown option: %~1
shift
goto :parse_args

:show_help
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --model MODEL          Path to the YOLOv5 model file
echo   --conf CONF            Detection confidence threshold (default: 0.25)
echo   --iou IOU              IoU threshold for NMS (default: 0.45)
echo   --tesseract PATH       Path to Tesseract OCR executable
echo   --no-ocr               Disable OCR text extraction
echo   --interval SEC         Time interval between detections (default: 0.5)
echo   --report               Generate and display project report periodically
echo   --report-interval SEC  Interval for generating reports (default: 10)
echo   --debug                Enable debug logging
echo   --start                Start Microsoft Project if not already running
echo.
echo Examples:
echo   %~nx0 --start --report
echo   %~nx0 --interval 0.2 --report --report-interval 20
echo   %~nx0 --model custom_model.pt --conf 0.35
echo.
pause
exit /b 0

:run_detection
echo Running real-time direct detection with the following parameters:
if not "%MODEL%"=="" echo   Model: %MODEL%
echo   Confidence threshold: %CONF%
echo   IoU threshold: %IOU%
if not "%TESSERACT%"=="" echo   Tesseract path: %TESSERACT%
if "%NO_OCR%"=="--no-ocr" echo   OCR: Disabled
echo   Detection interval: %INTERVAL% seconds
if "%REPORT%"=="--report" echo   Report: Enabled (interval: %REPORT_INTERVAL% seconds)
if "%DEBUG%"=="--debug" echo   Debug: Enabled
if "%START%"=="--start" echo   Auto-start: Enabled
echo.

rem Build command
set CMD=%PYTHON_EXE% "%~dp0run_msproject_direct_detection.py"

if not "%MODEL%"=="" set CMD=%CMD% --model "%MODEL%"
set CMD=%CMD% --conf %CONF% --iou %IOU%
if not "%TESSERACT%"=="" set CMD=%CMD% --tesseract "%TESSERACT%"
if "%NO_OCR%"=="--no-ocr" set CMD=%CMD% %NO_OCR%
set CMD=%CMD% --interval %INTERVAL%
if "%REPORT%"=="--report" set CMD=%CMD% %REPORT% --report-interval %REPORT_INTERVAL%
if "%DEBUG%"=="--debug" set CMD=%CMD% %DEBUG%
if "%START%"=="--start" set CMD=%CMD% %START%

echo Running command: %CMD%
echo.

%CMD%

echo.
echo Detection completed with exit code %ERRORLEVEL%.
echo ===============================================

pause
exit /b 0
