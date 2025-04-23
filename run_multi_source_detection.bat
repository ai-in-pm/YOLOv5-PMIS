@echo off
rem ===============================================
rem Multi-Source Project Management Detection Tool
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

echo Multi-Source Project Management Detection Tool
echo ===============================================
echo.

rem Parse command line arguments
set APP=msproject
set SOURCE=
set INPUT=
set CAMERA=0
set MODEL=
set CONF=0.25
set IOU=0.45
set TESSERACT=
set NO_OCR=
set CONTINUOUS=
set INTERVAL=1.0
set MAX_FRAMES=10
set MAX_TIME=30
set REPORT=
set DEBUG=

:parse_args
if "%~1"=="" goto check_args
if /i "%~1"=="--app" set APP=%~2 & shift
if /i "%~1"=="--source" set SOURCE=%~2 & shift
if /i "%~1"=="--input" set INPUT=%~2 & shift
if /i "%~1"=="--camera" set CAMERA=%~2 & shift
if /i "%~1"=="--model" set MODEL=%~2 & shift
if /i "%~1"=="--conf" set CONF=%~2 & shift
if /i "%~1"=="--iou" set IOU=%~2 & shift
if /i "%~1"=="--tesseract" set TESSERACT=%~2 & shift
if /i "%~1"=="--no-ocr" set NO_OCR=--no-ocr
if /i "%~1"=="--continuous" set CONTINUOUS=--continuous
if /i "%~1"=="--interval" set INTERVAL=%~2 & shift
if /i "%~1"=="--max-frames" set MAX_FRAMES=%~2 & shift
if /i "%~1"=="--max-time" set MAX_TIME=%~2 & shift
if /i "%~1"=="--report" set REPORT=--report
if /i "%~1"=="--debug" set DEBUG=--debug
shift
goto parse_args

:check_args
if "%SOURCE%"=="" (
    echo Error: Source type is required. Use --source option.
    goto show_help
)

if "%SOURCE%"=="image" (
    if "%INPUT%"=="" (
        echo Error: Input file path is required for image source. Use --input option.
        goto show_help
    )
)

if "%SOURCE%"=="video" (
    if "%INPUT%"=="" (
        echo Error: Input file path is required for video source. Use --input option.
        goto show_help
    )
)

if "%SOURCE%"=="stream" (
    if "%INPUT%"=="" (
        echo Error: Input URL is required for stream source. Use --input option.
        goto show_help
    )
)

goto run

:show_help
echo.
echo Usage: %~nx0 --source SOURCE [options]
echo.
echo Sources:
echo   --source webcam    Use webcam as input source
echo   --source image     Use image file as input source
echo   --source video     Use video file as input source
echo   --source screen    Use screen capture as input source
echo   --source stream    Use video stream as input source
echo.
echo Options:
echo   --app APP          Application type (msproject or primavera, default: msproject)
echo   --input PATH       Input file path or URL (required for image, video, stream)
echo   --camera ID        Camera device ID (default: 0)
echo   --model PATH       YOLOv5 model path
echo   --conf CONF        Detection confidence threshold (default: 0.25)
echo   --iou IOU          IoU threshold for NMS (default: 0.45)
echo   --tesseract PATH   Path to Tesseract OCR executable
echo   --no-ocr           Disable OCR text extraction
echo   --continuous       Run continuous detection
echo   --interval SEC     Time interval between detections (default: 1.0)
echo   --max-frames N     Maximum number of frames to process (default: 10)
echo   --max-time SEC     Maximum time to run detection (default: 30)
echo   --report           Generate and display project report
echo   --debug            Enable debug logging
echo.
echo Examples:
echo   %~nx0 --source webcam --app msproject --report
echo   %~nx0 --source image --input sample_images\msproject_screenshot.jpg --app msproject
echo   %~nx0 --source image --input sample_images\primavera_screenshot.jpg --app primavera
echo   %~nx0 --source screen --continuous --report
echo.
pause
exit /b 1

:run
echo Running detection with the following parameters:
echo   Application: %APP%
echo   Source: %SOURCE%
if not "%INPUT%"=="" echo   Input: %INPUT%
if "%SOURCE%"=="webcam" echo   Camera ID: %CAMERA%
if not "%MODEL%"=="" echo   Model: %MODEL%
echo   Confidence threshold: %CONF%
echo   IoU threshold: %IOU%
if not "%TESSERACT%"=="" echo   Tesseract path: %TESSERACT%
if "%NO_OCR%"=="--no-ocr" echo   OCR: Disabled
if "%CONTINUOUS%"=="--continuous" (
    echo   Mode: Continuous
    echo   Interval: %INTERVAL% seconds
)
if "%SOURCE%"=="video" echo   Max frames: %MAX_FRAMES%
if "%SOURCE%"=="webcam" echo   Max time: %MAX_TIME% seconds
if "%SOURCE%"=="stream" echo   Max time: %MAX_TIME% seconds
if "%REPORT%"=="--report" echo   Report: Enabled
if "%DEBUG%"=="--debug" echo   Debug: Enabled
echo.

rem Build command
set CMD=%PYTHON_EXE% "%~dp0multi_source_detector.py" --app %APP% --source %SOURCE%

if not "%INPUT%"=="" set CMD=%CMD% --input "%INPUT%"
if "%SOURCE%"=="webcam" set CMD=%CMD% --camera %CAMERA%
if not "%MODEL%"=="" set CMD=%CMD% --model "%MODEL%"
set CMD=%CMD% --conf %CONF% --iou %IOU%
if not "%TESSERACT%"=="" set CMD=%CMD% --tesseract "%TESSERACT%"
if "%NO_OCR%"=="--no-ocr" set CMD=%CMD% %NO_OCR%
if "%CONTINUOUS%"=="--continuous" set CMD=%CMD% %CONTINUOUS% --interval %INTERVAL%
if "%SOURCE%"=="video" set CMD=%CMD% --max-frames %MAX_FRAMES%
if "%SOURCE%"=="webcam" set CMD=%CMD% --max-time %MAX_TIME%
if "%SOURCE%"=="stream" set CMD=%CMD% --max-time %MAX_TIME%
if "%REPORT%"=="--report" set CMD=%CMD% %REPORT%
if "%DEBUG%"=="--debug" set CMD=%CMD% %DEBUG%

echo Running command: %CMD%

rem Check if sample images exist
if not exist "%~dp0sample_images\msproject_screenshot.jpg" (
    echo Sample images not found. Creating them...
    %PYTHON_EXE% "%~dp0create_sample_images.py"
    echo.
)
echo.

%CMD%

echo.
echo Detection completed with exit code %ERRORLEVEL%.
echo ===============================================

pause
