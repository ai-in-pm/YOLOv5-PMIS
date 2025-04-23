@echo off
setlocal enabledelayedexpansion

rem Activate the Python virtual environment if it exists
if exist "%~dp0..\..\yolov5env\Scripts\activate.bat" (
    call "%~dp0..\..\yolov5env\Scripts\activate.bat"
)

echo ===============================================
echo Microsoft Project Detection Tool
echo ===============================================
echo.

rem Parse command line arguments
set START=
set REPORT=
set OUTPUT=
set DEBUG=
set MODEL=msproject_model.pt
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
echo Running Microsoft Project Detection...
echo.

python "%~dp0run_msproject_detection.py" %START% %REPORT% %DEBUG% %NO_OCR% --model "%MODEL%" --conf %CONF% --iou %IOU% %OUTPUT% %TESSERACT%

echo.
echo Detection completed with exit code %ERRORLEVEL%.
echo ===============================================

rem Deactivate the virtual environment if it was activated
if exist "%~dp0..\..\yolov5env\Scripts\activate.bat" (
    call deactivate
)

pause
endlocal
