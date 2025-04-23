@echo off
rem ===============================================
rem Project Management Integration GUI Launcher
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

echo Project Management Integration GUI
echo ===============================================
echo.

rem Run the GUI
%PYTHON_EXE% "%~dp0pm_gui.py"

echo.
echo GUI closed with exit code %ERRORLEVEL%.
echo ===============================================

pause
