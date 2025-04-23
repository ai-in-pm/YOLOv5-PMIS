#!/bin/bash
# ===============================================
# Project Management Integration GUI Launcher
# ===============================================

# Set the Python executable
PYTHON_EXE=python3

# Check if Python is available
if ! command -v $PYTHON_EXE &> /dev/null; then
    echo "Python not found. Please make sure Python is installed."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Project Management Integration GUI"
echo "==============================================="
echo

# Run the GUI
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
$PYTHON_EXE "$SCRIPT_DIR/pm_gui.py"

echo
echo "GUI closed with exit code $?."
echo "==============================================="

read -p "Press Enter to exit..."
