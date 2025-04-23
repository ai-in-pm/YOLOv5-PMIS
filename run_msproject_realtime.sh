#!/bin/bash
# ===============================================
# Real-time Microsoft Project Detection Tool
# ===============================================

# Set the Python executable
PYTHON_EXE=python3

# Check if Python is available
if ! command -v $PYTHON_EXE &> /dev/null; then
    echo "Python not found. Please make sure Python is installed."
    read -r -p "Press Enter to exit..."
    exit 1
fi

echo "Real-time Microsoft Project Detection Tool"
echo "==============================================="
echo

# Default values
MODEL=""
CONF=0.25
IOU=0.45
TESSERACT=""
NO_OCR=""
INTERVAL=1.0
NO_HIGHLIGHT=""
REPORT=""
REPORT_INTERVAL=10
DEBUG=""
START=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --iou)
            IOU="$2"
            shift 2
            ;;
        --tesseract)
            TESSERACT="$2"
            shift 2
            ;;
        --no-ocr)
            NO_OCR="--no-ocr"
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --no-highlight)
            NO_HIGHLIGHT="--no-highlight"
            shift
            ;;
        --report)
            REPORT="--report"
            shift
            ;;
        --report-interval)
            REPORT_INTERVAL="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --start)
            START="--start"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display parameters
echo "Running real-time detection with the following parameters:"
if [ -n "$MODEL" ]; then echo "  Model: $MODEL"; fi
echo "  Confidence threshold: $CONF"
echo "  IoU threshold: $IOU"
if [ -n "$TESSERACT" ]; then echo "  Tesseract path: $TESSERACT"; fi
if [ "$NO_OCR" = "--no-ocr" ]; then echo "  OCR: Disabled"; fi
echo "  Detection interval: $INTERVAL seconds"
if [ "$NO_HIGHLIGHT" = "--no-highlight" ]; then echo "  Highlighting: Disabled"; fi
if [ "$REPORT" = "--report" ]; then echo "  Report: Enabled (interval: $REPORT_INTERVAL seconds)"; fi
if [ "$DEBUG" = "--debug" ]; then echo "  Debug: Enabled"; fi
if [ "$START" = "--start" ]; then echo "  Auto-start: Enabled"; fi
echo

# Build command
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CMD="$PYTHON_EXE $SCRIPT_DIR/msproject/msproject_realtime.py"

if [ -n "$MODEL" ]; then CMD="$CMD --model \"$MODEL\""; fi
CMD="$CMD --conf $CONF --iou $IOU"
if [ -n "$TESSERACT" ]; then CMD="$CMD --tesseract \"$TESSERACT\""; fi
if [ "$NO_OCR" = "--no-ocr" ]; then CMD="$CMD $NO_OCR"; fi
CMD="$CMD --interval $INTERVAL"
if [ "$NO_HIGHLIGHT" = "--no-highlight" ]; then CMD="$CMD $NO_HIGHLIGHT"; fi
if [ "$REPORT" = "--report" ]; then CMD="$CMD $REPORT --report-interval $REPORT_INTERVAL"; fi
if [ "$DEBUG" = "--debug" ]; then CMD="$CMD $DEBUG"; fi
if [ "$START" = "--start" ]; then CMD="$CMD $START"; fi

echo "Running command: $CMD"
echo

# Run the command
eval $CMD

echo
echo "Detection completed with exit code $?."
echo "==============================================="

read -r -p "Press Enter to exit..."
