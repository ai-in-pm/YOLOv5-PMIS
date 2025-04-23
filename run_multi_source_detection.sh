#!/bin/bash
# ===============================================
# Multi-Source Project Management Detection Tool
# ===============================================

# Set the Python executable
PYTHON_EXE=python3

# Check if Python is available
if ! command -v $PYTHON_EXE &> /dev/null; then
    echo "Python not found. Please make sure Python is installed."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Multi-Source Project Management Detection Tool"
echo "==============================================="
echo

# Default values
APP="msproject"
SOURCE=""
INPUT=""
CAMERA=0
MODEL=""
CONF=0.25
IOU=0.45
TESSERACT=""
NO_OCR=""
CONTINUOUS=""
INTERVAL=1.0
MAX_FRAMES=10
MAX_TIME=30
REPORT=""
DEBUG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --app)
            APP="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
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
        --continuous)
            CONTINUOUS="--continuous"
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --max-time)
            MAX_TIME="$2"
            shift 2
            ;;
        --report)
            REPORT="--report"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$SOURCE" ]; then
    echo "Error: Source type is required. Use --source option."
    show_help=true
fi

if [ "$SOURCE" = "image" ] && [ -z "$INPUT" ]; then
    echo "Error: Input file path is required for image source. Use --input option."
    show_help=true
fi

if [ "$SOURCE" = "video" ] && [ -z "$INPUT" ]; then
    echo "Error: Input file path is required for video source. Use --input option."
    show_help=true
fi

if [ "$SOURCE" = "stream" ] && [ -z "$INPUT" ]; then
    echo "Error: Input URL is required for stream source. Use --input option."
    show_help=true
fi

# Show help if needed
if [ "$show_help" = true ]; then
    echo
    echo "Usage: $(basename $0) --source SOURCE [options]"
    echo
    echo "Sources:"
    echo "  --source webcam    Use webcam as input source"
    echo "  --source image     Use image file as input source"
    echo "  --source video     Use video file as input source"
    echo "  --source screen    Use screen capture as input source"
    echo "  --source stream    Use video stream as input source"
    echo
    echo "Options:"
    echo "  --app APP          Application type (msproject or primavera, default: msproject)"
    echo "  --input PATH       Input file path or URL (required for image, video, stream)"
    echo "  --camera ID        Camera device ID (default: 0)"
    echo "  --model PATH       YOLOv5 model path"
    echo "  --conf CONF        Detection confidence threshold (default: 0.25)"
    echo "  --iou IOU          IoU threshold for NMS (default: 0.45)"
    echo "  --tesseract PATH   Path to Tesseract OCR executable"
    echo "  --no-ocr           Disable OCR text extraction"
    echo "  --continuous       Run continuous detection"
    echo "  --interval SEC     Time interval between detections (default: 1.0)"
    echo "  --max-frames N     Maximum number of frames to process (default: 10)"
    echo "  --max-time SEC     Maximum time to run detection (default: 30)"
    echo "  --report           Generate and display project report"
    echo "  --debug            Enable debug logging"
    echo
    echo "Examples:"
    echo "  $(basename $0) --source webcam --app msproject --report"
    echo "  $(basename $0) --source image --input sample_images/msproject_screenshot.jpg --app msproject"
    echo "  $(basename $0) --source image --input sample_images/primavera_screenshot.jpg --app primavera"
    echo "  $(basename $0) --source screen --continuous --report"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Display parameters
echo "Running detection with the following parameters:"
echo "  Application: $APP"
echo "  Source: $SOURCE"
if [ ! -z "$INPUT" ]; then echo "  Input: $INPUT"; fi
if [ "$SOURCE" = "webcam" ]; then echo "  Camera ID: $CAMERA"; fi
if [ ! -z "$MODEL" ]; then echo "  Model: $MODEL"; fi
echo "  Confidence threshold: $CONF"
echo "  IoU threshold: $IOU"
if [ ! -z "$TESSERACT" ]; then echo "  Tesseract path: $TESSERACT"; fi
if [ "$NO_OCR" = "--no-ocr" ]; then echo "  OCR: Disabled"; fi
if [ "$CONTINUOUS" = "--continuous" ]; then
    echo "  Mode: Continuous"
    echo "  Interval: $INTERVAL seconds"
fi
if [ "$SOURCE" = "video" ]; then echo "  Max frames: $MAX_FRAMES"; fi
if [ "$SOURCE" = "webcam" ]; then echo "  Max time: $MAX_TIME seconds"; fi
if [ "$SOURCE" = "stream" ]; then echo "  Max time: $MAX_TIME seconds"; fi
if [ "$REPORT" = "--report" ]; then echo "  Report: Enabled"; fi
if [ "$DEBUG" = "--debug" ]; then echo "  Debug: Enabled"; fi
echo

# Build command
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CMD="$PYTHON_EXE $SCRIPT_DIR/multi_source_detector.py --app $APP --source $SOURCE"

if [ ! -z "$INPUT" ]; then CMD="$CMD --input \"$INPUT\""; fi
if [ "$SOURCE" = "webcam" ]; then CMD="$CMD --camera $CAMERA"; fi
if [ ! -z "$MODEL" ]; then CMD="$CMD --model \"$MODEL\""; fi
CMD="$CMD --conf $CONF --iou $IOU"
if [ ! -z "$TESSERACT" ]; then CMD="$CMD --tesseract \"$TESSERACT\""; fi
if [ "$NO_OCR" = "--no-ocr" ]; then CMD="$CMD $NO_OCR"; fi
if [ "$CONTINUOUS" = "--continuous" ]; then CMD="$CMD $CONTINUOUS --interval $INTERVAL"; fi
if [ "$SOURCE" = "video" ]; then CMD="$CMD --max-frames $MAX_FRAMES"; fi
if [ "$SOURCE" = "webcam" ]; then CMD="$CMD --max-time $MAX_TIME"; fi
if [ "$SOURCE" = "stream" ]; then CMD="$CMD --max-time $MAX_TIME"; fi
if [ "$REPORT" = "--report" ]; then CMD="$CMD $REPORT"; fi
if [ "$DEBUG" = "--debug" ]; then CMD="$CMD $DEBUG"; fi

echo "Running command: $CMD"
echo

# Check if sample images exist
if [ ! -f "$SCRIPT_DIR/sample_images/msproject_screenshot.jpg" ]; then
    echo "Sample images not found. Creating them..."
    $PYTHON_EXE "$SCRIPT_DIR/create_sample_images.py"
    echo
fi

# Run the command
eval $CMD

echo
echo "Detection completed with exit code $?."
echo "==============================================="

read -p "Press Enter to exit..."
