#!/usr/bin/env python
"""
Real-time Direct Detection in Microsoft Project

This script provides real-time detection directly in Microsoft Project,
highlighting detected elements in the interface.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
import argparse
import win32gui
import win32con
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'msproject_direct_detection.log'))
    ]
)
logger = logging.getLogger('msproject_direct_detection')

# Import detector
try:
    from msproject.msproject_realtime import MSProjectRealTimeDetector
    HAS_DETECTOR = True
    logger.info("Microsoft Project real-time detector imported successfully")
except ImportError:
    HAS_DETECTOR = False
    logger.error("Microsoft Project real-time detector not available")

def close_detection_windows():
    """Close any existing detection windows to prevent cascading"""
    try:
        # Find and close any existing detection windows
        def callback(hwnd, windows):
            if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Microsoft Project Detection" in window_text:
                    logger.info(f"Closing existing detection window: {window_text}")
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True

        win32gui.EnumWindows(callback, None)
    except Exception as e:
        logger.warning(f"Error closing detection windows: {e}")

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Real-time Direct Detection in Microsoft Project"
    )
    parser.add_argument('--model', type=str, default=None,
                        help='YOLOv5 model path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--tesseract', type=str, default=None,
                        help='Path to Tesseract OCR executable')
    parser.add_argument('--no-ocr', action='store_true',
                        help='Disable OCR text extraction')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Time interval between detections (seconds)')
    parser.add_argument('--report', action='store_true',
                        help='Generate and display project report periodically')
    parser.add_argument('--report-interval', type=int, default=10,
                        help='Interval for generating reports (seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--start', action='store_true',
                        help='Start Microsoft Project if not already running')
    parser.add_argument('--clean', action='store_true',
                        help='Close any existing detection windows before starting')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if not HAS_DETECTOR:
        logger.error("Microsoft Project real-time detector not available")
        return 1

    try:
        print("\n" + "="*50)
        print("Real-time Direct Detection in Microsoft Project")
        print("="*50 + "\n")

        print("This tool will detect elements directly in Microsoft Project")
        print("and display them in real-time with highlighting.\n")

        print("Instructions:")
        print("1. Make sure Microsoft Project is open")
        print("2. Position the Microsoft Project window so it's visible")
        print("3. The detector will automatically find the window")
        print("4. Detected elements will be highlighted in a separate window")
        print("5. Press 'q' in the detection window or Ctrl+C here to stop\n")

        # Close any existing detection windows if requested
        if args.clean:
            logger.info("Closing any existing detection windows...")
            close_detection_windows()
            time.sleep(1)  # Give time for windows to close

        # Create detector
        detector = MSProjectRealTimeDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=args.tesseract,
            use_ocr=not args.no_ocr,
            interval=args.interval,
            highlight_elements=True
        )

        # Start Microsoft Project if requested
        if args.start and detector.detector:
            logger.info("Starting Microsoft Project...")
            detector.detector.start_application("msproject")
            # Wait for Microsoft Project to start
            time.sleep(5)

        # Define callback function
        last_report_time = time.time()

        def detection_callback(results):
            nonlocal last_report_time
            detection_count = results.get('detection_count', 0)

            # Print detection summary
            if detection_count > 0:
                print(f"\rDetected {detection_count} elements in Microsoft Project", end="")
                sys.stdout.flush()

            # Generate report if requested and interval has passed
            if args.report and (time.time() - last_report_time) >= args.report_interval:
                report = detector.generate_report()
                if report:
                    print("\n\n" + report)
                last_report_time = time.time()

        # Start real-time detection
        print("Starting real-time detection...\n")
        success = detector.start_realtime_detection(detection_callback)

        if not success:
            logger.error("Failed to start real-time detection")
            return 1

        # Wait for user to stop
        try:
            print("Detection running. Press Ctrl+C to stop...")
            while True:
                # Check if any detection windows are still open
                windows_open = False

                def check_windows(hwnd, _):
                    nonlocal windows_open
                    if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
                        if "Microsoft Project Detection" in win32gui.GetWindowText(hwnd):
                            windows_open = True
                    return True

                win32gui.EnumWindows(check_windows, None)

                # If no windows are open and detection is still running, recreate the window
                if not windows_open and detector.is_running():
                    # The window was closed but detection is still running
                    # This is normal - just wait for the next detection cycle
                    pass

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping real-time detection...")
            detector.stop_realtime_detection()
            # Close any remaining detection windows
            close_detection_windows()
            print("Detection stopped.")

        return 0

    except KeyboardInterrupt:
        print("\n\nDetection interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
