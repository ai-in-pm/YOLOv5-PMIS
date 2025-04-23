#!/usr/bin/env python
"""
Microsoft Project Detection Tool

This script provides a command-line interface for detecting elements in Microsoft Project
using YOLOv5 object detection.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Import the MSProjectDetector
from msproject.msproject_detector import MSProjectDetector

# Check if OCR is available
try:
    from ocr_utils import MSProjectOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def setup_logging(debug=False):
    """Configure logging"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = SCRIPT_DIR / "msproject_detection.log"

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logger = logging.getLogger('msproject_detection')

    # Log OCR availability
    if HAS_OCR:
        logger.info("OCR support is available")
    else:
        logger.warning("OCR support is not available. Text extraction will be disabled.")

    return logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Microsoft Project Detection Tool")
    parser.add_argument('--model', type=str, default="msproject_model.pt",
                        help='Path to the YOLOv5 model file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--start', action='store_true',
                        help='Start Microsoft Project before detection')
    parser.add_argument('--report', action='store_true',
                        help='Generate and display project report')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save detection results (JSON format)')
    parser.add_argument('--tesseract', type=str, default=None,
                        help='Path to Tesseract OCR executable')
    parser.add_argument('--no-ocr', action='store_true',
                        help='Disable OCR text extraction')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()

def find_tesseract() -> Optional[str]:
    """Find Tesseract OCR executable"""
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'/usr/bin/tesseract',
        r'/usr/local/bin/tesseract'
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.debug)
    logger.info("Starting Microsoft Project Detection Tool")

    # Find Tesseract if OCR is enabled and path not provided
    tesseract_path = args.tesseract
    if HAS_OCR and not args.no_ocr and not tesseract_path:
        tesseract_path = find_tesseract()
        if tesseract_path:
            logger.info(f"Found Tesseract at: {tesseract_path}")
        else:
            logger.warning("Tesseract not found. OCR may not work properly.")

    try:
        # Create detector
        logger.info(f"Initializing detector with model: {args.model}")
        detector = MSProjectDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=tesseract_path
        )

        # Start MS Project if requested
        if args.start:
            logger.info("Starting Microsoft Project...")
            success = detector.start_application("msproject")
            if not success:
                logger.error("Failed to start Microsoft Project")
                return 1

            # Wait for application to start
            import time
            logger.info("Waiting for application to start...")
            time.sleep(5)

        # Detect project elements
        logger.info("Detecting project elements...")
        results = detector.detect_project_elements(with_ocr=not args.no_ocr)

        if results:
            detection_count = results.get('detection_count', 0)
            logger.info(f"Detected {detection_count} elements")

            # Generate project report if requested
            if args.report:
                logger.info("Generating project report...")
                report = detector.generate_project_report()
                print("\n" + report)

            # Save results to file if requested
            if args.output:
                output_path = args.output
                if not output_path.endswith('.json'):
                    output_path += '.json'

                # Convert numpy arrays to lists for JSON serialization
                import json
                import numpy as np

                def convert_for_json(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(i) for i in obj]
                    else:
                        return obj

                # Save results to JSON file
                with open(output_path, 'w') as f:
                    json.dump(convert_for_json(results), f, indent=2)

                logger.info(f"Results saved to {output_path}")
        else:
            logger.error("Failed to detect project elements")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
