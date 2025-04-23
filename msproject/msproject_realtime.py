#!/usr/bin/env python
"""
Real-time Microsoft Project Detection Module

This module provides functionality to continuously monitor and detect elements
in an open Microsoft Project application in real-time.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
import threading
import pyautogui
import win32gui
import win32process
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(SCRIPT_DIR.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'msproject_realtime.log'))
    ]
)
logger = logging.getLogger('msproject_realtime')

# Import detector
try:
    from msproject.msproject_detector import MSProjectDetector
    HAS_DETECTOR = True
    logger.info("Microsoft Project detector imported successfully")
except ImportError:
    HAS_DETECTOR = False
    logger.warning("Microsoft Project detector not available")

class MSProjectRealTimeDetector:
    """Real-time detector for Microsoft Project"""

    def __init__(self, model_path: Optional[str] = None,
                 conf_thres: float = 0.25, iou_thres: float = 0.45,
                 tesseract_path: Optional[str] = None, use_ocr: bool = True,
                 interval: float = 1.0, highlight_elements: bool = True):
        """Initialize the real-time detector

        Args:
            model_path: Path to YOLOv5 model file
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            tesseract_path: Path to Tesseract OCR executable
            use_ocr: Whether to use OCR for text extraction
            interval: Time interval between detections (seconds)
            highlight_elements: Whether to highlight detected elements
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.tesseract_path = tesseract_path
        self.use_ocr = use_ocr
        self.interval = interval
        self.highlight_elements = highlight_elements

        # Set default model path if not provided
        if self.model_path is None:
            self.model_path = os.path.join(SCRIPT_DIR, "models", "msproject_model.pt")
            if not os.path.exists(self.model_path):
                self.model_path = "msproject_model.pt"
                logger.info(f"Using default model path: {self.model_path}")

        # Initialize detector
        if HAS_DETECTOR:
            self.detector = MSProjectDetector(
                model_path=self.model_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                tesseract_path=self.tesseract_path
            )
            logger.info("Microsoft Project detector initialized")
        else:
            self.detector = None
            logger.error("Microsoft Project detector not available")

        # For real-time processing
        self.stop_flag = False
        self.processing_thread = None

        # Results storage
        self.results_dir = os.path.join(SCRIPT_DIR.parent, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Last detection results
        self.last_results = None
        self.last_screenshot = None
        self.last_detection_time = None

        # Window handling
        self.msproject_hwnd = None
        self.msproject_rect = None

    def find_msproject_window(self) -> bool:
        """Find the Microsoft Project window

        Returns:
            True if the window was found, False otherwise
        """
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text and (" - Project" in window_text or "Microsoft Project" in window_text):
                    # Verify it's actually MS Project by checking the process
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process = psutil.Process(pid)
                        if "WINPROJ" in process.name().upper() or "PROJECT" in process.name().upper():
                            windows.append((hwnd, window_text))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            return True

        windows = []
        win32gui.EnumWindows(callback, windows)

        if not windows:
            logger.warning("No Microsoft Project window found")
            return False

        # Use the first window found
        self.msproject_hwnd = windows[0][0]
        self.msproject_rect = win32gui.GetWindowRect(self.msproject_hwnd)
        logger.info(f"Found Microsoft Project window: {windows[0][1]}")
        logger.info(f"Window position: {self.msproject_rect}")
        return True

    def capture_msproject_window(self) -> Optional[np.ndarray]:
        """Capture the Microsoft Project window

        Returns:
            Screenshot as numpy array or None if capture failed
        """
        try:
            # Find the window if not already found
            if self.msproject_hwnd is None:
                if not self.find_msproject_window():
                    return None

            # Check if the window is still valid
            try:
                if not win32gui.IsWindow(self.msproject_hwnd):
                    logger.warning("Microsoft Project window is no longer valid")
                    self.msproject_hwnd = None
                    return None
            except Exception:
                logger.warning("Error checking Microsoft Project window")
                self.msproject_hwnd = None
                return None

            # Get the window rectangle
            try:
                self.msproject_rect = win32gui.GetWindowRect(self.msproject_hwnd)
            except Exception:
                logger.warning("Error getting Microsoft Project window rectangle")
                self.msproject_hwnd = None
                return None

            # Capture the window
            x, y, right, bottom = self.msproject_rect
            width = right - x
            height = bottom - y

            if width <= 0 or height <= 0:
                logger.warning("Invalid window dimensions")
                return None

            # Temporarily hide our detection window if it exists
            detection_window = None
            try:
                detection_window = win32gui.FindWindow(None, "Microsoft Project Detection")
                if detection_window:
                    win32gui.ShowWindow(detection_window, 0)  # Hide window
            except Exception:
                pass  # Ignore errors if window not found

            # Capture the screen region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Show our detection window again if it was hidden
            try:
                if detection_window:
                    win32gui.ShowWindow(detection_window, 1)  # Show window
            except Exception:
                pass  # Ignore errors

            return screenshot

        except Exception as e:
            logger.error(f"Error capturing Microsoft Project window: {e}", exc_info=True)
            return None

    def start_realtime_detection(self, callback=None) -> bool:
        """Start real-time detection

        Args:
            callback: Function to call with detection results

        Returns:
            True if detection started successfully, False otherwise
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return False

        if self.processing_thread and self.processing_thread.is_alive():
            logger.error("Real-time detection already running")
            return False

        # Reset stop flag
        self.stop_flag = False

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._realtime_detection_thread,
            args=(callback,),
            daemon=True
        )
        self.processing_thread.start()

        return True

    def stop_realtime_detection(self) -> bool:
        """Stop real-time detection

        Returns:
            True if detection stopped successfully, False otherwise
        """
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.warning("No real-time detection running")
            return False

        # Set stop flag
        self.stop_flag = True

        # Wait for thread to finish
        self.processing_thread.join(timeout=5.0)

        return not self.processing_thread.is_alive()

    def is_running(self) -> bool:
        """Check if real-time detection is running

        Returns:
            True if detection is running, False otherwise
        """
        return self.processing_thread is not None and self.processing_thread.is_alive() and not self.stop_flag

    def _realtime_detection_thread(self, callback) -> None:
        """Thread function for real-time detection

        Args:
            callback: Function to call with detection results
        """
        try:
            logger.info("Starting real-time detection in Microsoft Project")

            # Main detection loop
            while not self.stop_flag:
                start_time = time.time()

                # Capture the Microsoft Project window
                screenshot = self.capture_msproject_window()
                if screenshot is None:
                    logger.warning("Failed to capture Microsoft Project window")
                    # Wait before trying again
                    time.sleep(self.interval)
                    continue

                # Store the screenshot
                self.last_screenshot = screenshot

                # Run detection
                logger.info("Running detection on Microsoft Project window")
                results = self._run_detection(screenshot)

                # Store the results
                if results:
                    self.last_results = results
                    self.last_detection_time = datetime.now()

                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_results(results, f"msproject_realtime_{timestamp}")

                    # Highlight detected elements if requested
                    if self.highlight_elements:
                        highlighted = self._highlight_elements(screenshot, results)
                        if highlighted is not None:
                            self._display_highlighted(highlighted)

                    # Call callback if provided
                    if callback:
                        callback(results)

                # Calculate time to wait
                elapsed = time.time() - start_time
                wait_time = max(0, self.interval - elapsed)

                # Wait for next detection
                if wait_time > 0 and not self.stop_flag:
                    time.sleep(wait_time)

            logger.info("Real-time detection stopped")

        except Exception as e:
            logger.error(f"Error in real-time detection: {e}", exc_info=True)

    def _run_detection(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run detection on an image

        Args:
            image: Input image as numpy array

        Returns:
            Detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        try:
            # Run detection
            results = self.detector.detect_objects("msproject", image)

            # Apply OCR if requested and available
            if results and 'detections' in results and self.use_ocr:
                if hasattr(self.detector, 'ocr') and self.detector.ocr:
                    if hasattr(self.detector.ocr, 'tesseract_available') and self.detector.ocr.tesseract_available:
                        logger.info("Applying OCR to detected elements...")
                        results['detections'] = self.detector.ocr.process_detections(image, results['detections'])
                    else:
                        logger.warning("Tesseract OCR not available. Cannot process detections.")
                else:
                    logger.warning("OCR processor not available. Skipping text extraction.")

            # Extract project data
            if results and 'detections' in results:
                project_data = self.detector._extract_project_data(results['detections'])
                results['project_data'] = project_data

            return results

        except Exception as e:
            logger.error(f"Error running detection: {e}", exc_info=True)
            return None

    def _highlight_elements(self, image: np.ndarray, results: Dict[str, Any]) -> Optional[np.ndarray]:
        """Highlight detected elements in the image

        Args:
            image: Input image as numpy array
            results: Detection results

        Returns:
            Image with highlighted elements or None if highlighting failed
        """
        try:
            # Create a copy of the image
            highlighted = image.copy()

            # Get detections
            detections = results.get('detections', [])

            # Define colors for different element types
            colors = {
                'task': (0, 255, 0),       # Green
                'milestone': (0, 0, 255),  # Red
                'summary': (255, 0, 0),    # Blue
                'critical': (0, 0, 255),   # Red
                'dependency': (255, 255, 0), # Yellow
                'resource': (255, 0, 255), # Magenta
                'constraint': (0, 255, 255), # Cyan
                'gantt_bar': (0, 255, 0),  # Green
                'task_name': (255, 165, 0), # Orange
                'duration': (128, 0, 128), # Purple
                'start_date': (0, 128, 255), # Light Blue
                'finish_date': (255, 0, 128), # Pink
                'header': (255, 128, 0),   # Orange
                'grid_cell': (200, 200, 200), # Light Gray
                'default': (255, 255, 255) # White
            }

            # Draw bounding boxes and labels
            for detection in detections:
                # Get detection data
                x1, y1, x2, y2 = detection['bbox']
                label = detection['class']
                confidence = detection['confidence']

                # Get color based on label
                color = colors.get(label.lower(), colors['default'])

                # Draw semi-transparent overlay for better visibility
                overlay = highlighted.copy()
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)  # Filled rectangle
                highlighted = cv2.addWeighted(overlay, 0.3, highlighted, 0.7, 0)  # Transparency effect

                # Draw bounding box
                cv2.rectangle(highlighted, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label with better visibility
                text = f"{label} ({confidence:.2f})"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size

                # Draw text background
                cv2.rectangle(highlighted,
                             (int(x1), int(y1) - text_h - 8),
                             (int(x1) + text_w + 5, int(y1)),
                             color, -1)

                # Draw text
                cv2.putText(highlighted, text, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw OCR text if available
                if 'text' in detection and detection['text']:
                    ocr_text = detection['text']
                    # Truncate long OCR text
                    if len(ocr_text) > 20:
                        ocr_text = ocr_text[:17] + "..."

                    # Draw text background
                    text_size, _ = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_w, text_h = text_size

                    cv2.rectangle(highlighted,
                                 (int(x1), int(y2) + 5),
                                 (int(x1) + text_w + 5, int(y2) + text_h + 8),
                                 (0, 0, 0), -1)

                    # Draw text
                    cv2.putText(highlighted, ocr_text, (int(x1) + 2, int(y2) + text_h + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add legend to the image
            self._add_legend(highlighted, colors)

            return highlighted

        except Exception as e:
            logger.error(f"Error highlighting elements: {e}", exc_info=True)
            return None

    def _add_legend(self, image: np.ndarray, colors: Dict[str, tuple]) -> None:
        """Add a legend to the image

        Args:
            image: Image to add legend to
            colors: Dictionary of colors for different element types
        """
        try:
            # Define legend position and size
            legend_x = 10
            legend_y = 10
            legend_w = 200
            legend_h = 30 * (len(colors) - 1)  # Exclude 'default' color
            padding = 5

            # Draw legend background
            cv2.rectangle(image,
                         (legend_x, legend_y),
                         (legend_x + legend_w, legend_y + legend_h),
                         (0, 0, 0), -1)
            cv2.rectangle(image,
                         (legend_x, legend_y),
                         (legend_x + legend_w, legend_y + legend_h),
                         (255, 255, 255), 1)

            # Draw legend title
            cv2.putText(image, "Detected Elements",
                        (legend_x + padding, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw legend items
            y_offset = 50
            for label, color in colors.items():
                if label != 'default':
                    # Draw color box
                    cv2.rectangle(image,
                                 (legend_x + padding, legend_y + y_offset - 15),
                                 (legend_x + padding + 20, legend_y + y_offset),
                                 color, -1)

                    # Draw label
                    cv2.putText(image, label.capitalize(),
                                (legend_x + padding + 30, legend_y + y_offset - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    y_offset += 25

        except Exception as e:
            logger.error(f"Error adding legend: {e}", exc_info=True)

    def _display_highlighted(self, image: np.ndarray) -> None:
        """Display the highlighted image

        Args:
            image: Image with highlighted elements
        """
        try:
            # Create a window name - use a unique window name to prevent cascading
            window_name = "Microsoft Project Detection"

            # Check if window already exists and destroy it to prevent cascading
            try:
                # Get window property to check if window exists
                existing = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                if existing >= 0:
                    # Window exists, destroy it first
                    cv2.destroyWindow(window_name)
            except:
                # Window doesn't exist yet, which is fine
                pass

            # Create a new window with specific position to prevent cascading
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Set window position to fixed location
            cv2.moveWindow(window_name, 50, 50)

            # Add a note to the image about how to close
            note = "Press 'q' to close this window"
            h, w = image.shape[:2]
            cv2.putText(image, note, (w - 300, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, note, (w - 300, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Display the image
            cv2.imshow(window_name, image)

            # Check for 'q' key press to close window
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)

        except Exception as e:
            logger.error(f"Error displaying highlighted image: {e}", exc_info=True)

    def _save_results(self, results: Dict[str, Any], prefix: str) -> None:
        """Save detection results

        Args:
            results: Detection results
            prefix: Prefix for output files
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save JSON results
            import json

            # Convert numpy arrays to lists for JSON serialization
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
            json_path = os.path.join(self.results_dir, f"{prefix}_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(convert_for_json(results), f, indent=2)

            logger.info(f"Results saved to {json_path}")

            # Save highlighted image if available
            if self.last_screenshot is not None and self.highlight_elements:
                highlighted = self._highlight_elements(self.last_screenshot, results)
                if highlighted is not None:
                    image_path = os.path.join(self.results_dir, f"{prefix}_{timestamp}.jpg")
                    cv2.imwrite(image_path, highlighted)
                    logger.info(f"Highlighted image saved to {image_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    def generate_report(self) -> Optional[str]:
        """Generate a report from the last detection results

        Returns:
            Report text or None if generation failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        if self.last_results is None:
            logger.warning("No detection results available")
            return None

        try:
            # Generate report
            if hasattr(self.detector, 'generate_project_report'):
                report = self.detector.generate_project_report()
                return report
            else:
                logger.error("Detector does not support report generation")
                return None

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            return None

def main():
    """Main function"""
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Real-time Microsoft Project Detection")
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
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Time interval between detections (seconds)')
    parser.add_argument('--no-highlight', action='store_true',
                        help='Disable highlighting of detected elements')
    parser.add_argument('--report', action='store_true',
                        help='Generate and display project report periodically')
    parser.add_argument('--report-interval', type=int, default=10,
                        help='Interval for generating reports (seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--start', action='store_true',
                        help='Start Microsoft Project if not already running')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    try:
        # Create detector
        detector = MSProjectRealTimeDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=args.tesseract,
            use_ocr=not args.no_ocr,
            interval=args.interval,
            highlight_elements=not args.no_highlight
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
            logger.info(f"Detection completed with {results.get('detection_count', 0)} elements")

            # Generate report if requested and interval has passed
            if args.report and (time.time() - last_report_time) >= args.report_interval:
                report = detector.generate_report()
                if report:
                    print("\n" + report)
                last_report_time = time.time()

        # Start real-time detection
        success = detector.start_realtime_detection(detection_callback)

        if not success:
            logger.error("Failed to start real-time detection")
            return 1

        # Wait for user to stop
        print("\nReal-time detection started. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping real-time detection...")
            detector.stop_realtime_detection()

        return 0

    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
