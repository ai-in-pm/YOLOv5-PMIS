#!/usr/bin/env python
"""
Multi-Source Detector for Project Management Integration

This module provides functionality to run YOLOv5 detection on various input sources
including webcams, images, videos, screen captures, and streams for Microsoft Project
and Primavera P6 detection.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'multi_source_detector.log'))
    ]
)
logger = logging.getLogger('multi_source_detector')

# Import detectors
try:
    from msproject.msproject_detector import MSProjectDetector
    HAS_MSPROJECT = True
    logger.info("Microsoft Project detector imported successfully")
except ImportError:
    HAS_MSPROJECT = False
    logger.warning("Microsoft Project detector not available")

try:
    from primavera.primavera_detector import PrimaveraDetector
    HAS_PRIMAVERA = True
    logger.info("Primavera P6 detector imported successfully")
except ImportError:
    HAS_PRIMAVERA = False
    logger.warning("Primavera P6 detector not available")

# Check if OCR is available
try:
    from ocr_utils import get_ocr_processor
    HAS_OCR = True
    logger.info("OCR utilities imported successfully")
except ImportError:
    HAS_OCR = False
    logger.warning("OCR utilities not available")

class MultiSourceDetector:
    """Detector for multiple input sources"""

    def __init__(self, app_type: str = "msproject", model_path: Optional[str] = None,
                 conf_thres: float = 0.25, iou_thres: float = 0.45,
                 tesseract_path: Optional[str] = None, use_ocr: bool = True):
        """Initialize the detector

        Args:
            app_type: Application type ('msproject' or 'primavera')
            model_path: Path to YOLOv5 model file
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            tesseract_path: Path to Tesseract OCR executable
            use_ocr: Whether to use OCR for text extraction
        """
        self.app_type = app_type.lower()
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.tesseract_path = tesseract_path
        self.use_ocr = use_ocr and HAS_OCR

        # Set default model paths based on app type
        if self.model_path is None:
            if self.app_type == "msproject":
                self.model_path = os.path.join(SCRIPT_DIR, "msproject", "models", "msproject_model.pt")
                if not os.path.exists(self.model_path):
                    self.model_path = "msproject_model.pt"
                    logger.info(f"Using default model path: {self.model_path}")
            elif self.app_type == "primavera":
                self.model_path = os.path.join(SCRIPT_DIR, "primavera", "models", "primavera_model.pt")
                if not os.path.exists(self.model_path):
                    self.model_path = "primavera_model.pt"
                    logger.info(f"Using default model path: {self.model_path}")

        # Initialize detector based on app type
        self._initialize_detector()

        # For video processing
        self.stop_flag = False
        self.processing_thread = None

        # Results storage
        self.results_dir = os.path.join(SCRIPT_DIR, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _initialize_detector(self):
        """Initialize the appropriate detector"""
        if self.app_type == "msproject":
            if not HAS_MSPROJECT:
                logger.error("Microsoft Project detector not available")
                self.detector = None
                return

            self.detector = MSProjectDetector(
                model_path=self.model_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                tesseract_path=self.tesseract_path
            )
            logger.info("Microsoft Project detector initialized")

        elif self.app_type == "primavera":
            if not HAS_PRIMAVERA:
                logger.error("Primavera P6 detector not available")
                self.detector = None
                return

            self.detector = PrimaveraDetector(
                model_path=self.model_path,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                tesseract_path=self.tesseract_path
            )
            logger.info("Primavera P6 detector initialized")

        else:
            logger.error(f"Unknown application type: {self.app_type}")
            self.detector = None

    def detect_from_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Run detection on an image file

        Args:
            image_path: Path to image file

        Returns:
            Detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        # Try to find the image file in multiple locations
        if not os.path.isabs(image_path) and not os.path.exists(image_path):
            # Check in sample_images directory
            sample_path = os.path.join(SCRIPT_DIR, "sample_images", image_path)
            if os.path.exists(sample_path):
                image_path = sample_path
                logger.info(f"Using sample image: {image_path}")
            else:
                # Check if we have a default sample for this app type
                default_sample = os.path.join(SCRIPT_DIR, "sample_images", f"{self.app_type}_screenshot.jpg")
                if os.path.exists(default_sample):
                    image_path = default_sample
                    logger.info(f"Using default sample image for {self.app_type}: {image_path}")

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            logger.info("You can create sample images by running create_sample_images.py")
            return None

        try:
            # Load image
            logger.info(f"Loading image from {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Run detection
            logger.info("Running detection on image")
            results = self._run_detection(image)

            # Save results
            if results:
                self._save_results(results, f"image_{Path(image_path).stem}")

            return results

        except Exception as e:
            logger.error(f"Error detecting from image: {e}", exc_info=True)
            return None

    def detect_from_webcam(self, camera_id: int = 0, max_time: int = 30) -> Optional[Dict[str, Any]]:
        """Run detection on webcam feed

        Args:
            camera_id: Camera device ID
            max_time: Maximum time to run detection (seconds)

        Returns:
            Detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        try:
            # Open webcam
            logger.info(f"Opening webcam with ID {camera_id}")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open webcam with ID {camera_id}")
                return None

            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Read frame
            logger.info("Capturing frame from webcam")
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from webcam")
                cap.release()
                return None

            # Run detection
            logger.info("Running detection on webcam frame")
            results = self._run_detection(frame)

            # Save results
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_results(results, f"webcam_{timestamp}")

            # Release webcam
            cap.release()

            return results

        except Exception as e:
            logger.error(f"Error detecting from webcam: {e}", exc_info=True)
            return None

    def detect_from_video(self, video_path: str, interval: float = 1.0,
                          max_frames: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Run detection on video file

        Args:
            video_path: Path to video file
            interval: Time interval between frames (seconds)
            max_frames: Maximum number of frames to process

        Returns:
            List of detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        try:
            # Open video
            logger.info(f"Opening video from {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")

            # Calculate frame interval
            frame_interval = int(fps * interval)
            if frame_interval < 1:
                frame_interval = 1

            # Process frames
            all_results = []
            frame_idx = 0
            frames_processed = 0

            while frames_processed < max_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection
                logger.info(f"Running detection on frame {frame_idx}")
                results = self._run_detection(frame)

                if results:
                    # Add frame info
                    results['frame_idx'] = frame_idx
                    results['frame_time'] = frame_idx / fps

                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_results(results, f"video_{Path(video_path).stem}_frame{frame_idx}")

                    all_results.append(results)

                # Move to next frame
                frame_idx += frame_interval
                frames_processed += 1

                # Check if we've reached the end of the video
                if frame_idx >= frame_count:
                    break

            # Release video
            cap.release()

            return all_results

        except Exception as e:
            logger.error(f"Error detecting from video: {e}", exc_info=True)
            return None

    def detect_from_screen(self) -> Optional[Dict[str, Any]]:
        """Run detection on screen capture

        Returns:
            Detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        try:
            # Capture screen
            logger.info("Capturing screen")
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Run detection
            logger.info("Running detection on screen capture")
            results = self._run_detection(screenshot)

            # Save results
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_results(results, f"screen_{timestamp}")

            return results

        except Exception as e:
            logger.error(f"Error detecting from screen: {e}", exc_info=True)
            return None

    def detect_from_stream(self, stream_url: str, max_time: int = 30) -> Optional[Dict[str, Any]]:
        """Run detection on video stream

        Args:
            stream_url: URL of the video stream
            max_time: Maximum time to run detection (seconds)

        Returns:
            Detection results or None if detection failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return None

        try:
            # Open stream
            logger.info(f"Opening stream from {stream_url}")
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open stream: {stream_url}")
                return None

            # Read frame
            logger.info("Capturing frame from stream")
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from stream")
                cap.release()
                return None

            # Run detection
            logger.info("Running detection on stream frame")
            results = self._run_detection(frame)

            # Save results
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_results(results, f"stream_{timestamp}")

            # Release stream
            cap.release()

            return results

        except Exception as e:
            logger.error(f"Error detecting from stream: {e}", exc_info=True)
            return None

    def start_continuous_detection(self, source_type: str, source: Any = None,
                                  interval: float = 1.0, callback=None) -> bool:
        """Start continuous detection from a source

        Args:
            source_type: Type of source ('webcam', 'video', 'screen', 'stream')
            source: Source identifier (camera ID, file path, or URL)
            interval: Time interval between detections (seconds)
            callback: Function to call with detection results

        Returns:
            True if detection started successfully, False otherwise
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return False

        if self.processing_thread and self.processing_thread.is_alive():
            logger.error("Detection already running")
            return False

        # Reset stop flag
        self.stop_flag = False

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._continuous_detection_thread,
            args=(source_type, source, interval, callback),
            daemon=True
        )
        self.processing_thread.start()

        return True

    def stop_continuous_detection(self) -> bool:
        """Stop continuous detection

        Returns:
            True if detection stopped successfully, False otherwise
        """
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.warning("No detection running")
            return False

        # Set stop flag
        self.stop_flag = True

        # Wait for thread to finish
        self.processing_thread.join(timeout=5.0)

        return not self.processing_thread.is_alive()

    def _continuous_detection_thread(self, source_type: str, source: Any,
                                    interval: float, callback) -> None:
        """Thread function for continuous detection

        Args:
            source_type: Type of source ('webcam', 'video', 'screen', 'stream')
            source: Source identifier (camera ID, file path, or URL)
            interval: Time interval between detections (seconds)
            callback: Function to call with detection results
        """
        try:
            # Initialize capture
            cap = None

            if source_type == 'webcam':
                logger.info(f"Starting continuous detection from webcam {source}")
                cap = cv2.VideoCapture(source if source is not None else 0)

            elif source_type == 'video':
                logger.info(f"Starting continuous detection from video {source}")
                if not os.path.exists(source):
                    logger.error(f"Video file not found: {source}")
                    return
                cap = cv2.VideoCapture(source)

            elif source_type == 'stream':
                logger.info(f"Starting continuous detection from stream {source}")
                cap = cv2.VideoCapture(source)

            # Check if capture initialized successfully
            if cap is not None and not cap.isOpened():
                logger.error(f"Failed to open {source_type}")
                return

            # Main detection loop
            while not self.stop_flag:
                start_time = time.time()

                # Get frame
                if source_type in ['webcam', 'video', 'stream']:
                    ret, frame = cap.read()
                    if not ret:
                        # For video, loop back to beginning
                        if source_type == 'video':
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            logger.error(f"Failed to capture frame from {source_type}")
                            break

                elif source_type == 'screen':
                    # Capture screen
                    import pyautogui
                    screenshot = pyautogui.screenshot()
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                else:
                    logger.error(f"Unknown source type: {source_type}")
                    break

                # Run detection
                results = self._run_detection(frame)

                # Save results
                if results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_results(results, f"{source_type}_{timestamp}")

                    # Call callback if provided
                    if callback:
                        callback(results)

                # Calculate time to wait
                elapsed = time.time() - start_time
                wait_time = max(0, interval - elapsed)

                # Wait for next detection
                if wait_time > 0 and not self.stop_flag:
                    time.sleep(wait_time)

            # Clean up
            if cap is not None:
                cap.release()

            logger.info(f"Continuous detection from {source_type} stopped")

        except Exception as e:
            logger.error(f"Error in continuous detection: {e}", exc_info=True)

            # Clean up
            if cap is not None:
                cap.release()

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
            # Run detection based on app type
            if self.app_type == "msproject":
                # Use the detect_objects method directly with the image
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

            elif self.app_type == "primavera":
                # Use the detect_objects method directly with the image
                results = self.detector.detect_objects("primavera", image)

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

            else:
                logger.error(f"Unknown application type: {self.app_type}")
                return None

            return results

        except Exception as e:
            logger.error(f"Error running detection: {e}", exc_info=True)
            return None

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
            with open(json_path, 'w') as f:
                json.dump(convert_for_json(results), f, indent=2)

            logger.info(f"Results saved to {json_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate a report from detection results

        Args:
            results: Detection results (if None, use the last results)

        Returns:
            Report text or None if generation failed
        """
        if self.detector is None:
            logger.error("Detector not initialized")
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
    parser = argparse.ArgumentParser(description="Multi-Source Project Management Detection")
    parser.add_argument('--app', type=str, choices=['msproject', 'primavera'], default='msproject',
                        help='Project management application to analyze')
    parser.add_argument('--source', type=str, choices=['webcam', 'image', 'video', 'screen', 'stream'], required=True,
                        help='Detection source')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file path or URL (for image, video, stream)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (for webcam)')
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
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous detection')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Time interval between detections (seconds)')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Maximum number of frames to process (for video)')
    parser.add_argument('--max-time', type=int, default=30,
                        help='Maximum time to run detection (seconds)')
    parser.add_argument('--report', action='store_true',
                        help='Generate and display project report')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    try:
        # Create detector
        detector = MultiSourceDetector(
            app_type=args.app,
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=args.tesseract,
            use_ocr=not args.no_ocr
        )

        # Run detection based on source
        results = None

        if args.continuous:
            # Define callback function
            def detection_callback(results):
                logger.info(f"Detection completed with {results.get('detection_count', 0)} elements")

                if args.report:
                    report = detector.generate_report(results)
                    if report:
                        print("\n" + report)

            # Start continuous detection
            if args.source == 'webcam':
                success = detector.start_continuous_detection('webcam', args.camera, args.interval, detection_callback)
            elif args.source == 'video':
                success = detector.start_continuous_detection('video', args.input, args.interval, detection_callback)
            elif args.source == 'screen':
                success = detector.start_continuous_detection('screen', None, args.interval, detection_callback)
            elif args.source == 'stream':
                success = detector.start_continuous_detection('stream', args.input, args.interval, detection_callback)
            elif args.source == 'image':
                logger.error("Continuous detection not supported for image source")
                return 1

            if not success:
                logger.error("Failed to start continuous detection")
                return 1

            # Wait for user to stop
            print("\nContinuous detection started. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping continuous detection...")
                detector.stop_continuous_detection()

        else:
            # Run single detection
            if args.source == 'webcam':
                results = detector.detect_from_webcam(args.camera, args.max_time)
            elif args.source == 'image':
                if not args.input:
                    logger.error("Input file path required for image source")
                    return 1
                results = detector.detect_from_image(args.input)
            elif args.source == 'video':
                if not args.input:
                    logger.error("Input file path required for video source")
                    return 1
                results = detector.detect_from_video(args.input, args.interval, args.max_frames)
            elif args.source == 'screen':
                results = detector.detect_from_screen()
            elif args.source == 'stream':
                if not args.input:
                    logger.error("Input URL required for stream source")
                    return 1
                results = detector.detect_from_stream(args.input, args.max_time)

            if not results:
                logger.error("Detection failed")
                return 1

            # Log results
            if isinstance(results, list):
                logger.info(f"Detection completed with {len(results)} frames")
                for i, frame_results in enumerate(results):
                    logger.info(f"Frame {i+1}: {frame_results.get('detection_count', 0)} elements")
            else:
                logger.info(f"Detection completed with {results.get('detection_count', 0)} elements")

            # Generate report if requested
            if args.report:
                report = detector.generate_report(results)
                if report:
                    print("\n" + report)

        return 0

    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
