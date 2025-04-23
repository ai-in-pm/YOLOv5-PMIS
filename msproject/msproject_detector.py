import os
import sys
import time
import json
import cv2
import numpy as np
import pyautogui
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger('msproject_detector')

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import the core integration module
from pm_integration import ProjectManagementDetector, MS_PROJECT_PATH, MS_PROJECT_PATHS

# Import OCR utilities
try:
    from ocr_utils import MSProjectOCR, get_ocr_processor
    HAS_OCR = True
except ImportError:
    logger.warning("OCR utilities not available. Text extraction will be disabled.")
    HAS_OCR = False

class MSProjectDetector(ProjectManagementDetector):
    """Specialized detector for Microsoft Project elements"""

    def __init__(self, model_path: str = "msproject_model.pt", conf_thres: float = 0.25,
                 iou_thres: float = 0.45, tesseract_path: Optional[str] = None):
        """Initialize the MSProjectDetector with model and detection parameters

        Args:
            model_path: Path to the YOLOv5 model file trained for MS Project detection
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for non-maximum suppression
            tesseract_path: Optional path to Tesseract OCR executable
        """
        # Try to find the specialized model in multiple locations
        if not os.path.isabs(model_path):
            possible_paths = [
                os.path.join(SCRIPT_DIR, model_path),
                os.path.join(SCRIPT_DIR, 'models', model_path),
                os.path.join(PARENT_DIR, 'models', model_path),
                os.path.join(PARENT_DIR.parent, 'models', model_path)
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found specialized model at: {model_path}")
                    break
            else:
                # If specialized model doesn't exist, fall back to standard YOLOv5s
                model_path = "yolov5s.pt"
                logger.warning(f"Specialized model not found, using {model_path} instead.")

        # Initialize the parent class
        super().__init__(model_path, conf_thres, iou_thres)

        # Define MS Project specific object classes
        self.object_classes = [
            'task', 'milestone', 'summary', 'critical_task', 'resource',
            'gantt_bar', 'dependency', 'constraint', 'deadline', 'baseline'
        ]

        # Initialize OCR processor
        self.tesseract_path = tesseract_path
        if HAS_OCR:
            self.ocr = MSProjectOCR(tesseract_path)
            logger.info("OCR processor initialized")
        else:
            self.ocr = None
            logger.warning("OCR processor not available")

    def capture_project_view(self) -> Optional[np.ndarray]:
        """Capture the current view of MS Project

        Returns:
            numpy.ndarray: Screenshot as a numpy array in BGR format, or None if capture failed
        """
        try:
            # Attempt to focus MS Project window (this varies by platform)
            if not self._focus_msproject_window():
                logger.warning("Failed to focus MS Project window. Capture may include other windows.")

            # Brief pause to ensure window is in focus
            time.sleep(1)

            # Capture the screen
            logger.info("Capturing screenshot...")
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            logger.info(f"Screenshot captured: {screenshot.shape}")

            return screenshot
        except Exception as e:
            logger.error(f"Error capturing MS Project view: {e}", exc_info=True)
            return None

    def _focus_msproject_window(self) -> bool:
        """Focus the MS Project window - Windows specific

        Returns:
            bool: True if successfully focused the window, False otherwise
        """
        try:
            # Import Windows-specific modules
            try:
                import win32gui
                import win32con
            except ImportError:
                logger.warning("win32gui not available - window focus not supported")
                return False

            # Define callback function to find MS Project windows
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd) and "Microsoft Project" in win32gui.GetWindowText(hwnd):
                    windows.append((hwnd, win32gui.GetWindowText(hwnd)))

            # Find all MS Project windows
            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                # Log found windows
                logger.info(f"Found {len(windows)} MS Project windows:")
                for i, (hwnd, title) in enumerate(windows):
                    logger.info(f"  {i+1}. {title} (hwnd: {hwnd})")

                # Focus the first found MS Project window
                hwnd = windows[0][0]
                win32gui.SetForegroundWindow(hwnd)
                logger.info(f"Focused window: {windows[0][1]}")
                return True
            else:
                logger.warning("MS Project window not found")
                return False
        except Exception as e:
            logger.error(f"Error focusing MS Project window: {e}", exc_info=True)
            return False

    def detect_project_elements(self, with_ocr: bool = True) -> Optional[Dict[str, Any]]:
        """Detect project elements in the current MS Project view

        Args:
            with_ocr: Whether to perform OCR on detected elements

        Returns:
            Dict containing detection results and project data, or None if detection failed
        """
        # Capture the current view
        logger.info("Capturing MS Project view...")
        screenshot = self.capture_project_view()
        if screenshot is None:
            logger.error("Failed to capture MS Project view")
            return None

        # Run detection on the screenshot
        logger.info("Running object detection...")
        results = self.detect_objects("msproject", screenshot)

        # Process the results for MS Project specific analysis
        if results and 'detections' in results:
            # Apply OCR if requested and available
            if with_ocr and HAS_OCR and self.ocr and hasattr(self.ocr, 'tesseract_available') and self.ocr.tesseract_available:
                logger.info("Applying OCR to detected elements...")
                # Process detections with OCR
                results['detections'] = self.ocr.process_detections(screenshot, results['detections'])
            elif with_ocr:
                logger.warning("OCR requested but not available. Skipping text extraction.")

            # Extract project data from detections
            logger.info(f"Processing {len(results['detections'])} detections...")
            project_data = self._extract_project_data(results['detections'])

            # Add project data to results
            results['project_data'] = project_data

            # Log summary of extracted data
            logger.info(f"Extracted project data from {len(results['detections'])} detections:")
            for key, items in project_data.items():
                logger.info(f"  - {key}: {len(items)} items")
        else:
            logger.warning("No detections found or detection failed")

        return results

    def _extract_project_data(self, detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract structured project data from detections

        Args:
            detections: List of detection dictionaries from detect_objects

        Returns:
            Dict containing categorized project elements
        """
        # If OCR is available, use it to extract project data
        if HAS_OCR and self.ocr:
            return self.ocr.extract_project_data(None, detections)

        # Fallback to manual extraction if OCR is not available
        project_data = {
            'tasks': [],
            'milestones': [],
            'resources': [],
            'dependencies': [],
            'critical_path': [],
            'summary': [],
            'constraints': [],
            'deadlines': [],
            'baselines': []
        }

        # Group detections by type
        for det in detections:
            obj_class = det['class']

            if obj_class == 'task' or obj_class == 'critical_task':
                task_data = {
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'is_critical': obj_class == 'critical_task',
                    'ocr_text': det['ocr_text']
                }
                project_data['tasks'].append(task_data)

                if obj_class == 'critical_task':
                    project_data['critical_path'].append(task_data)

            elif obj_class == 'milestone':
                project_data['milestones'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'resource':
                project_data['resources'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'dependency':
                project_data['dependencies'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'summary':
                project_data['summary'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

        return project_data

    def analyze_critical_path(self):
        """Analyze the critical path of the project"""
        # Detect project elements
        results = self.detect_project_elements()
        if results is None or 'project_data' not in results:
            return None

        project_data = results['project_data']

        # If critical path tasks are detected
        if project_data['critical_path']:
            # Sort tasks by x-coordinate to get sequence
            critical_tasks = sorted(project_data['critical_path'], key=lambda t: t['bbox'][0])

            # Calculate total critical path length
            total_tasks = len(critical_tasks)

            # Simple analysis
            analysis = {
                'critical_path_length': total_tasks,
                'critical_tasks': critical_tasks,
                'analysis': f"Detected {total_tasks} tasks on the critical path."
            }

            return analysis
        else:
            return {'critical_path_length': 0, 'critical_tasks': [], 'analysis': "No critical path detected."}

    def analyze_resource_allocation(self):
        """Analyze resource allocation in the project"""
        # Detect project elements
        results = self.detect_project_elements()
        if results is None or 'project_data' not in results:
            return None

        project_data = results['project_data']

        # Count resources
        resource_count = len(project_data['resources'])

        # Simple analysis
        analysis = {
            'resource_count': resource_count,
            'resources': project_data['resources'],
            'analysis': f"Detected {resource_count} resources allocated in the project."
        }

        return analysis

    def generate_project_report(self):
        """Generate a comprehensive report of the MS Project analysis"""
        # Detect project elements
        results = self.detect_project_elements()
        if results is None or 'project_data' not in results:
            return "No project data detected."

        project_data = results['project_data']

        # Basic counts
        task_count = len(project_data['tasks'])
        milestone_count = len(project_data['milestones'])
        resource_count = len(project_data['resources'])
        critical_path_length = len(project_data['critical_path'])

        # Generate report
        report = "Microsoft Project Analysis Report\n"
        report += "================================\n\n"

        report += f"Tasks: {task_count}\n"
        report += f"Milestones: {milestone_count}\n"
        report += f"Resources: {resource_count}\n"
        report += f"Critical Path Tasks: {critical_path_length}\n\n"

        # Add critical path analysis
        if critical_path_length > 0:
            report += "Critical Path Analysis:\n"
            report += "-----------------------\n"

            for i, task in enumerate(sorted(project_data['critical_path'], key=lambda t: t['bbox'][0])):
                report += f"  {i+1}. {task['ocr_text'] or 'Task ' + str(task['id'])}\n"

        # Add milestone analysis
        if milestone_count > 0:
            report += "\nMilestone Analysis:\n"
            report += "------------------\n"

            for i, milestone in enumerate(project_data['milestones']):
                report += f"  {i+1}. {milestone['ocr_text'] or 'Milestone ' + str(milestone['id'])}\n"

        # Add resource analysis
        if resource_count > 0:
            report += "\nResource Analysis:\n"
            report += "------------------\n"

            for i, resource in enumerate(project_data['resources']):
                report += f"  {i+1}. {resource['ocr_text'] or 'Resource ' + str(resource['id'])}\n"

        return report

def main():
    # Configure logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(SCRIPT_DIR, 'msproject_detector.log'))
        ]
    )

    # Parse command line arguments
    import argparse
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

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    try:
        # Create detector
        detector = MSProjectDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=args.tesseract
        )

        # Start MS Project if requested
        if args.start:
            logger.info("Starting Microsoft Project...")
            success = detector.start_application("msproject")
            if not success:
                logger.error("Failed to start Microsoft Project")
                sys.exit(1)

            # Wait for application to start
            logger.info("Waiting for application to start...")
            time.sleep(5)

        # Detect project elements
        logger.info("Detecting project elements...")
        results = detector.detect_project_elements(with_ocr=not args.no_ocr)

        if results:
            logger.info(f"Detected {results['detection_count']} elements")

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
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
