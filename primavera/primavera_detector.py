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
logger = logging.getLogger('primavera_detector')

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PARENT_DIR))

# Import the core integration module
from pm_integration import ProjectManagementDetector, PRIMAVERA_PATH, PRIMAVERA_PATHS

# Import OCR utilities
try:
    from ocr_utils import PrimaveraOCR, get_ocr_processor
    HAS_OCR = True
except ImportError:
    logger.warning("OCR utilities not available. Text extraction will be disabled.")
    HAS_OCR = False

class PrimaveraDetector(ProjectManagementDetector):
    """Specialized detector for Primavera P6 Professional elements"""

    def __init__(self, model_path: str = "primavera_model.pt", conf_thres: float = 0.25,
                 iou_thres: float = 0.45, tesseract_path: Optional[str] = None):
        """Initialize the PrimaveraDetector with model and detection parameters

        Args:
            model_path: Path to the YOLOv5 model file trained for Primavera detection
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

        # Define Primavera specific object classes
        self.object_classes = [
            'activity', 'milestone', 'wbs', 'critical_activity', 'resource',
            'relationship', 'constraint', 'progress_bar', 'baseline', 'notebook'
        ]

        # Initialize OCR processor
        self.tesseract_path = tesseract_path
        if HAS_OCR:
            self.ocr = PrimaveraOCR(tesseract_path)
            logger.info("OCR processor initialized")
        else:
            self.ocr = None
            logger.warning("OCR processor not available")

    def capture_primavera_view(self) -> Optional[np.ndarray]:
        """Capture the current view of Primavera P6

        Returns:
            numpy.ndarray: Screenshot as a numpy array in BGR format, or None if capture failed
        """
        try:
            # Attempt to focus Primavera window
            if not self._focus_primavera_window():
                logger.warning("Failed to focus Primavera window. Capture may include other windows.")

            # Brief pause to ensure window is in focus
            time.sleep(1)

            # Capture the screen
            logger.info("Capturing screenshot...")
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            logger.info(f"Screenshot captured: {screenshot.shape}")

            return screenshot
        except Exception as e:
            logger.error(f"Error capturing Primavera view: {e}", exc_info=True)
            return None

    def _focus_primavera_window(self) -> bool:
        """Focus the Primavera P6 window - Windows specific

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

            # Define callback function to find Primavera windows
            def callback(hwnd, windows):
                title = win32gui.GetWindowText(hwnd)
                if win32gui.IsWindowVisible(hwnd) and ("Primavera" in title or "P6 Professional" in title):
                    windows.append((hwnd, title))

            # Find all Primavera windows
            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                # Log found windows
                logger.info(f"Found {len(windows)} Primavera windows:")
                for i, (hwnd, title) in enumerate(windows):
                    logger.info(f"  {i+1}. {title} (hwnd: {hwnd})")

                # Focus the first found Primavera window
                hwnd = windows[0][0]
                win32gui.SetForegroundWindow(hwnd)
                logger.info(f"Focused window: {windows[0][1]}")
                return True
            else:
                logger.warning("Primavera P6 window not found")
                return False
        except Exception as e:
            logger.error(f"Error focusing Primavera window: {e}", exc_info=True)
            return False

    def detect_primavera_elements(self, with_ocr: bool = True) -> Optional[Dict[str, Any]]:
        """Detect project elements in the current Primavera P6 view

        Args:
            with_ocr: Whether to perform OCR on detected elements

        Returns:
            Dict containing detection results and project data, or None if detection failed
        """
        # Capture the current view
        logger.info("Capturing Primavera P6 view...")
        screenshot = self.capture_primavera_view()
        if screenshot is None:
            logger.error("Failed to capture Primavera P6 view")
            return None

        # Run detection on the screenshot
        logger.info("Running object detection...")
        results = self.detect_objects("primavera", screenshot)

        # Process the results for Primavera specific analysis
        if results and 'detections' in results:
            # Apply OCR if requested and available
            if with_ocr and HAS_OCR and self.ocr:
                logger.info("Applying OCR to detected elements...")
                # Process detections with OCR
                results['detections'] = self.ocr.process_detections(screenshot, results['detections'])

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

    def _extract_project_data(self, detections):
        """Extract structured project data from detections"""
        project_data = {
            'activities': [],
            'milestones': [],
            'wbs_elements': [],
            'resources': [],
            'relationships': [],
            'critical_path': [],
            'constraints': [],
            'notebooks': []
        }

        # Group detections by type
        for det in detections:
            obj_class = det['class']

            if obj_class == 'activity' or obj_class == 'critical_activity':
                activity_data = {
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'is_critical': obj_class == 'critical_activity',
                    'ocr_text': det['ocr_text']
                }
                project_data['activities'].append(activity_data)

                if obj_class == 'critical_activity':
                    project_data['critical_path'].append(activity_data)

            elif obj_class == 'milestone':
                project_data['milestones'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'wbs':
                project_data['wbs_elements'].append({
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

            elif obj_class == 'relationship':
                project_data['relationships'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'constraint':
                project_data['constraints'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

            elif obj_class == 'notebook':
                project_data['notebooks'].append({
                    'id': det['id'],
                    'bbox': det['bbox'],
                    'ocr_text': det['ocr_text']
                })

        return project_data

    def analyze_critical_path(self):
        """Analyze the critical path of the project"""
        # Detect project elements
        results = self.detect_primavera_elements()
        if results is None or 'project_data' not in results:
            return None

        project_data = results['project_data']

        # If critical path activities are detected
        if project_data['critical_path']:
            # Sort activities by x-coordinate to get sequence
            critical_activities = sorted(project_data['critical_path'], key=lambda t: t['bbox'][0])

            # Calculate total critical path length
            total_activities = len(critical_activities)

            # Simple analysis
            analysis = {
                'critical_path_length': total_activities,
                'critical_activities': critical_activities,
                'analysis': f"Detected {total_activities} activities on the critical path."
            }

            return analysis
        else:
            return {'critical_path_length': 0, 'critical_activities': [], 'analysis': "No critical path detected."}

    def analyze_wbs_structure(self):
        """Analyze the WBS structure of the project"""
        # Detect project elements
        results = self.detect_primavera_elements()
        if results is None or 'project_data' not in results:
            return None

        project_data = results['project_data']

        # Count WBS elements
        wbs_count = len(project_data['wbs_elements'])

        # Simple analysis
        analysis = {
            'wbs_count': wbs_count,
            'wbs_elements': project_data['wbs_elements'],
            'analysis': f"Detected {wbs_count} WBS elements in the project structure."
        }

        return analysis

    def analyze_resource_allocation(self):
        """Analyze resource allocation in the project"""
        # Detect project elements
        results = self.detect_primavera_elements()
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
        """Generate a comprehensive report of the Primavera P6 analysis"""
        # Detect project elements
        results = self.detect_primavera_elements()
        if results is None or 'project_data' not in results:
            return "No project data detected."

        project_data = results['project_data']

        # Basic counts
        activity_count = len(project_data['activities'])
        milestone_count = len(project_data['milestones'])
        wbs_count = len(project_data['wbs_elements'])
        resource_count = len(project_data['resources'])
        relationship_count = len(project_data['relationships'])
        critical_path_length = len(project_data['critical_path'])
        constraint_count = len(project_data['constraints'])
        notebook_count = len(project_data['notebooks'])

        # Generate report
        report = "Primavera P6 Professional Analysis Report\n"
        report += "=======================================\n\n"

        report += f"Activities: {activity_count}\n"
        report += f"Milestones: {milestone_count}\n"
        report += f"WBS Elements: {wbs_count}\n"
        report += f"Resources: {resource_count}\n"
        report += f"Relationships: {relationship_count}\n"
        report += f"Constraints: {constraint_count}\n"
        report += f"Notebooks: {notebook_count}\n"
        report += f"Critical Path Activities: {critical_path_length}\n\n"

        # Add critical path analysis
        if critical_path_length > 0:
            report += "Critical Path Analysis:\n"
            report += "-----------------------\n"

            for i, activity in enumerate(sorted(project_data['critical_path'], key=lambda t: t['bbox'][0])):
                report += f"  {i+1}. {activity['ocr_text'] or 'Activity ' + str(activity['id'])}\n"

        # Add WBS analysis
        if wbs_count > 0:
            report += "\nWBS Structure Analysis:\n"
            report += "----------------------\n"

            for i, wbs in enumerate(project_data['wbs_elements']):
                report += f"  {i+1}. {wbs['ocr_text'] or 'WBS Element ' + str(wbs['id'])}\n"

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
            logging.FileHandler(os.path.join(SCRIPT_DIR, 'primavera_detector.log'))
        ]
    )

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Primavera P6 Detection Tool")
    parser.add_argument('--model', type=str, default="primavera_model.pt",
                        help='Path to the YOLOv5 model file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--start', action='store_true',
                        help='Start Primavera P6 before detection')
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
        detector = PrimaveraDetector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou,
            tesseract_path=args.tesseract
        )

        # Start Primavera P6 if requested
        if args.start:
            logger.info("Starting Primavera P6 Professional...")
            success = detector.start_application("primavera")
            if not success:
                logger.error("Failed to start Primavera P6")
                sys.exit(1)

            # Wait for application to start
            logger.info("Waiting for application to start...")
            time.sleep(5)

        # Detect project elements
        logger.info("Detecting project elements...")
        results = detector.detect_primavera_elements(with_ocr=not args.no_ocr)

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
