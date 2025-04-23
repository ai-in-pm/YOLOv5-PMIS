import os
import sys
import subprocess
import time
import json
import sqlite3
import numpy as np
import torch
import cv2
import pyautogui
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageGrab
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'pm_integration.log'))
    ]
)
logger = logging.getLogger('pm_integration')

# Add YOLOv5 directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
yolov5_path = str(ROOT_DIR / 'yolov5-master')
sys.path.append(yolov5_path)

# Import YOLOv5 modules
try:
    # Try importing from the yolov5 directory
    sys.path.insert(0, yolov5_path)
    try:
        from models.common import DetectMultiBackend
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.torch_utils import select_device
        logger.info("YOLOv5 modules imported successfully")
    except ImportError:
        # For compatibility with different YOLOv5 versions
        from models.common import DetectMultiBackend
        from utils.general import check_img_size, non_max_suppression
        # In newer versions, scale_coords was renamed to scale_boxes
        try:
            from utils.general import scale_coords
        except ImportError:
            from utils.general import scale_boxes as scale_coords
        from utils.torch_utils import select_device
        logger.info("YOLOv5 modules imported successfully (compatibility mode)")
except ImportError as e:
    logger.error(f"Error importing YOLOv5 modules: {e}")
    sys.exit(1)

# Define common paths for project management applications
MS_PROJECT_PATHS = [
    r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINPROJ.EXE",
    r"C:\Program Files\Microsoft Office\Office16\WINPROJ.EXE",
    r"C:\Program Files (x86)\Microsoft Office\Office16\WINPROJ.EXE"
]

PRIMAVERA_PATHS = [
    r"C:\Program Files\Oracle\Primavera P6\P6 Professional\23.12.1\PM.exe",
    r"C:\Program Files\Oracle\Primavera P6\P6 Professional\PM.exe"
]

# Find the first existing path
def find_executable(paths: List[str]) -> Optional[str]:
    """Find the first existing executable from a list of possible paths"""
    for path in paths:
        if os.path.exists(path):
            return path
    return None

# Define paths for project management applications
MS_PROJECT_PATH = find_executable(MS_PROJECT_PATHS) or MS_PROJECT_PATHS[0]
PRIMAVERA_PATH = find_executable(PRIMAVERA_PATHS) or PRIMAVERA_PATHS[0]

class ProjectManagementDetector:
    """AI-powered detector for project management data from application windows"""

    def __init__(self, model_path: str = "yolov5s.pt", conf_thres: float = 0.25, iou_thres: float = 0.45):
        """Initialize the ProjectManagementDetector with model and detection parameters

        Args:
            model_path: Path to the YOLOv5 model file
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for non-maximum suppression
        """
        # Resolve model path
        if not os.path.isabs(model_path):
            # Try multiple locations for the model
            possible_paths = [
                os.path.join(yolov5_path, model_path),
                os.path.join(SCRIPT_DIR, 'models', model_path),
                os.path.join(ROOT_DIR, 'models', model_path)
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            else:
                # If model not found, use the first path as default
                self.model_path = possible_paths[0]
                logger.warning(f"Model not found at any expected location. Will attempt to use: {self.model_path}")
        else:
            self.model_path = model_path

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device('')
        self.db_path = os.path.join(SCRIPT_DIR.parent, 'database', 'pm_detections.db')
        self.model = None
        self.names = None
        self.window_title = None
        self.last_detection_time = None

        # Create necessary directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self._init_database()

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the YOLOv5 model

        Loads the model and prepares it for inference.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                # Try to download the model if it's a standard YOLOv5 model
                if self.model_path.endswith('.pt') and os.path.basename(self.model_path).startswith('yolov5'):
                    logger.info(f"Attempting to download standard model: {os.path.basename(self.model_path)}")
                    # This would typically use torch.hub.load or a similar mechanism
                    # For now, we'll just warn the user
                    logger.warning("Automatic model download not implemented. Please download the model manually.")
                return

            self.model = DetectMultiBackend(self.model_path, device=self.device)
            self.stride = self.model.stride
            self.names = self.model.names
            self.pt = self.model.pt
            self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
            self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
            logger.info(f"Model initialized: {self.model_path}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            # Don't exit, allow fallback to default model
            self.model = None

    def _init_database(self):
        """Initialize the SQLite database for storing detections"""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create PM applications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    path TEXT,
                    last_accessed TIMESTAMP
                )
            ''')

            # Create detection sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_id INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    window_title TEXT,
                    detection_count INTEGER,
                    FOREIGN KEY (app_id) REFERENCES pm_applications (id)
                )
            ''')

            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TIMESTAMP,
                    object_type TEXT,
                    confidence REAL,
                    x_min REAL,
                    y_min REAL,
                    x_max REAL,
                    y_max REAL,
                    screenshot_path TEXT,
                    ocr_text TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES detection_sessions (id)
                )
            ''')

            # Insert application records if they don't exist
            cursor.execute("SELECT COUNT(*) FROM pm_applications WHERE path = ?", (MS_PROJECT_PATH,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO pm_applications (name, path) VALUES (?, ?)", ("Microsoft Project", MS_PROJECT_PATH))

            cursor.execute("SELECT COUNT(*) FROM pm_applications WHERE path = ?", (PRIMAVERA_PATH,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO pm_applications (name, path) VALUES (?, ?)", ("Primavera P6", PRIMAVERA_PATH))

            conn.commit()
            conn.close()
            print(f"Database initialized: {self.db_path}")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def start_application(self, app_type: str) -> bool:
        """Start the specified project management application

        Args:
            app_type: Type of application to start ('msproject' or 'primavera')

        Returns:
            bool: True if application started successfully, False otherwise
        """
        app_type = app_type.lower()
        if app_type == "msproject":
            app_paths = MS_PROJECT_PATHS
            app_path = MS_PROJECT_PATH
            app_name = "Microsoft Project"
        elif app_type == "primavera":
            app_paths = PRIMAVERA_PATHS
            app_path = PRIMAVERA_PATH
            app_name = "Primavera P6"
        else:
            logger.error(f"Unknown application type: {app_type}")
            return False

        # Check if the application exists at the expected path
        if not os.path.exists(app_path):
            logger.warning(f"{app_name} not found at {app_path}")
            # Try to find the application in alternative locations
            app_path = find_executable(app_paths)
            if not app_path:
                logger.error(f"{app_name} not found in any expected location")
                return False
            logger.info(f"Found {app_name} at alternative location: {app_path}")

        try:
            logger.info(f"Starting {app_name} from {app_path}...")
            subprocess.Popen(app_path)

            # Update last_accessed timestamp in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE pm_applications SET last_accessed = ? WHERE path = ?",
                          (datetime.now().isoformat(), app_path))
            conn.commit()
            conn.close()

            # Wait for application to start
            logger.info(f"Waiting for {app_name} to start...")
            time.sleep(5)  # Adjust as needed
            return True
        except Exception as e:
            logger.error(f"Error starting {app_name}: {e}")
            return False

    def detect_objects(self, app_type: str, screenshot: Optional[Union[np.ndarray, str]] = None) -> Optional[Dict]:
        """Detect objects in the application window

        Args:
            app_type: Type of application to analyze ('msproject' or 'primavera')
            screenshot: Optional pre-captured screenshot as numpy array or path to image file.
                        If None, a new screenshot will be taken.

        Returns:
            Dict containing detection results or None if detection failed
        """
        app_type = app_type.lower()
        if app_type == "msproject":
            app_name = "Microsoft Project"
        elif app_type == "primavera":
            app_name = "Primavera P6"
        else:
            logger.error(f"Unknown application type: {app_type}")
            return None

        # Check if model is initialized
        if self.model is None:
            logger.error("Model not initialized. Cannot perform detection.")
            return None

        # Process the screenshot input
        if screenshot is None:
            # Take a new screenshot if none provided
            try:
                logger.info("Capturing screenshot...")
                screenshot = ImageGrab.grab()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                logger.info(f"Screenshot captured: {screenshot.shape}")
            except Exception as e:
                logger.error(f"Error capturing screenshot: {e}")
                return None
        elif isinstance(screenshot, str):
            # Load image from file path
            try:
                logger.info(f"Loading image from file: {screenshot}")
                if not os.path.exists(screenshot):
                    logger.error(f"Image file not found: {screenshot}")
                    return None
                screenshot = cv2.imread(screenshot)
                if screenshot is None or screenshot.size == 0:
                    logger.error(f"Failed to load image: {screenshot}")
                    return None
                logger.info(f"Image loaded: {screenshot.shape}")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return None
        elif not isinstance(screenshot, np.ndarray):
            logger.error(f"Invalid screenshot type: {type(screenshot)}. Expected numpy array or file path.")
            return None

        # Ensure the image is in BGR format (OpenCV default)
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
            # Check if we need to convert from RGB to BGR
            if hasattr(screenshot, '_is_rgb') and screenshot._is_rgb:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        else:
            logger.warning(f"Unusual image format: {screenshot.shape}. Attempting to process anyway.")

        # Start a new detection session
        session_id = self._start_detection_session(app_type)
        if session_id is None:
            logger.error("Failed to start detection session")
            return None

        try:
            # Preprocess image for YOLO
            logger.debug(f"Preprocessing image for detection: original shape {screenshot.shape}")
            img = cv2.resize(screenshot, self.imgsz)
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            logger.debug("Running inference with YOLOv5 model")
            pred = self.model(img)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)
            logger.debug(f"Non-maximum suppression applied with thresholds: conf={self.conf_thres}, iou={self.iou_thres}")

            # Process detections
            detection_count = 0
            detection_results = []

            for i, det in enumerate(pred):  # per image
                if len(det):
                    logger.debug(f"Processing {len(det)} detections")
                    # Rescale boxes from img_size to screenshot size
                    try:
                        # Try the newer function name first
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], screenshot.shape).round()
                    except NameError:
                        # Fall back to the older function name
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], screenshot.shape).round()

                    # Save each detection
                    for *xyxy, conf, cls in reversed(det):
                        x_min, y_min, x_max, y_max = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                        cls_id = int(cls.item())

                        # Ensure class index is valid
                        if cls_id < len(self.names):
                            cls_name = self.names[cls_id]
                        else:
                            cls_name = f"class_{cls_id}"
                            logger.warning(f"Unknown class ID: {cls_id}, using generic name: {cls_name}")

                        confidence = conf.item()

                        # Extract region for OCR (simplified - would need proper OCR integration)
                        # Ensure coordinates are within image bounds
                        y_min_safe = max(0, int(y_min))
                        y_max_safe = min(screenshot.shape[0], int(y_max))
                        x_min_safe = max(0, int(x_min))
                        x_max_safe = min(screenshot.shape[1], int(x_max))

                        # Only extract region if coordinates are valid
                        if y_min_safe < y_max_safe and x_min_safe < x_max_safe:
                            region = screenshot[y_min_safe:y_max_safe, x_min_safe:x_max_safe]
                            ocr_text = ""  # Default empty text
                        else:
                            logger.warning(f"Invalid region coordinates: [{x_min_safe}:{x_max_safe}, {y_min_safe}:{y_max_safe}]")
                            region = None
                            ocr_text = ""

                        # Save detection to database
                        detection_id = self._save_detection(
                            session_id,
                            cls_name,
                            confidence,
                            x_min, y_min, x_max, y_max,
                            ocr_text
                        )

                        detection_count += 1
                        detection_results.append({
                            'id': detection_id,
                            'class': cls_name,
                            'confidence': confidence,
                            'bbox': [x_min, y_min, x_max, y_max],
                            'ocr_text': ocr_text
                        })
                else:
                    logger.debug("No detections found after NMS")

            # Update session with detection count
            self._update_session(session_id, detection_count)

            # Draw boxes on image (for display/debug)
            annotated_img = self._draw_detections(screenshot.copy(), detection_results)

            # Save the annotated image
            results_dir = os.path.join(SCRIPT_DIR.parent, 'results', app_type.lower())
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(results_dir, f"{app_type.lower()}_detection_{timestamp}.jpg")
            cv2.imwrite(image_path, annotated_img)

            logger.info(f"Detected {detection_count} objects in {app_name}")
            logger.info(f"Results saved to {image_path}")

            return {
                'session_id': session_id,
                'detection_count': detection_count,
                'detections': detection_results,
                'annotated_image_path': image_path
            }

        except Exception as e:
            logger.error(f"Error detecting objects: {e}", exc_info=True)
            return None
        finally:
            # End the detection session
            self._end_detection_session(session_id)

    def _start_detection_session(self, app_type):
        """Start a new detection session in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get app_id
            app_name = "Microsoft Project" if app_type.lower() == "msproject" else "Primavera P6"
            cursor.execute("SELECT id FROM pm_applications WHERE name = ?", (app_name,))
            app_id = cursor.fetchone()[0]

            # Create new session
            start_time = datetime.now().isoformat()
            self.window_title = self._get_active_window_title()  # Get current window title

            cursor.execute(
                "INSERT INTO detection_sessions (app_id, start_time, window_title, detection_count) VALUES (?, ?, ?, ?)",
                (app_id, start_time, self.window_title, 0)
            )

            session_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return session_id
        except Exception as e:
            print(f"Error starting detection session: {e}")
            return None

    def _end_detection_session(self, session_id):
        """End a detection session in the database"""
        if session_id is None:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            end_time = datetime.now().isoformat()
            cursor.execute(
                "UPDATE detection_sessions SET end_time = ? WHERE id = ?",
                (end_time, session_id)
            )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error ending detection session: {e}")

    def _update_session(self, session_id, detection_count):
        """Update the detection count for a session"""
        if session_id is None:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE detection_sessions SET detection_count = ? WHERE id = ?",
                (detection_count, session_id)
            )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating session: {e}")

    def _save_detection(self, session_id, object_type, confidence, x_min, y_min, x_max, y_max, ocr_text):
        """Save a detection to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            cursor.execute(
                """INSERT INTO pm_detections
                (session_id, timestamp, object_type, confidence, x_min, y_min, x_max, y_max, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, timestamp, object_type, confidence, x_min, y_min, x_max, y_max, ocr_text)
            )

            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return detection_id
        except Exception as e:
            print(f"Error saving detection: {e}")
            return None

    def _draw_detections(self, image, detections):
        """Draw detection boxes on the image"""
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            cls_name = det['class']
            confidence = det['confidence']

            color = (0, 255, 0)  # Green
            thickness = 2

            # Draw rectangle
            cv2.rectangle(image,
                         (int(x_min), int(y_min)),
                         (int(x_max), int(y_max)),
                         color, thickness)

            # Draw label
            label = f"{cls_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1

            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            cv2.rectangle(image,
                         (int(x_min), int(y_min) - text_height - 5),
                         (int(x_min) + text_width, int(y_min)),
                         color, -1)

            cv2.putText(image, label,
                       (int(x_min), int(y_min) - 5),
                       font, font_scale, (0, 0, 0), text_thickness)

        return image

    def _get_active_window_title(self):
        """Get the title of the active window (platform specific)"""
        try:
            # This is Windows-specific
            import win32gui
            window = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(window)
        except ImportError:
            return "Unknown Window"  # Fallback
        except Exception as e:
            print(f"Error getting window title: {e}")
            return "Unknown Window"

    def generate_report(self, app_type=None):
        """Generate a report of detections"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # If app_type is specified, filter by app_type
            if app_type:
                app_name = "Microsoft Project" if app_type.lower() == "msproject" else "Primavera P6"

                query = """
                SELECT
                    s.id, a.name, s.start_time, s.end_time, s.window_title, s.detection_count,
                    COUNT(d.id) as actual_detection_count,
                    GROUP_CONCAT(DISTINCT d.object_type) as object_types
                FROM detection_sessions s
                JOIN pm_applications a ON s.app_id = a.id
                LEFT JOIN pm_detections d ON s.id = d.session_id
                WHERE a.name = ?
                GROUP BY s.id
                ORDER BY s.start_time DESC
                LIMIT 10
                """
                cursor.execute(query, (app_name,))
            else:
                query = """
                SELECT
                    s.id, a.name, s.start_time, s.end_time, s.window_title, s.detection_count,
                    COUNT(d.id) as actual_detection_count,
                    GROUP_CONCAT(DISTINCT d.object_type) as object_types
                FROM detection_sessions s
                JOIN pm_applications a ON s.app_id = a.id
                LEFT JOIN pm_detections d ON s.id = d.session_id
                GROUP BY s.id
                ORDER BY s.start_time DESC
                LIMIT 10
                """
                cursor.execute(query)

            sessions = cursor.fetchall()

            # Get object type distribution
            cursor.execute("""
                SELECT object_type, COUNT(*) as count
                FROM pm_detections
                GROUP BY object_type
                ORDER BY count DESC
            """)

            object_types = cursor.fetchall()

            conn.close()

            # Format report
            report = "\nDetection Report\n"
            report += "================\n\n"

            report += "Recent Detection Sessions:\n"
            report += "--------------------------\n"
            for session in sessions:
                session_id, app_name, start_time, end_time, window_title, detection_count, actual_count, object_types = session
                report += f"Session {session_id} ({app_name}):\n"
                report += f"  Started: {start_time}\n"
                report += f"  Ended: {end_time or 'In progress'}\n"
                report += f"  Window: {window_title}\n"
                report += f"  Detections: {actual_count}\n"
                report += f"  Objects: {object_types or 'None'}\n\n"

            report += "Object Type Distribution:\n"
            report += "-------------------------\n"
            for obj_type, count in object_types:
                report += f"  {obj_type}: {count}\n"

            print(report)
            return report
        except Exception as e:
            print(f"Error generating report: {e}")
            return None

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Project Management AI Detection")
    parser.add_argument('--app', type=str, choices=['msproject', 'primavera'], required=True,
                        help='Project management application to analyze')
    parser.add_argument('--action', type=str, choices=['start', 'detect', 'report'], required=True,
                        help='Action to perform')
    parser.add_argument('--model', type=str, default='yolov5s.pt',
                        help='YOLOv5 model path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Create detector
    try:
        detector = ProjectManagementDetector(model_path=args.model, conf_thres=args.conf, iou_thres=args.iou)

        # Perform requested action
        if args.action == 'start':
            success = detector.start_application(args.app)
            if not success:
                logger.error(f"Failed to start {args.app}")
                sys.exit(1)
        elif args.action == 'detect':
            results = detector.detect_objects(args.app)
            if results:
                logger.info(f"Detection complete. Found {results['detection_count']} objects.")
            else:
                logger.error("Detection failed")
                sys.exit(1)
        elif args.action == 'report':
            report = detector.generate_report(args.app)
            if not report:
                logger.error("Failed to generate report")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
