import os
import sys
import torch
import numpy as np
from PIL import ImageGrab
import time
from datetime import datetime
import sqlite3

# Add YOLOv5 path to system path
yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(yolo_path)

class MSProjectDetector:
    """Detect and analyze elements in Microsoft Project interface"""
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'project_detections.db')
        self._init_database()
        self.model = None
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _init_database(self):
        """Initialize SQLite database for storing project data and detections"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create detection_runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                image_path TEXT,
                detection_count INTEGER
            )
        ''')
        
        # Create detection_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                object_class TEXT,
                confidence REAL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                extracted_text TEXT,
                FOREIGN KEY (run_id) REFERENCES detection_runs (id)
            )
        ''')
        
        # Check if the table already exists and has the required columns
        cursor.execute("PRAGMA table_info(detection_results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # If the table exists but doesn't have the run_id column, alter the table
        if 'run_id' not in columns and columns:  # Only if table exists and missing column
            print("Adding run_id column to detection_results table...")
            cursor.execute("ALTER TABLE detection_results ADD COLUMN run_id INTEGER")
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def load_model(self, weights=None):
        """Load YOLOv5 model"""
        try:
            if weights is None:
                # Use a pre-trained model
                print("Loading YOLOv5s model...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            else:
                # Load custom weights
                print(f"Loading custom weights from {weights}...")
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
                
            model.conf = 0.25  # confidence threshold
            model.iou = 0.45   # NMS IoU threshold
            self.model = model
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def capture_screen(self):
        """Capture the current screen"""
        try:
            print("Capturing screen...")
            screenshot = ImageGrab.grab()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
            screenshot.save(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            return screenshot_path
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def detect_elements(self, image_path=None):
        """Run YOLOv5 detection on the image"""
        if self.model is None:
            if not self.load_model():
                print("Failed to load YOLOv5 model. Aborting detection.")
                return None
        
        if image_path is None:
            image_path = self.capture_screen()
            if image_path is None:
                return None
        
        try:
            print(f"Running YOLOv5 detection on {image_path}...")
            results = self.model(image_path)
            
            # Save detection results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"detection_{timestamp}")
            results.save(save_dir=self.output_dir)
            
            # Extract and store detections
            detections = self._process_results(results, image_path)
            print(f"Detection completed. Found {len(detections)} objects.")
            return detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def _process_results(self, results, image_path):
        """Process detection results and store in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create a new detection run
            cursor.execute('''
                INSERT INTO detection_runs (timestamp, image_path, detection_count)
                VALUES (?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                image_path,
                len(results.xyxy[0])
            ))
            
            run_id = cursor.lastrowid
            
            detections = []
            # Get the detection results
            if hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
                # Pandas DataFrame format
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    det = {
                        'class': row['name'],
                        'confidence': row['confidence'],
                        'box': (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    }
                    detections.append(det)
                    
                    # Store in database
                    cursor.execute('''
                        INSERT INTO detection_results 
                        (run_id, object_class, confidence, x1, y1, x2, y2, extracted_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        run_id,
                        det['class'],
                        det['confidence'],
                        det['box'][0], det['box'][1], det['box'][2], det['box'][3],
                        ""  # No text extraction yet
                    ))
            else:
                # Tensor format
                for det_tensor in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det_tensor.tolist()
                    cls_name = results.names[int(cls)]
                    
                    det = {
                        'class': cls_name,
                        'confidence': conf,
                        'box': (x1, y1, x2, y2)
                    }
                    detections.append(det)
                    
                    # Store in database
                    cursor.execute('''
                        INSERT INTO detection_results 
                        (run_id, object_class, confidence, x1, y1, x2, y2, extracted_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        run_id,
                        det['class'],
                        det['confidence'],
                        det['box'][0], det['box'][1], det['box'][2], det['box'][3],
                        ""  # No text extraction yet
                    ))
            
            conn.commit()
            conn.close()
            
            # Print detection results
            self._print_detections(detections)
            
            return detections
        except Exception as e:
            print(f"Error processing results: {e}")
            return []
    
    def _print_detections(self, detections):
        """Print detection results in a readable format"""
        print("\nDetection Results:")
        print("-" * 50)
        print(f"{'Class':<15} {'Confidence':<10} {'Box Coordinates':<30}")
        print("-" * 50)
        
        for det in detections:
            box_str = f"({det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f})"
            print(f"{det['class']:<15} {det['confidence']:.4f}      {box_str:<30}")
    
    def analyze_project_screen(self):
        """Analyze the Microsoft Project screen and identify project elements"""
        # Classes specific to Microsoft Project
        ms_project_classes = [
            'task', 'milestone', 'resource', 'gantt_bar', 'critical_path',
            'dependency', 'constraint', 'deadline', 'baseline', 'progress'
        ]
        
        print("\nAnalyzing Microsoft Project screen...")
        print("Note: YOLOv5 will use general object detection since no custom model")
        print("for Microsoft Project elements is available yet.\n")
        
        # Capture and detect
        detections = self.detect_elements()
        
        if not detections:
            print("No elements detected or an error occurred.")
            return
        
        # In this proof-of-concept, we're using standard YOLOv5 classes
        # A full implementation would use a custom-trained model for MS Project elements
        print("\nProjectect Element Analysis:")
        print("-" * 50)
        
        # Sample interpretation (for demonstration)
        print("Identified project elements (simulated):")
        print("- 3 tasks in view")
        print("- 1 milestone")
        print("- 2 critical path elements")
        print("- 4 resource assignments")
        
        print("\nIn an enhanced version, we would:")
        print("1. Use OCR to extract text from tasks and resources")
        print("2. Analyze Gantt chart elements to detect dependencies")
        print("3. Identify critical path based on formatting")
        print("4. Extract dates and durations from timeline elements")

def main():
    detector = MSProjectDetector()
    
    print("YOLOv5 Microsoft Project Screen Detector")
    print("=======================================")
    
    # Detect elements in the current screen
    detector.analyze_project_screen()

if __name__ == "__main__":
    main()
