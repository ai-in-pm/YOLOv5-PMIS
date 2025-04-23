import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
from datetime import datetime
import sqlite3
import pyautogui
from pathlib import Path

# Add root path for shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

# Import OCR integration
from ocr_integration import MSProjectOCR

class MSProjectDetector:
    """Integrated Microsoft Project element detector with OCR"""
    
    def __init__(self, model_path=None, tesseract_path=None, database_path=None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tesseract_path = tesseract_path
        self.ocr = MSProjectOCR(tesseract_path=tesseract_path)
        
        # Set up database connection
        self.database_path = database_path or os.path.join(
            os.path.dirname(__file__), '..', 'database', 'ms_project_detections.db'
        )
        self._init_database()
        
        # Output directory for detection results
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Class names for MS Project elements
        self.class_names = [
            'task', 'milestone', 'resource', 'gantt_bar', 'critical_path',
            'dependency', 'constraint', 'deadline', 'baseline', 'progress'
        ]
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            'task': (0, 0, 255),        # Red
            'milestone': (0, 0, 128),    # Dark red
            'resource': (0, 255, 0),     # Green
            'gantt_bar': (255, 0, 0),    # Blue
            'critical_path': (0, 128, 255), # Orange
            'dependency': (255, 0, 255),  # Magenta
            'constraint': (255, 255, 0),  # Cyan
            'deadline': (128, 0, 128),    # Purple
            'baseline': (0, 255, 255),    # Yellow
            'progress': (0, 128, 0)       # Dark green
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing detection results"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create detection_runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                model_path TEXT,
                detection_count INTEGER
            )
        ''')
        
        # Create ms_project_elements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ms_project_elements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                element_type TEXT,
                confidence REAL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                extracted_text TEXT,
                FOREIGN KEY (run_id) REFERENCES detection_runs (id)
            )
        ''')
        
        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                detection_run_id INTEGER,
                timestamp TEXT,
                task_count INTEGER,
                milestone_count INTEGER,
                resource_count INTEGER,
                FOREIGN KEY (detection_run_id) REFERENCES detection_runs (id)
            )
        ''')
        
        # Create tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                name TEXT,
                start_date TEXT,
                end_date TEXT,
                duration TEXT,
                is_critical INTEGER,
                is_milestone INTEGER,
                resource_name TEXT,
                element_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (element_id) REFERENCES ms_project_elements (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.database_path}")
    
    def load_model(self, model_path=None):
        """Load YOLOv5 model for MS Project element detection"""
        model_path = model_path or self.model_path
        
        if not model_path:
            # Try to find the newest trained model
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            if os.path.exists(models_dir):
                model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                if model_dirs:
                    # Sort by modification time (newest first)
                    model_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(models_dir, d)), reverse=True)
                    best_model = os.path.join(models_dir, model_dirs[0], 'weights', 'best.pt')
                    
                    if os.path.exists(best_model):
                        model_path = best_model
        
        if not model_path or not os.path.exists(model_path):
            # Use standard YOLOv5s if no trained model is available
            print("No trained model found. Loading standard YOLOv5s...")
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        try:
            print(f"Loading model from {model_path}...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.to(self.device)
            print("Model loaded successfully!")
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def capture_screenshot(self):
        """Capture screenshot of the current screen"""
        try:
            print("Capturing screen...")
            screenshot = pyautogui.screenshot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
            screenshot.save(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            return screenshot_path
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def detect(self, image_path=None, confidence=0.25, iou=0.45, with_ocr=True):
        """Run detection on an image"""
        if image_path is None:
            image_path = self.capture_screenshot()
            if image_path is None:
                return None
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                print("Failed to load model. Aborting detection.")
                return None
        
        try:
            # Set model parameters
            self.model.conf = confidence  # confidence threshold
            self.model.iou = iou         # NMS IoU threshold
            
            print(f"Running detection on {image_path}...")
            results = self.model(image_path)
            
            # Save the results visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
            results.save(save_dir=self.output_dir)
            
            # Process results
            detections = self._process_results(results, image_path)
            
            # Apply OCR if requested
            if with_ocr and detections:
                img = cv2.imread(image_path)
                for det in detections:
                    element_type = det['class']
                    bbox = [det['x1'], det['y1'], det['x2'], det['y2']]
                    if element_type in ['task', 'milestone', 'resource']:
                        text = self.ocr.extract_from_detection(img, bbox, element_type)
                        det['text'] = text
            
            # Store results in database
            run_id = self._store_detection_run(image_path, detections)
            
            # Create results with absolute paths for output
            return {
                'run_id': run_id,
                'detections': detections,
                'image_path': image_path,
                'output_path': output_path
            }
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def _process_results(self, results, image_path):
        """Process YOLOv5 detection results"""
        detections = []
        
        # Process each detection
        if hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
            # Pandas DataFrame version
            df = results.pandas().xyxy[0]
            for i, row in df.iterrows():
                detection = {
                    'class': row['name'],
                    'confidence': float(row['confidence']),
                    'x1': float(row['xmin']),
                    'y1': float(row['ymin']),
                    'x2': float(row['xmax']),
                    'y2': float(row['ymax']),
                    'text': ""
                }
                detections.append(detection)
        else:
            # Tensor version
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            
            for i, det in enumerate(results.pred[0]):
                if len(det) >= 6:  # Should have at least 6 elements (x1, y1, x2, y2, conf, cls)
                    x1, y1, x2, y2, conf, cls = det.tolist()[:6]
                    cls_idx = int(cls)
                    class_name = results.names[cls_idx] if cls_idx < len(results.names) else f"class_{cls_idx}"
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'text': ""
                    }
                    detections.append(detection)
        
        return detections
    
    def _store_detection_run(self, image_path, detections):
        """Store detection results in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create a new detection run
        cursor.execute('''
            INSERT INTO detection_runs (timestamp, image_path, model_path, detection_count)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_path,
            self.model_path or "default_model",
            len(detections)
        ))
        
        run_id = cursor.lastrowid
        
        # Store each detection
        for det in detections:
            cursor.execute('''
                INSERT INTO ms_project_elements 
                (run_id, element_type, confidence, x1, y1, x2, y2, extracted_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                det['class'],
                det['confidence'],
                det['x1'], det['y1'], det['x2'], det['y2'],
                det.get('text', "")
            ))
        
        conn.commit()
        conn.close()
        
        print(f"Stored detection run with ID {run_id} in database")
        return run_id
    
    def analyze_schedule(self, detection_result):
        """Analyze project schedule from detection results"""
        if not detection_result or 'detections' not in detection_result:
            print("No detection results to analyze")
            return None
        
        detections = detection_result['detections']
        image_path = detection_result['image_path']
        run_id = detection_result['run_id']
        
        # Count elements by type
        element_counts = {}
        for det in detections:
            element_type = det['class']
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        task_count = element_counts.get('task', 0)
        milestone_count = element_counts.get('milestone', 0)
        resource_count = element_counts.get('resource', 0)
        
        # Create a project entry
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO projects (name, detection_run_id, timestamp, task_count, milestone_count, resource_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            f"Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_id,
            datetime.now().isoformat(),
            task_count,
            milestone_count,
            resource_count
        ))
        
        project_id = cursor.lastrowid
        
        # Extract tasks and store them
        tasks = [det for det in detections if det['class'] == 'task']
        for task in tasks:
            task_name = task.get('text', f"Task {tasks.index(task) + 1}")
            # Get element ID from database
            cursor.execute('''
                SELECT id FROM ms_project_elements 
                WHERE run_id = ? AND element_type = ? AND 
                      x1 = ? AND y1 = ? AND x2 = ? AND y2 = ?
            ''', (run_id, 'task', task['x1'], task['y1'], task['x2'], task['y2']))
            
            element_id = cursor.fetchone()
            if element_id:
                element_id = element_id[0]
            else:
                element_id = None
            
            # Check if it's on the critical path
            is_critical = 0
            for det in detections:
                if det['class'] == 'critical_path':
                    # Check if task is inside or overlaps with critical path
                    if self._is_overlapping(
                        [task['x1'], task['y1'], task['x2'], task['y2']],
                        [det['x1'], det['y1'], det['x2'], det['y2']]
                    ):
                        is_critical = 1
                        break
            
            # Store task in database
            cursor.execute('''
                INSERT INTO tasks 
                (project_id, name, is_critical, is_milestone, element_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                project_id,
                task_name,
                is_critical,
                0,  # Not a milestone
                element_id
            ))
        
        # Extract milestones and store them
        milestones = [det for det in detections if det['class'] == 'milestone']
        for milestone in milestones:
            milestone_name = milestone.get('text', f"Milestone {milestones.index(milestone) + 1}")
            # Get element ID from database
            cursor.execute('''
                SELECT id FROM ms_project_elements 
                WHERE run_id = ? AND element_type = ? AND 
                      x1 = ? AND y1 = ? AND x2 = ? AND y2 = ?
            ''', (run_id, 'milestone', milestone['x1'], milestone['y1'], milestone['x2'], milestone['y2']))
            
            element_id = cursor.fetchone()
            if element_id:
                element_id = element_id[0]
            else:
                element_id = None
            
            # Store milestone in database
            cursor.execute('''
                INSERT INTO tasks 
                (project_id, name, is_critical, is_milestone, element_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                project_id,
                milestone_name,
                1,  # Milestones are usually on critical path
                1,  # Is a milestone
                element_id
            ))
        
        conn.commit()
        conn.close()
        
        # Create analysis result
        analysis = {
            'project_id': project_id,
            'element_counts': element_counts,
            'tasks': [{'name': t.get('text', f"Task {i+1}"), 'box': [t['x1'], t['y1'], t['x2'], t['y2']]} 
                     for i, t in enumerate(tasks)],
            'milestones': [{'name': m.get('text', f"Milestone {i+1}"), 'box': [m['x1'], m['y1'], m['x2'], m['y2']]} 
                          for i, m in enumerate(milestones)]
        }
        
        return analysis
    
    def _is_overlapping(self, box1, box2):
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)
    
    def visualize_detection(self, image_path, detections, output_path=None):
        """Visualize detection results with OCR text"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Load image
        img = cv2.imread(image_path)
        
        # Create PIL Image for better text rendering
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Draw each detection
        for det in detections:
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            text = det.get('text', '')
            
            # Get color for this class
            color_rgb = tuple(reversed(self.colors.get(class_name, (0, 0, 255))))  # Convert BGR to RGB
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:4]
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color_rgb)
            
            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
            
            # Draw OCR text if available
            if text:
                draw.text((x1 + 2, y2 + 2), text[:20], fill=color_rgb, font=font)
        
        # Convert back to OpenCV format
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Save the result
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"visualized_{timestamp}.jpg")
        
        cv2.imwrite(output_path, result_img)
        print(f"Visualization saved to {output_path}")
        return output_path
    
    def generate_report(self, detection_result, analysis_result=None):
        """Generate a comprehensive report of detection and analysis results"""
        if not detection_result:
            print("No detection results to generate report from")
            return None
        
        detections = detection_result['detections']
        image_path = detection_result['image_path']
        run_id = detection_result['run_id']
        
        # Count elements by type
        element_counts = {}
        for det in detections:
            element_type = det['class']
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        # Format the report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"Microsoft Project Detection Report\n"
        report += f"=================================\n\n"
        report += f"Generated: {timestamp}\n"
        report += f"Image: {image_path}\n"
        report += f"Model: {self.model_path or 'default_model'}\n\n"
        
        report += f"Detection Summary:\n"
        report += f"------------------\n"
        report += f"Total elements detected: {len(detections)}\n"
        for element_type, count in element_counts.items():
            report += f"  - {element_type}: {count}\n"
        
        report += f"\nDetected Elements:\n"
        report += f"------------------\n"
        for i, det in enumerate(detections):
            report += f"Element {i+1}: {det['class']} (Confidence: {det['confidence']:.2f})\n"
            if 'text' in det and det['text']:
                report += f"  Text: {det['text']}\n"
            report += f"  Coordinates: [{det['x1']:.1f}, {det['y1']:.1f}, {det['x2']:.1f}, {det['y2']:.1f}]\n\n"
        
        if analysis_result:
            report += f"\nProject Analysis:\n"
            report += f"------------------\n"
            report += f"Project ID: {analysis_result['project_id']}\n\n"
            
            report += f"Tasks ({len(analysis_result['tasks'])}):\n"
            for i, task in enumerate(analysis_result['tasks']):
                report += f"  {i+1}. {task['name']}\n"
            
            report += f"\nMilestones ({len(analysis_result['milestones'])}):\n"
            for i, milestone in enumerate(analysis_result['milestones']):
                report += f"  {i+1}. {milestone['name']}\n"
        
        # Save the report to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description="Microsoft Project Element Detector with OCR")
    parser.add_argument('--image', type=str, help='Path to image file (optional, captures screenshot if not provided)')
    parser.add_argument('--model', type=str, help='Path to trained YOLOv5 model')
    parser.add_argument('--tesseract', type=str, help='Path to Tesseract executable')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--no-ocr', action='store_true', help='Disable OCR text extraction')
    parser.add_argument('--visualize', action='store_true', help='Visualize detection results')
    parser.add_argument('--analyze', action='store_true', help='Analyze project schedule from detections')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MSProjectDetector(model_path=args.model, tesseract_path=args.tesseract)
    
    # Run detection
    detection_result = detector.detect(
        image_path=args.image,
        confidence=args.confidence,
        with_ocr=not args.no_ocr
    )
    
    if detection_result:
        print(f"\nDetection Results:")
        print(f"Found {len(detection_result['detections'])} elements")
        
        # Visualize if requested
        if args.visualize:
            visualization_path = detector.visualize_detection(
                detection_result['image_path'],
                detection_result['detections']
            )
        
        # Analyze if requested
        analysis_result = None
        if args.analyze:
            analysis_result = detector.analyze_schedule(detection_result)
        
        # Generate report if requested
        if args.report:
            report_path = detector.generate_report(detection_result, analysis_result)
            print(f"\nFull report available at: {report_path}")
    else:
        print("Detection failed or no elements found")

if __name__ == "__main__":
    main()
