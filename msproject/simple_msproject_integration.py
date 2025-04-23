import os
import sys
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import subprocess
import sqlite3

# Add YOLOv5 path to system path
yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(yolo_path)

class SimpleProjectIntegration:
    """A simplified integration between YOLOv5 and Microsoft Project using MPXJ concepts"""
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'project_detections.db')
        self._init_database()
        self.model = None
        
    def _init_database(self):
        """Initialize SQLite database for storing project data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                file_path TEXT,
                import_date TIMESTAMP,
                task_count INTEGER,
                resource_count INTEGER
            )
        ''')
        
        # Create tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                task_id TEXT,
                name TEXT,
                start_date TEXT,
                end_date TEXT,
                duration TEXT,
                is_critical INTEGER,
                is_milestone INTEGER,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Create detection_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                image_path TEXT,
                timestamp TIMESTAMP,
                object_class TEXT,
                confidence REAL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def load_model(self, weights='yolov5s.pt'):
        """Load YOLOv5 model"""
        try:
            # Try to load the model using the new torch hub approach
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
            model.conf = 0.25  # confidence threshold
            model.iou = 0.45   # NMS IoU threshold
            model.classes = None  # filter by class
            self.model = model
            print(f"Loaded YOLOv5 model from {weights}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def start_ms_project(self):
        """Start Microsoft Project application"""
        ms_project_path = r"C:\Program Files\Microsoft Office\root\Office16\WINPROJ.EXE"
        
        try:
            print(f"Starting Microsoft Project from {ms_project_path}...")
            subprocess.Popen(ms_project_path)
            print("Microsoft Project launched. Please open your project file.")
            return True
        except Exception as e:
            print(f"Error starting Microsoft Project: {e}")
            print("The application may not be installed at the expected location.")
            return False
    
    def detect_from_screenshot(self):
        """Take a screenshot and run YOLOv5 detection"""
        if self.model is None:
            if not self.load_model():
                print("Failed to load YOLOv5 model.")
                return False
        
        try:
            # Use PIL to take a screenshot
            from PIL import ImageGrab
            print("Taking screenshot...")
            screenshot = ImageGrab.grab()
            screenshot_np = np.array(screenshot)
            
            # Save screenshot temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(os.path.dirname(__file__), f"screenshot_{timestamp}.jpg")
            screenshot.save(screenshot_path)
            
            print(f"Running YOLOv5 detection on screenshot...")
            results = self.model(screenshot_path)
            
            # Save detection results
            detect_path = os.path.join(os.path.dirname(__file__), f"detection_{timestamp}.jpg")
            results.save(save_dir=os.path.dirname(detect_path), exist_ok=True)
            
            # Extract and store detections
            result_data = self._extract_detections(results, screenshot_path)
            
            print(f"Detection completed. Found {len(result_data)} objects.")
            print(f"Results saved to {os.path.abspath(os.path.dirname(detect_path))}")
            
            return result_data
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def _extract_detections(self, results, image_path):
        """Extract detection results from YOLOv5 output"""
        detections = []
        
        # Process each detection
        try:
            # First, detect if the Pandas version is used or tensor version
            if hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
                # Pandas version
                df = results.pandas().xyxy[0]  # Results as pandas DataFrame
                
                for i, row in df.iterrows():
                    detection = {
                        'class': row['name'],
                        'confidence': row['confidence'],
                        'box': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    }
                    detections.append(detection)
            else:
                # Tensor version (older YOLOv5 versions)
                for pred in results.pred[0]:
                    x1, y1, x2, y2, conf, cls = pred
                    cls_name = results.names[int(cls)]  # Get class name from class index
                    
                    detection = {
                        'class': cls_name,
                        'confidence': float(conf),
                        'box': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
            
            # Store detections in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get project ID (use 1 for now as default)
            project_id = 1
            
            # Insert detections
            for det in detections:
                cursor.execute('''
                    INSERT INTO detection_results 
                    (project_id, image_path, timestamp, object_class, confidence, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id, 
                    image_path,
                    datetime.now().isoformat(),
                    det['class'],
                    det['confidence'],
                    det['box'][0], det['box'][1], det['box'][2], det['box'][3]
                ))
            
            conn.commit()
            conn.close()
            
            return detections
        except Exception as e:
            print(f"Error extracting detections: {e}")
            return []
    
    def process_ms_project_file(self, file_path):
        """Process a Microsoft Project file using concepts from MPXJ integration"""
        print(f"\nProcessing Microsoft Project file: {file_path}")
        print("Note: This is a simplified version that simulates MPXJ integration.")
        print("In a full implementation, we would use actual MPXJ to extract data from the file.")
        
        # In a real implementation, we would use the MPXJ library to extract data from the file
        # Similar to what was described in the memory about MPXJ integration
        
        # Simulate project extraction (in real version, use MPXJ)
        project_info = {
            'name': os.path.basename(file_path),
            'file_path': file_path,
            'task_count': 42,  # Simulated count
            'resource_count': 8  # Simulated count
        }
        
        # Store project info in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO projects (name, file_path, import_date, task_count, resource_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            project_info['name'],
            project_info['file_path'],
            datetime.now().isoformat(),
            project_info['task_count'],
            project_info['resource_count']
        ))
        
        project_id = cursor.lastrowid
        
        # Simulate task extraction (in real version, use MPXJ)
        tasks = [
            {'id': '1', 'name': 'Project Planning', 'duration': '5d', 'is_critical': 1, 'is_milestone': 0},
            {'id': '2', 'name': 'Requirements Gathering', 'duration': '10d', 'is_critical': 1, 'is_milestone': 0},
            {'id': '3', 'name': 'Design Phase', 'duration': '15d', 'is_critical': 1, 'is_milestone': 0},
            {'id': '4', 'name': 'Implementation', 'duration': '20d', 'is_critical': 0, 'is_milestone': 0},
            {'id': '5', 'name': 'Testing', 'duration': '10d', 'is_critical': 0, 'is_milestone': 0},
            {'id': '6', 'name': 'Project Complete', 'duration': '0d', 'is_critical': 1, 'is_milestone': 1}
        ]
        
        for task in tasks:
            cursor.execute('''
                INSERT INTO tasks (project_id, task_id, name, duration, is_critical, is_milestone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                project_id,
                task['id'],
                task['name'],
                task['duration'],
                task['is_critical'],
                task['is_milestone']
            ))
        
        conn.commit()
        conn.close()
        
        print(f"\nProject information stored in database:")
        print(f"  Name: {project_info['name']}")
        print(f"  Tasks: {project_info['task_count']}")
        print(f"  Resources: {project_info['resource_count']}")
        print(f"\nExtracted {len(tasks)} sample tasks:")
        
        for task in tasks:
            critical = "(Critical)" if task['is_critical'] == 1 else ""
            milestone = "(Milestone)" if task['is_milestone'] == 1 else ""
            print(f"  {task['id']}: {task['name']} - {task['duration']} {critical} {milestone}")
        
        return project_info, tasks
    
    def generate_report(self):
        """Generate a report of all project data and detections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get projects
        cursor.execute("SELECT * FROM projects")
        projects = cursor.fetchall()
        
        # Get tasks
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
        
        # Get detections
        cursor.execute("SELECT * FROM detection_results")
        detections = cursor.fetchall()
        
        conn.close()
        
        report = "YOLOv5 Microsoft Project Integration Report\n"
        report += "===========================================\n\n"
        
        # Projects section
        report += f"Projects ({len(projects)}):\n"
        report += "------------------\n"
        for proj in projects:
            id, name, file_path, import_date, task_count, resource_count = proj
            report += f"  ID: {id}\n"
            report += f"  Name: {name}\n"
            report += f"  File: {file_path}\n"
            report += f"  Imported: {import_date}\n"
            report += f"  Tasks: {task_count}\n"
            report += f"  Resources: {resource_count}\n\n"
        
        # Tasks section
        report += f"Tasks ({len(tasks)}):\n"
        report += "---------------\n"
        for task in tasks:
            id, project_id, task_id, name, start_date, end_date, duration, is_critical, is_milestone = task
            critical = "(Critical)" if is_critical == 1 else ""
            milestone = "(Milestone)" if is_milestone == 1 else ""
            report += f"  {task_id}: {name} - {duration} {critical} {milestone}\n"
        
        # Detections section
        report += f"\nYOLOv5 Detections ({len(detections)}):\n"
        report += "-----------------------------\n"
        for det in detections:
            id, project_id, image_path, timestamp, obj_class, confidence, x1, y1, x2, y2 = det
            report += f"  Object: {obj_class} (Confidence: {confidence:.2f})\n"
            report += f"  Coords: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n"
            report += f"  Image: {image_path}\n"
            report += f"  Time: {timestamp}\n\n"
        
        print(report)
        
        # Save report to file
        report_path = os.path.join(os.path.dirname(__file__), "ms_project_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Report saved to {os.path.abspath(report_path)}")
        return report

def main():
    integration = SimpleProjectIntegration()
    
    print("YOLOv5 Microsoft Project Integration\n")
    print("Options:")
    print("1. Start Microsoft Project")
    print("2. Process a Microsoft Project file")
    print("3. Run YOLOv5 detection on current screen")
    print("4. Generate report")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            integration.start_ms_project()
        elif choice == "2":
            file_path = input("Enter the path to an MS Project file (or press Enter for sample data): ")
            if not file_path:
                file_path = "sample_project.mpp"  # Simulate a file path
            integration.process_ms_project_file(file_path)
        elif choice == "3":
            integration.detect_from_screenshot()
        elif choice == "4":
            integration.generate_report()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
