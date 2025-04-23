import os
import sys
import sqlite3
import json
import datetime
from pathlib import Path

# Add YOLOv5 directory to path
yolov5_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolov5-master'))
sys.path.append(str(yolov5_path))

class DetectionDatabase:
    def __init__(self, db_path):
        """Initialize the detection database."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return True
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def initialize_db(self):
        """Create necessary tables if they don't exist."""
        if self.connect():
            # Create detections table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_name TEXT,
                    source_path TEXT,
                    confidence_threshold REAL,
                    iou_threshold REAL,
                    result_path TEXT
                )
            ''')
            
            # Create detection results table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    image_path TEXT,
                    class_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    x_min REAL,
                    y_min REAL,
                    x_max REAL,
                    y_max REAL,
                    FOREIGN KEY (run_id) REFERENCES detection_runs (id)
                )
            ''')
            
            self.conn.commit()
            self.close()
    
    def record_detection_run(self, model_name, source_path, conf_thres, iou_thres, result_path):
        """Record a detection run in the database."""
        if self.connect():
            timestamp = datetime.datetime.now().isoformat()
            self.cursor.execute('''
                INSERT INTO detection_runs 
                (timestamp, model_name, source_path, confidence_threshold, iou_threshold, result_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, model_name, source_path, conf_thres, iou_thres, result_path))
            run_id = self.cursor.lastrowid
            self.conn.commit()
            self.close()
            return run_id
        return None
    
    def record_detection_results(self, run_id, results):
        """Record detection results in the database."""
        if self.connect():
            for result in results:
                self.cursor.execute('''
                    INSERT INTO detection_results 
                    (run_id, image_path, class_id, class_name, confidence, x_min, y_min, x_max, y_max)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id, 
                    result['image_path'],
                    result['class_id'],
                    result['class_name'],
                    result['confidence'],
                    result['x_min'],
                    result['y_min'],
                    result['x_max'],
                    result['y_max']
                ))
            self.conn.commit()
            self.close()
            return True
        return False
    
    def get_detection_runs(self, limit=10):
        """Get recent detection runs."""
        if self.connect():
            self.cursor.execute('''
                SELECT * FROM detection_runs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            runs = self.cursor.fetchall()
            self.close()
            return runs
        return []
    
    def get_detection_results(self, run_id):
        """Get detection results for a specific run."""
        if self.connect():
            self.cursor.execute('''
                SELECT * FROM detection_results
                WHERE run_id = ?
            ''', (run_id,))
            results = self.cursor.fetchall()
            self.close()
            return results
        return []
    
    def get_statistics(self):
        """Get detection statistics."""
        if self.connect():
            # Get count of runs
            self.cursor.execute('SELECT COUNT(*) FROM detection_runs')
            run_count = self.cursor.fetchone()[0]
            
            # Get count of detections
            self.cursor.execute('SELECT COUNT(*) FROM detection_results')
            detection_count = self.cursor.fetchone()[0]
            
            # Get class distribution
            self.cursor.execute('''
                SELECT class_name, COUNT(*) as count 
                FROM detection_results 
                GROUP BY class_name
                ORDER BY count DESC
            ''')
            class_distribution = self.cursor.fetchall()
            
            # Get average confidence
            self.cursor.execute('SELECT AVG(confidence) FROM detection_results')
            avg_confidence = self.cursor.fetchone()[0]
            
            self.close()
            return {
                'run_count': run_count,
                'detection_count': detection_count,
                'class_distribution': class_distribution,
                'avg_confidence': avg_confidence
            }
        return {}

# Create database instance
if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'yolo_detections.db')
    db = DetectionDatabase(db_path)
    print(f"Database initialized at {db_path}")
