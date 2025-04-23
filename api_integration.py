import os
import sys
import win32com.client
import sqlite3
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add YOLOv5 directory to path
yolov5_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolov5-master')
sys.path.append(yolov5_path)

class PMApplicationAPI:
    """Base class for Project Management application API integration"""
    
    def __init__(self, app_name):
        self.app_name = app_name
        self.app = None
        self.connected = False
        self.project = None
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'pm_api_data.db')
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for storing API data"""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create PM projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT,
                    project_name TEXT,
                    file_path TEXT,
                    start_date TEXT,
                    finish_date TEXT,
                    task_count INTEGER,
                    resource_count INTEGER,
                    last_accessed TIMESTAMP
                )
            ''')
            
            # Create tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    task_id TEXT,
                    task_name TEXT,
                    start_date TEXT,
                    finish_date TEXT,
                    duration TEXT,
                    percent_complete REAL,
                    is_critical BOOLEAN,
                    is_milestone BOOLEAN,
                    wbs TEXT,
                    outline_level INTEGER,
                    predecessor_ids TEXT,
                    resource_names TEXT,
                    FOREIGN KEY (project_id) REFERENCES pm_projects (id)
                )
            ''')
            
            # Create resources table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    resource_id TEXT,
                    resource_name TEXT,
                    resource_type TEXT,
                    standard_rate REAL,
                    overtime_rate REAL,
                    cost_per_use REAL,
                    max_units REAL,
                    FOREIGN KEY (project_id) REFERENCES pm_projects (id)
                )
            ''')
            
            # Create assignments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pm_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    task_id TEXT,
                    resource_id TEXT,
                    units REAL,
                    work TEXT,
                    cost REAL,
                    FOREIGN KEY (project_id) REFERENCES pm_projects (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"Database initialized: {self.db_path}")
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def connect(self):
        """Connect to the application via COM"""
        raise NotImplementedError("Subclasses must implement connect()")
    
    def open_project(self, file_path):
        """Open a project file"""
        raise NotImplementedError("Subclasses must implement open_project()")
    
    def close_project(self):
        """Close the current project"""
        raise NotImplementedError("Subclasses must implement close_project()")
    
    def disconnect(self):
        """Disconnect from the application"""
        raise NotImplementedError("Subclasses must implement disconnect()")
    
    def get_project_info(self):
        """Get basic project information"""
        raise NotImplementedError("Subclasses must implement get_project_info()")
    
    def get_tasks(self):
        """Get all tasks in the project"""
        raise NotImplementedError("Subclasses must implement get_tasks()")
    
    def get_resources(self):
        """Get all resources in the project"""
        raise NotImplementedError("Subclasses must implement get_resources()")
    
    def get_assignments(self):
        """Get all resource assignments in the project"""
        raise NotImplementedError("Subclasses must implement get_assignments()")
    
    def get_critical_path(self):
        """Get the critical path of the project"""
        raise NotImplementedError("Subclasses must implement get_critical_path()")
    
    def export_to_yolo_dataset(self, output_dir):
        """Export the project data as a YOLO dataset for training"""
        raise NotImplementedError("Subclasses must implement export_to_yolo_dataset()")

class MSProjectAPI(PMApplicationAPI):
    """Microsoft Project API integration through COM"""
    
    def __init__(self):
        super().__init__("Microsoft Project")
    
    def connect(self):
        """Connect to MS Project via COM"""
        try:
            self.app = win32com.client.Dispatch("MSProject.Application")
            self.app.Visible = True
            self.connected = True
            print("Connected to Microsoft Project")
            return True
        except Exception as e:
            print(f"Error connecting to Microsoft Project: {e}")
            return False
    
    def open_project(self, file_path):
        """Open a project file in MS Project"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            self.project = self.app.Projects.Open(file_path)
            print(f"Opened project: {file_path}")
            
            # Store project info in database
            self._store_project_info(file_path)
            
            return True
        except Exception as e:
            print(f"Error opening project: {e}")
            return False
    
    def close_project(self):
        """Close the current project"""
        if self.project:
            try:
                self.app.FileClose(0)  # 0 = don't save changes
                self.project = None
                print("Project closed")
                return True
            except Exception as e:
                print(f"Error closing project: {e}")
                return False
        return True
    
    def disconnect(self):
        """Disconnect from MS Project"""
        if self.connected:
            try:
                if self.project:
                    self.close_project()
                self.app.Quit()
                self.app = None
                self.connected = False
                print("Disconnected from Microsoft Project")
                return True
            except Exception as e:
                print(f"Error disconnecting: {e}")
                return False
        return True
    
    def get_project_info(self):
        """Get basic project information"""
        if not self.project:
            print("No project open")
            return None
        
        try:
            info = {
                'name': self.project.Name,
                'start_date': self.project.ProjectStart,
                'finish_date': self.project.ProjectFinish,
                'task_count': self.project.Tasks.Count,
                'resource_count': self.project.Resources.Count,
                'status_date': self.project.StatusDate,
                'current_date': self.project.CurrentDate,
                'calendar_name': self.project.Calendar.Name
            }
            return info
        except Exception as e:
            print(f"Error getting project info: {e}")
            return None
    
    def get_tasks(self):
        """Get all tasks in the project"""
        if not self.project:
            print("No project open")
            return None
        
        tasks = []
        try:
            for i in range(1, self.project.Tasks.Count + 1):
                task = self.project.Tasks(i)
                
                # Skip summary tasks if desired
                # if task.Summary:
                #     continue
                
                predecessors = []
                for j in range(1, task.PredecessorTasks.Count + 1):
                    pred_task = task.PredecessorTasks(j)
                    predecessors.append(str(pred_task.ID))
                
                resource_names = []
                for j in range(1, task.Resources.Count + 1):
                    resource = task.Resources(j)
                    resource_names.append(resource.Name)
                
                task_data = {
                    'id': task.ID,
                    'name': task.Name,
                    'start': task.Start,
                    'finish': task.Finish,
                    'duration': task.Duration,
                    'percent_complete': task.PercentComplete,
                    'critical': task.Critical,
                    'milestone': task.Milestone,
                    'summary': task.Summary,
                    'wbs': task.WBS,
                    'outline_level': task.OutlineLevel,
                    'predecessors': ', '.join(predecessors),
                    'resource_names': ', '.join(resource_names)
                }
                tasks.append(task_data)
            
            # Store tasks in database
            self._store_tasks(tasks)
            
            return tasks
        except Exception as e:
            print(f"Error getting tasks: {e}")
            return None
    
    def get_resources(self):
        """Get all resources in the project"""
        if not self.project:
            print("No project open")
            return None
        
        resources = []
        try:
            for i in range(1, self.project.Resources.Count + 1):
                resource = self.project.Resources(i)
                
                resource_data = {
                    'id': resource.ID,
                    'name': resource.Name,
                    'type': 'Work' if resource.Type == 1 else 'Material' if resource.Type == 2 else 'Cost',
                    'standard_rate': resource.StandardRate,
                    'overtime_rate': resource.OvertimeRate,
                    'cost_per_use': resource.CostPerUse,
                    'max_units': resource.MaxUnits
                }
                resources.append(resource_data)
            
            # Store resources in database
            self._store_resources(resources)
            
            return resources
        except Exception as e:
            print(f"Error getting resources: {e}")
            return None
    
    def get_assignments(self):
        """Get all resource assignments in the project"""
        if not self.project:
            print("No project open")
            return None
        
        assignments = []
        try:
            for i in range(1, self.project.Tasks.Count + 1):
                task = self.project.Tasks(i)
                task_id = task.ID
                
                for j in range(1, task.Resources.Count + 1):
                    assignment = task.Resources(j)
                    resource = assignment.Resource
                    
                    assignment_data = {
                        'task_id': task_id,
                        'task_name': task.Name,
                        'resource_id': resource.ID,
                        'resource_name': resource.Name,
                        'units': assignment.Units,
                        'work': assignment.Work,
                        'cost': assignment.Cost
                    }
                    assignments.append(assignment_data)
            
            # Store assignments in database
            self._store_assignments(assignments)
            
            return assignments
        except Exception as e:
            print(f"Error getting assignments: {e}")
            return None
    
    def get_critical_path(self):
        """Get the critical path of the project"""
        if not self.project:
            print("No project open")
            return None
        
        critical_tasks = []
        try:
            for i in range(1, self.project.Tasks.Count + 1):
                task = self.project.Tasks(i)
                
                if task.Critical:
                    task_data = {
                        'id': task.ID,
                        'name': task.Name,
                        'start': task.Start,
                        'finish': task.Finish,
                        'duration': task.Duration,
                        'slack': task.TotalSlack,
                        'predecessors': [pred.ID for pred in task.PredecessorTasks if pred.Critical]
                    }
                    critical_tasks.append(task_data)
            
            # Sort by start date
            critical_tasks.sort(key=lambda t: t['start'])
            
            return critical_tasks
        except Exception as e:
            print(f"Error getting critical path: {e}")
            return None
    
    def _store_project_info(self, file_path):
        """Store project info in the database"""
        info = self.get_project_info()
        if not info:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if project already exists
            cursor.execute("SELECT id FROM pm_projects WHERE file_path = ?", (file_path,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing project
                project_id = existing[0]
                cursor.execute("""
                    UPDATE pm_projects 
                    SET project_name = ?, start_date = ?, finish_date = ?, 
                        task_count = ?, resource_count = ?, last_accessed = ?
                    WHERE id = ?
                """, (
                    info['name'], str(info['start_date']), str(info['finish_date']),
                    info['task_count'], info['resource_count'], datetime.now().isoformat(),
                    project_id
                ))
            else:
                # Insert new project
                cursor.execute("""
                    INSERT INTO pm_projects 
                    (app_name, project_name, file_path, start_date, finish_date, 
                     task_count, resource_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.app_name, info['name'], file_path, 
                    str(info['start_date']), str(info['finish_date']),
                    info['task_count'], info['resource_count'],
                    datetime.now().isoformat()
                ))
                project_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            return project_id
        except Exception as e:
            print(f"Error storing project info: {e}")
            return None
    
    def _store_tasks(self, tasks):
        """Store tasks in the database"""
        if not tasks:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get project ID
            cursor.execute("SELECT id FROM pm_projects WHERE project_name = ?", (self.project.Name,))
            project_id = cursor.fetchone()[0]
            
            # Clear existing tasks for this project
            cursor.execute("DELETE FROM pm_tasks WHERE project_id = ?", (project_id,))
            
            # Insert tasks
            for task in tasks:
                cursor.execute("""
                    INSERT INTO pm_tasks 
                    (project_id, task_id, task_name, start_date, finish_date, 
                     duration, percent_complete, is_critical, is_milestone, 
                     wbs, outline_level, predecessor_ids, resource_names)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id, str(task['id']), task['name'], 
                    str(task['start']), str(task['finish']), str(task['duration']),
                    task['percent_complete'], task['critical'], task['milestone'],
                    task['wbs'], task['outline_level'], task['predecessors'], task['resource_names']
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error storing tasks: {e}")
    
    def _store_resources(self, resources):
        """Store resources in the database"""
        if not resources:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get project ID
            cursor.execute("SELECT id FROM pm_projects WHERE project_name = ?", (self.project.Name,))
            project_id = cursor.fetchone()[0]
            
            # Clear existing resources for this project
            cursor.execute("DELETE FROM pm_resources WHERE project_id = ?", (project_id,))
            
            # Insert resources
            for resource in resources:
                cursor.execute("""
                    INSERT INTO pm_resources 
                    (project_id, resource_id, resource_name, resource_type, 
                     standard_rate, overtime_rate, cost_per_use, max_units)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id, str(resource['id']), resource['name'], resource['type'],
                    resource['standard_rate'], resource['overtime_rate'], 
                    resource['cost_per_use'], resource['max_units']
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error storing resources: {e}")
    
    def _store_assignments(self, assignments):
        """Store assignments in the database"""
        if not assignments:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get project ID
            cursor.execute("SELECT id FROM pm_projects WHERE project_name = ?", (self.project.Name,))
            project_id = cursor.fetchone()[0]
            
            # Clear existing assignments for this project
            cursor.execute("DELETE FROM pm_assignments WHERE project_id = ?", (project_id,))
            
            # Insert assignments
            for assignment in assignments:
                cursor.execute("""
                    INSERT INTO pm_assignments 
                    (project_id, task_id, resource_id, units, work, cost)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    project_id, str(assignment['task_id']), str(assignment['resource_id']),
                    assignment['units'], str(assignment['work']), assignment['cost']
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error storing assignments: {e}")
    
    def export_to_yolo_dataset(self, output_dir):
        """Export the project as a YOLO dataset for training"""
        if not self.project:
            print("No project open")
            return False
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            images_dir = os.path.join(output_dir, 'images')
            labels_dir = os.path.join(output_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Take screenshots of different views
            views = ['Gantt Chart', 'Task Usage', 'Resource Sheet', 'Resource Usage']
            
            for i, view in enumerate(views):
                # Switch to view
                if view == 'Gantt Chart':
                    self.app.ViewApply(1)  # 1 = Gantt Chart
                elif view == 'Task Usage':
                    self.app.ViewApply(4)  # 4 = Task Usage
                elif view == 'Resource Sheet':
                    self.app.ViewApply(3)  # 3 = Resource Sheet
                elif view == 'Resource Usage':
                    self.app.ViewApply(5)  # 5 = Resource Usage
                
                # Wait for view to update
                time.sleep(1)
                
                # Take screenshot
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                
                # Save image
                image_path = os.path.join(images_dir, f"{view.lower().replace(' ', '_')}_{i}.jpg")
                cv2.imwrite(image_path, screenshot)
                
                # Create synthetic labels for key elements
                # This is a simplified approach - real implementation would need actual object detection
                label_path = os.path.join(labels_dir, f"{view.lower().replace(' ', '_')}_{i}.txt")
                with open(label_path, 'w') as f:
                    # Create sample labels in YOLO format: class x_center y_center width height
                    # These are placeholder values
                    if view == 'Gantt Chart':
                        f.write("0 0.25 0.3 0.5 0.1\n")  # Task bars
                        f.write("1 0.1 0.3 0.2 0.1\n")   # Milestones
                    elif view == 'Task Usage':
                        f.write("2 0.3 0.4 0.4 0.2\n")  # Task grid
                    elif view == 'Resource Sheet':
                        f.write("3 0.5 0.3 0.9 0.1\n")  # Resource rows
            
            # Create dataset.yaml
            yaml_path = os.path.join(output_dir, 'dataset.yaml')
            with open(yaml_path, 'w') as f:
                f.write(f"path: {output_dir}\n")
                f.write("train: images\n")
                f.write("val: images\n")
                f.write("\n")
                f.write("nc: 4\n")
                f.write("names: ['task', 'milestone', 'taskgrid', 'resource']\n")
            
            print(f"Dataset exported to {output_dir}")
            return True
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False

class PrimaveraAPI(PMApplicationAPI):
    """Primavera P6 Professional API integration through COM"""
    
    def __init__(self):
        super().__init__("Primavera P6")
    
    def connect(self):
        """Connect to Primavera P6 via COM"""
        try:
            # Note: This is a hypothetical COM object name
            # Actual implementation may require different approach or SDK
            self.app = win32com.client.Dispatch("PrimaveraP6.Application")
            self.app.Visible = True
            self.connected = True
            print("Connected to Primavera P6")
            return True
        except Exception as e:
            print(f"Error connecting to Primavera P6: {e}")
            print("Note: Direct COM automation for Primavera P6 may not be available.")
            print("Consider using Primavera P6 Integration API, ODBC, or SDK.")
            return False

# Example usage
def main():
    # Create MS Project API
    ms_project = MSProjectAPI()
    
    # Connect to MS Project
    if ms_project.connect():
        # Prompt for project file
        file_path = input("Enter MS Project file path: ")
        
        # Open project
        if ms_project.open_project(file_path):
            # Get project info
            project_info = ms_project.get_project_info()
            print("\nProject Information:")
            for key, value in project_info.items():
                print(f"{key}: {value}")
            
            # Get tasks
            tasks = ms_project.get_tasks()
            print(f"\nTasks: {len(tasks)}")
            
            # Get resources
            resources = ms_project.get_resources()
            print(f"\nResources: {len(resources)}")
            
            # Get critical path
            critical_path = ms_project.get_critical_path()
            print(f"\nCritical Path: {len(critical_path)} tasks")
            
            # Close project
            ms_project.close_project()
        
        # Disconnect
        ms_project.disconnect()

if __name__ == "__main__":
    main()
