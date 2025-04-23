import os
import sys
import win32com.client
import pythoncom
import datetime
import json
import sqlite3
import logging
from pathlib import Path

# Add root path for shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectCOMAPI:
    """Microsoft Project COM API integration for direct data access"""
    
    def __init__(self, db_path=None):
        self.project_app = None
        self.active_project = None
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), '..', 'database', 'ms_project_data.db'
        )
        
        # Set up logging
        self.logger = logging.getLogger('MSProjectCOMAPI')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing project data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create API projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                file_path TEXT,
                extraction_date TEXT,
                task_count INTEGER,
                resource_count INTEGER
            )
        ''')
        
        # Create API tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                task_id INTEGER,
                name TEXT,
                start_date TEXT,
                finish_date TEXT,
                duration TEXT,
                percent_complete REAL,
                milestone INTEGER,
                summary INTEGER,
                critical INTEGER,
                resource_names TEXT,
                predecessors TEXT,
                outline_level INTEGER,
                wbs TEXT,
                notes TEXT,
                FOREIGN KEY (project_id) REFERENCES api_projects (id)
            )
        ''')
        
        # Create API resources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                resource_id INTEGER,
                name TEXT,
                standard_rate REAL,
                overtime_rate REAL,
                cost REAL,
                work TEXT,
                type INTEGER,
                email TEXT,
                FOREIGN KEY (project_id) REFERENCES api_projects (id)
            )
        ''')
        
        # Create API assignments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                task_id INTEGER,
                resource_id INTEGER,
                work TEXT,
                percent_work_complete REAL,
                cost REAL,
                FOREIGN KEY (project_id) REFERENCES api_projects (id),
                FOREIGN KEY (task_id) REFERENCES api_tasks (id),
                FOREIGN KEY (resource_id) REFERENCES api_resources (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def connect(self):
        """Connect to Microsoft Project via COM"""
        try:
            # Initialize COM in the current thread
            pythoncom.CoInitialize()
            
            # Create MS Project application object
            self.project_app = win32com.client.Dispatch("MSProject.Application")
            self.project_app.Visible = True
            
            self.logger.info("Connected to Microsoft Project")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Microsoft Project: {e}")
            return False
    
    def open_project(self, file_path=None):
        """Open a project file or use the active project"""
        if not self.project_app:
            if not self.connect():
                return False
        
        try:
            if file_path and os.path.exists(file_path):
                self.logger.info(f"Opening project file: {file_path}")
                self.active_project = self.project_app.FileOpen(file_path)
            else:
                # Try to get the active project
                if self.project_app.ActiveProject:
                    self.active_project = self.project_app.ActiveProject
                    self.logger.info(f"Using active project: {self.active_project.Name}")
                else:
                    self.logger.warning("No active project found.")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error opening project: {e}")
            return False
    
    def get_project_info(self):
        """Get basic information about the active project"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return None
        
        try:
            info = {
                'name': self.active_project.Name,
                'file_path': self.active_project.FullName,
                'start_date': self._format_date(self.active_project.ProjectStart),
                'finish_date': self._format_date(self.active_project.ProjectFinish),
                'current_date': self._format_date(self.active_project.CurrentDate),
                'status_date': self._format_date(self.active_project.StatusDate),
                'task_count': self.active_project.Tasks.Count,
                'resource_count': self.active_project.Resources.Count
            }
            
            self.logger.info(f"Project info retrieved for {info['name']}")
            return info
        except Exception as e:
            self.logger.error(f"Error getting project info: {e}")
            return None
    
    def get_tasks(self):
        """Get all tasks in the project"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return []
        
        tasks = []
        try:
            task_count = self.active_project.Tasks.Count
            
            for i in range(1, task_count + 1):
                task = self.active_project.Tasks(i)
                
                # Skip empty tasks
                if not task.Name:
                    continue
                
                # Get predecessor info
                predecessors = ""
                try:
                    pred_count = task.PredecessorTasks.Count
                    pred_list = []
                    for j in range(1, pred_count + 1):
                        pred_task = task.PredecessorTasks(j)
                        pred_list.append(f"{pred_task.ID}")
                    predecessors = ",".join(pred_list)
                except:
                    pass
                
                # Get resource names
                resource_names = ""
                try:
                    res_count = task.ResourceAssignments.Count
                    res_list = []
                    for j in range(1, res_count + 1):
                        assignment = task.ResourceAssignments(j)
                        res = assignment.Resource
                        if res and res.Name:
                            res_list.append(res.Name)
                    resource_names = ",".join(res_list)
                except:
                    pass
                
                # Create task dictionary
                task_data = {
                    'id': task.ID,
                    'name': task.Name,
                    'start_date': self._format_date(task.Start),
                    'finish_date': self._format_date(task.Finish),
                    'duration': task.Duration/60/8 if task.Duration else 0,  # Convert minutes to days
                    'duration_text': task.DurationText,
                    'percent_complete': task.PercentComplete,
                    'milestone': 1 if task.Milestone else 0,
                    'summary': 1 if task.Summary else 0,
                    'critical': 1 if task.Critical else 0,
                    'resource_names': resource_names,
                    'predecessors': predecessors,
                    'outline_level': task.OutlineLevel,
                    'wbs': task.WBS if hasattr(task, 'WBS') else "",
                    'notes': task.Notes if task.Notes else ""
                }
                
                tasks.append(task_data)
            
            self.logger.info(f"Retrieved {len(tasks)} tasks")
            return tasks
        except Exception as e:
            self.logger.error(f"Error getting tasks: {e}")
            return []
    
    def get_resources(self):
        """Get all resources in the project"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return []
        
        resources = []
        try:
            resource_count = self.active_project.Resources.Count
            
            for i in range(1, resource_count + 1):
                res = self.active_project.Resources(i)
                
                # Skip empty resources
                if not res.Name:
                    continue
                
                resource_data = {
                    'id': res.ID,
                    'name': res.Name,
                    'standard_rate': res.StandardRate,
                    'overtime_rate': res.OvertimeRate,
                    'cost': res.Cost,
                    'work': res.Work/60 if res.Work else 0,  # Convert minutes to hours
                    'work_text': res.WorkText,
                    'type': res.Type,  # 0=Work, 1=Material, 2=Cost
                    'email': res.EmailAddress if hasattr(res, 'EmailAddress') else ""
                }
                
                resources.append(resource_data)
            
            self.logger.info(f"Retrieved {len(resources)} resources")
            return resources
        except Exception as e:
            self.logger.error(f"Error getting resources: {e}")
            return []
    
    def get_assignments(self):
        """Get all resource assignments in the project"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return []
        
        assignments = []
        try:
            for task in self.active_project.Tasks:
                # Skip empty tasks
                if not task.Name:
                    continue
                
                for i in range(1, task.ResourceAssignments.Count + 1):
                    assignment = task.ResourceAssignments(i)
                    res = assignment.Resource
                    
                    # Skip if no resource
                    if not res or not res.Name:
                        continue
                    
                    assignment_data = {
                        'task_id': task.ID,
                        'task_name': task.Name,
                        'resource_id': res.ID,
                        'resource_name': res.Name,
                        'work': assignment.Work/60 if assignment.Work else 0,  # Convert minutes to hours
                        'work_text': assignment.WorkText,
                        'percent_work_complete': assignment.PercentWorkComplete,
                        'cost': assignment.Cost
                    }
                    
                    assignments.append(assignment_data)
            
            self.logger.info(f"Retrieved {len(assignments)} assignments")
            return assignments
        except Exception as e:
            self.logger.error(f"Error getting assignments: {e}")
            return []
    
    def store_project_data(self):
        """Store all project data in the SQLite database"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return False
        
        try:
            # Get all project data
            project_info = self.get_project_info()
            tasks = self.get_tasks()
            resources = self.get_resources()
            assignments = self.get_assignments()
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store project info
            cursor.execute('''
                INSERT INTO api_projects (name, file_path, extraction_date, task_count, resource_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                project_info['name'],
                project_info['file_path'],
                datetime.datetime.now().isoformat(),
                project_info['task_count'],
                project_info['resource_count']
            ))
            
            project_id = cursor.lastrowid
            
            # Store tasks
            for task in tasks:
                cursor.execute('''
                    INSERT INTO api_tasks (
                        project_id, task_id, name, start_date, finish_date, duration,
                        percent_complete, milestone, summary, critical, resource_names,
                        predecessors, outline_level, wbs, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id,
                    task['id'],
                    task['name'],
                    task['start_date'],
                    task['finish_date'],
                    task['duration_text'],
                    task['percent_complete'],
                    task['milestone'],
                    task['summary'],
                    task['critical'],
                    task['resource_names'],
                    task['predecessors'],
                    task['outline_level'],
                    task['wbs'],
                    task['notes']
                ))
            
            # Store resources
            for res in resources:
                cursor.execute('''
                    INSERT INTO api_resources (
                        project_id, resource_id, name, standard_rate, overtime_rate,
                        cost, work, type, email
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id,
                    res['id'],
                    res['name'],
                    res['standard_rate'],
                    res['overtime_rate'],
                    res['cost'],
                    res['work_text'],
                    res['type'],
                    res['email']
                ))
            
            # Store assignments
            for assign in assignments:
                cursor.execute('''
                    INSERT INTO api_assignments (
                        project_id, task_id, resource_id, work, percent_work_complete, cost
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    project_id,
                    assign['task_id'],
                    assign['resource_id'],
                    assign['work_text'],
                    assign['percent_work_complete'],
                    assign['cost']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored complete project data for {project_info['name']} in database")
            return project_id
        except Exception as e:
            self.logger.error(f"Error storing project data: {e}")
            return False
    
    def export_to_json(self, output_path=None):
        """Export project data to JSON format"""
        if not self.active_project:
            self.logger.warning("No active project. Call open_project() first.")
            return None
        
        try:
            # Get all project data
            project_info = self.get_project_info()
            tasks = self.get_tasks()
            resources = self.get_resources()
            assignments = self.get_assignments()
            
            # Create project data structure
            project_data = {
                'project': project_info,
                'tasks': tasks,
                'resources': resources,
                'assignments': assignments
            }
            
            # Determine output path
            if not output_path:
                output_dir = os.path.join(os.path.dirname(__file__), 'output', 'api_exports')
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{project_info['name'].replace(' ', '_')}_{timestamp}.json"
                output_path = os.path.join(output_dir, filename)
            
            # Write to JSON file
            with open(output_path, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported project data to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            return None
    
    def close(self):
        """Close the connection to Microsoft Project"""
        try:
            if self.project_app:
                # Close without saving
                self.project_app.FileCloseAll(0)  # 0 = don't save changes
                self.project_app.Quit()
                self.project_app = None
                self.active_project = None
                
                # Uninitialize COM
                pythoncom.CoUninitialize()
                
                self.logger.info("Closed connection to Microsoft Project")
                return True
        except Exception as e:
            self.logger.error(f"Error closing Microsoft Project: {e}")
        return False
    
    def _format_date(self, date_value):
        """Format a date value as ISO string"""
        if not date_value:
            return ""
        
        try:
            # Convert to Python datetime if it's a COM date
            if isinstance(date_value, float):
                date_value = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=date_value)
            
            return date_value.isoformat()
        except:
            return str(date_value)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Microsoft Project COM API integration")
    parser.add_argument('--file', type=str, help='Path to Microsoft Project file')
    parser.add_argument('--export', action='store_true', help='Export project data to JSON')
    parser.add_argument('--store', action='store_true', help='Store project data in database')
    parser.add_argument('--output', type=str, help='Output path for JSON export')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    com_api = MSProjectCOMAPI()
    
    try:
        # Connect to MS Project
        if com_api.connect():
            # Open project file or use active project
            if com_api.open_project(args.file):
                # Get basic project info
                project_info = com_api.get_project_info()
                if project_info:
                    print(f"\nProject: {project_info['name']}")
                    print(f"Tasks: {project_info['task_count']}")
                    print(f"Resources: {project_info['resource_count']}")
                    print(f"Start Date: {project_info['start_date']}")
                    print(f"Finish Date: {project_info['finish_date']}")
                
                # Export to JSON if requested
                if args.export:
                    json_path = com_api.export_to_json(args.output)
                    if json_path:
                        print(f"\nExported project data to: {json_path}")
                
                # Store in database if requested
                if args.store:
                    project_id = com_api.store_project_data()
                    if project_id:
                        print(f"\nStored project data in database with ID: {project_id}")
    finally:
        # Always close the connection
        com_api.close()

if __name__ == "__main__":
    main()
