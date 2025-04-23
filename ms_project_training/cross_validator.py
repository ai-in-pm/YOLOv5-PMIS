import os
import sys
import sqlite3
import json
from typing import Dict, List, Optional
from pathlib import Path

# Add YOLOv5 directory to path
yolov5_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolov5-master'))
sys.path.append(str(yolov5_path))

from .com_api import MSProjectAPI
from .detector import MSProjectDetector
from .detection_database import DetectionDatabase

class MSProjectCrossValidator:
    def __init__(self, project_path: str, model_path: str, db_path: str):
        """Initialize the cross-validator with project path and model path."""
        self.project_path = project_path
        self.model_path = model_path
        self.db_path = db_path
        
        # Initialize components
        self.api = MSProjectAPI()
        self.detector = MSProjectDetector(model_path=model_path)
        self.db = DetectionDatabase(db_path)
        
    def validate_project_elements(self) -> Dict[str, Dict]:
        """
        Validate project elements by comparing visual detection with API data.
        Returns a dictionary containing validation results.
        """
        # Get API data
        api_data = self._get_api_data()
        
        # Get visual detection results
        detection_results = self._get_detection_results()
        
        # Compare and validate
        validation_results = {
            'tasks': self._validate_tasks(api_data['tasks'], detection_results['tasks']),
            'resources': self._validate_resources(api_data['resources'], detection_results['resources']),
            'assignments': self._validate_assignments(api_data['assignments'], detection_results['assignments'])
        }
        
        return validation_results
    
    def _get_api_data(self) -> Dict[str, List]:
        """Retrieve project data using COM API."""
        try:
            self.api.open_project(self.project_path)
            return {
                'tasks': self.api.get_tasks(),
                'resources': self.api.get_resources(),
                'assignments': self.api.get_assignments()
            }
        except Exception as e:
            print(f"Error retrieving API data: {str(e)}")
            return {'tasks': [], 'resources': [], 'assignments': []}
    
    def _get_detection_results(self) -> Dict[str, List]:
        """Retrieve detection results from the database."""
        try:
            return self.db.get_latest_detection_results()
        except Exception as e:
            print(f"Error retrieving detection results: {str(e)}")
            return {'tasks': [], 'resources': [], 'assignments': []}
    
    def _validate_tasks(self, api_tasks: List[Dict], detected_tasks: List[Dict]) -> Dict[str, List]:
        """Validate task data between API and detection results."""
        validation = {
            'matches': [],
            'differences': [],
            'missing': []
        }
        
        # Create lookup dictionaries
        api_task_dict = {task['id']: task for task in api_tasks}
        detected_task_dict = {task['id']: task for task in detected_tasks}
        
        # Check for matches and differences
        for task_id, api_task in api_task_dict.items():
            if task_id in detected_task_dict:
                detected_task = detected_task_dict[task_id]
                if self._compare_task_data(api_task, detected_task):
                    validation['matches'].append(task_id)
                else:
                    validation['differences'].append({
                        'id': task_id,
                        'api_data': api_task,
                        'detected_data': detected_task
                    })
            else:
                validation['missing'].append(task_id)
        
        return validation
    
    def _validate_resources(self, api_resources: List[Dict], detected_resources: List[Dict]) -> Dict[str, List]:
        """Validate resource data between API and detection results."""
        validation = {
            'matches': [],
            'differences': [],
            'missing': []
        }
        
        # Create lookup dictionaries
        api_resource_dict = {res['id']: res for res in api_resources}
        detected_resource_dict = {res['id']: res for res in detected_resources}
        
        # Check for matches and differences
        for res_id, api_res in api_resource_dict.items():
            if res_id in detected_resource_dict:
                detected_res = detected_resource_dict[res_id]
                if self._compare_resource_data(api_res, detected_res):
                    validation['matches'].append(res_id)
                else:
                    validation['differences'].append({
                        'id': res_id,
                        'api_data': api_res,
                        'detected_data': detected_res
                    })
            else:
                validation['missing'].append(res_id)
        
        return validation
    
    def _validate_assignments(self, api_assignments: List[Dict], detected_assignments: List[Dict]) -> Dict[str, List]:
        """Validate assignment data between API and detection results."""
        validation = {
            'matches': [],
            'differences': [],
            'missing': []
        }
        
        # Create lookup dictionaries
        api_assign_dict = {(a['task_id'], a['resource_id']): a for a in api_assignments}
        detected_assign_dict = {(a['task_id'], a['resource_id']): a for a in detected_assignments}
        
        # Check for matches and differences
        for key, api_assign in api_assign_dict.items():
            if key in detected_assign_dict:
                detected_assign = detected_assign_dict[key]
                if self._compare_assignment_data(api_assign, detected_assign):
                    validation['matches'].append(key)
                else:
                    validation['differences'].append({
                        'key': key,
                        'api_data': api_assign,
                        'detected_data': detected_assign
                    })
            else:
                validation['missing'].append(key)
        
        return validation
    
    def _compare_task_data(self, api_task: Dict, detected_task: Dict) -> bool:
        """Compare task data fields between API and detection results."""
        # Critical fields to compare
        critical_fields = ['name', 'start_date', 'end_date', 'duration', 'status']
        
        for field in critical_fields:
            if api_task.get(field) != detected_task.get(field):
                return False
        return True
    
    def _compare_resource_data(self, api_res: Dict, detected_res: Dict) -> bool:
        """Compare resource data fields between API and detection results."""
        # Critical fields to compare
        critical_fields = ['name', 'type', 'cost', 'work']
        
        for field in critical_fields:
            if api_res.get(field) != detected_res.get(field):
                return False
        return True
    
    def _compare_assignment_data(self, api_assign: Dict, detected_assign: Dict) -> bool:
        """Compare assignment data fields between API and detection results."""
        # Critical fields to compare
        critical_fields = ['units', 'work', 'cost']
        
        for field in critical_fields:
            if api_assign.get(field) != detected_assign.get(field):
                return False
        return True
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate a detailed validation report in JSON format."""
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'project_path': self.project_path,
            'validation_results': validation_results,
            'summary': {
                'tasks': {
                    'total': len(validation_results['tasks']['matches']) + 
                            len(validation_results['tasks']['differences']) + 
                            len(validation_results['tasks']['missing']),
                    'matches': len(validation_results['tasks']['matches']),
                    'differences': len(validation_results['tasks']['differences']),
                    'missing': len(validation_results['tasks']['missing'])
                },
                'resources': {
                    'total': len(validation_results['resources']['matches']) + 
                            len(validation_results['resources']['differences']) + 
                            len(validation_results['resources']['missing']),
                    'matches': len(validation_results['resources']['matches']),
                    'differences': len(validation_results['resources']['differences']),
                    'missing': len(validation_results['resources']['missing'])
                },
                'assignments': {
                    'total': len(validation_results['assignments']['matches']) + 
                            len(validation_results['assignments']['differences']) + 
                            len(validation_results['assignments']['missing']),
                    'matches': len(validation_results['assignments']['matches']),
                    'differences': len(validation_results['assignments']['differences']),
                    'missing': len(validation_results['assignments']['missing'])
                }
            }
        }
        
        return json.dumps(report, indent=4)
    
    def save_validation_report(self, validation_results: Dict, output_path: str) -> None:
        """Save the validation report to a file."""
        report = self.generate_validation_report(validation_results)
        with open(output_path, 'w') as f:
            f.write(report)

if __name__ == "__main__":
    # Example usage
    project_path = "path/to/your/project.mpp"
    model_path = "path/to/your/model.pt"
    db_path = "path/to/your/detections.db"
    
    validator = MSProjectCrossValidator(project_path, model_path, db_path)
    results = validator.validate_project_elements()
    validator.save_validation_report(results, "validation_report.json")
