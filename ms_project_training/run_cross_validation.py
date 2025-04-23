import os
import sys
from pathlib import Path

# Add YOLOv5 directory to path
yolov5_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolov5-master'))
sys.path.append(str(yolov5_path))

from .cross_validator import MSProjectCrossValidator

def run_cross_validation(project_path: str, model_path: str, db_path: str, output_dir: str):
    """
    Run the cross-validation process for Microsoft Project elements.
    
    Args:
        project_path: Path to the Microsoft Project file (.mpp)
        model_path: Path to the trained YOLOv5 model
        db_path: Path to the detection database
        output_dir: Directory to save validation reports
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize validator
    validator = MSProjectCrossValidator(
        project_path=project_path,
        model_path=model_path,
        db_path=db_path
    )
    
    print("Starting cross-validation process...")
    print("===================================")
    
    try:
        # Run validation
        results = validator.validate_project_elements()
        
        # Generate and save report
        report_path = os.path.join(output_dir, "validation_report.json")
        validator.save_validation_report(results, report_path)
        
        # Print summary
        print("\nValidation Results Summary:")
        print("-------------------------")
        
        # Print task validation summary
        task_results = results['tasks']
        print(f"\nTasks:")
        print(f"Total: {len(task_results['matches']) + len(task_results['differences']) + len(task_results['missing'])}")
        print(f"Matches: {len(task_results['matches'])}")
        print(f"Differences: {len(task_results['differences'])}")
        print(f"Missing: {len(task_results['missing'])}")
        
        # Print resource validation summary
        resource_results = results['resources']
        print(f"\nResources:")
        print(f"Total: {len(resource_results['matches']) + len(resource_results['differences']) + len(resource_results['missing'])}")
        print(f"Matches: {len(resource_results['matches'])}")
        print(f"Differences: {len(resource_results['differences'])}")
        print(f"Missing: {len(resource_results['missing'])}")
        
        # Print assignment validation summary
        assignment_results = results['assignments']
        print(f"\nAssignments:")
        print(f"Total: {len(assignment_results['matches']) + len(assignment_results['differences']) + len(assignment_results['missing'])}")
        print(f"Matches: {len(assignment_results['matches'])}")
        print(f"Differences: {len(assignment_results['differences'])}")
        print(f"Missing: {len(assignment_results['missing'])}")
        
        print(f"\nValidation report saved to: {report_path}")
        print("\nCross-validation process completed successfully!")
        
    except Exception as e:
        print(f"\nError during cross-validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    project_path = "path/to/your/project.mpp"
    model_path = "path/to/your/model.pt"
    db_path = "path/to/your/detections.db"
    output_dir = "path/to/output"
    
    run_cross_validation(project_path, model_path, db_path, output_dir)
