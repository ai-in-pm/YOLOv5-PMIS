@echo off
echo Running Microsoft Project Cross-Validation
echo =========================================

set PROJECT_PATH="c:\CC-WorkingDir\yolov5-master\yolo_projects\integrations\ms_project_training\test_data\test_project.mpp"
set MODEL_PATH="c:\CC-WorkingDir\yolov5-master\yolo_projects\integrations\ms_project_training\test_data\test_model.pt"
set DB_PATH="c:\CC-WorkingDir\yolov5-master\yolo_projects\integrations\ms_project_training\test_data\test_detections.db"
set OUTPUT_DIR="c:\CC-WorkingDir\yolov5-master\yolo_projects\integrations\ms_project_training\test_output"

python run_cross_validation.py %PROJECT_PATH% %MODEL_PATH% %DB_PATH% %OUTPUT_DIR%

echo.
echo Cross-validation completed.
echo =========================================
pause
