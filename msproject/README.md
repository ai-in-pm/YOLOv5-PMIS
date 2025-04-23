# Microsoft Project Integration for YOLOv5

This module provides integration between YOLOv5 object detection and Microsoft Project, enabling automated detection and extraction of project elements from Microsoft Project interfaces.

## Features

- Automatic detection of Microsoft Project elements (tasks, milestones, resources, etc.)
- Extraction of structured project data from visual interfaces
- Generation of project reports and analysis
- Support for critical path analysis
- Resource allocation analysis

## Requirements

- Python 3.8+
- PyTorch 1.7+
- YOLOv5
- OpenCV
- PyAutoGUI
- pywin32 (for Windows integration)

## Installation

1. Ensure you have the YOLOv5 repository cloned and set up
2. Install additional dependencies:
   ```
   pip install pyautogui pywin32
   ```

## Usage

### Command Line Interface

The easiest way to use the integration is through the provided command-line interface:

```bash
python run_msproject_detection.py --start --report
```

Or use the batch file on Windows:

```
run_msproject_detection.bat --start --report
```

### Command Line Options

- `--model MODEL`: Path to the YOLOv5 model file (default: "msproject_model.pt")
- `--conf CONF`: Detection confidence threshold (default: 0.25)
- `--iou IOU`: IoU threshold for NMS (default: 0.45)
- `--start`: Start Microsoft Project before detection
- `--report`: Generate and display project report
- `--output OUTPUT`: Path to save detection results (JSON format)
- `--debug`: Enable debug logging

### Python API

You can also use the Python API directly in your code:

```python
from msproject.msproject_detector import MSProjectDetector

# Create detector
detector = MSProjectDetector()

# Start Microsoft Project (optional)
detector.start_application("msproject")

# Detect project elements
results = detector.detect_project_elements()

# Generate report
report = detector.generate_project_report()
print(report)
```

## Model Training

For best results, use a specialized model trained on Microsoft Project interfaces. The default model is trained to detect:

- Tasks
- Milestones
- Summary tasks
- Critical path tasks
- Resources
- Gantt bars
- Dependencies
- Constraints
- Deadlines
- Baselines

If you don't have a specialized model, the system will fall back to using the standard YOLOv5s model, but detection accuracy may be reduced.

## Troubleshooting

- **Microsoft Project not found**: Ensure Microsoft Project is installed and the path is correct
- **Detection fails**: Try adjusting the confidence threshold with `--conf`
- **Window focus issues**: Make sure Microsoft Project is visible and not minimized
- **Import errors**: Check that all dependencies are installed correctly

## License

This project is licensed under the same license as the parent YOLOv5 project.
