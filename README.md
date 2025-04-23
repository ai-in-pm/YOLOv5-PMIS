# YOLOv5 Enhanced Detection System with Project Management Integration

This is an advanced implementation of YOLOv5 with integrated database functionality for systematic tracking, storage, and analysis of object detection results. The system extends the core YOLOv5 capabilities with enhanced features designed for scientific research and industrial applications. It also includes specialized integrations for project management software like Microsoft Project and Primavera P6.

## System Architecture

The enhanced system follows a structured design pattern with the following components:

```
yolo_projects/
├── models/                  # Pre-trained and custom YOLOv5 models
├── datasets/                # Training and validation datasets
├── results/                 # Detection results and output files
├── custom_scripts/          # Enhanced detection functionality
│   ├── detection_database.py   # SQLite database integration
│   └── enhanced_detect.py      # Extended detection with database support
├── database/                # SQLite database storage
│   └── yolo_detections.db      # SQLite database for detection results
├── integrations/            # Project management integrations
│   ├── msproject/           # Microsoft Project integration
│   │   ├── msproject_detector.py
│   │   ├── msproject_realtime.py
│   │   └── ...
│   ├── primavera/           # Primavera P6 integration
│   │   ├── primavera_detector.py
│   │   └── ...
│   ├── ms_project_training/ # Training pipeline for MS Project models
│   ├── database/            # SQLite database for PM detections
│   ├── results/             # PM detection results
│   ├── pm_integration.py    # Core PM integration framework
│   ├── ocr_utils.py         # OCR utilities for text extraction
│   ├── pm_gui.py            # Graphical user interface
│   └── multi_source_detector.py # Multi-source detection module
├── run_yolov5.py            # Python launcher script
└── run_yolov5.bat           # Windows batch launcher
```

## Core Features

1. **Enhanced Object Detection**
   - High-performance neural network inference
   - Support for images, videos, and webcam streams
   - Multiple model sizes (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
   - Configurable confidence thresholds

2. **Database Integration**
   - SQLite embedded database for detection results
   - Structured storage of detection metadata
   - Comprehensive querying capabilities
   - Statistical analysis of detection patterns

3. **Analysis Capabilities**
   - Class distribution analysis
   - Confidence level statistics
   - Performance metrics
   - Detection history tracking

4. **Project Management Integration**
   - AI-Powered Project Element Detection with OCR
   - Microsoft Project integration for Gantt chart analysis
   - Primavera P6 integration for activity network analysis
   - Multi-source detection (webcam, images, videos, screen captures, streams)
   - Real-time detection directly in Microsoft Project
   - OCR text extraction from detected elements

## Usage Instructions

### Option 1: Using the Batch Launcher

The easiest way to use the system is through the `run_yolov5.bat` batch file:

1. Double-click `run_yolov5.bat`
2. Select the desired operation mode from the menu:
   - Standard Detection: Original YOLOv5 without database
   - Enhanced Detection: YOLOv5 with database integration
   - Analyze Previous Detections: View statistics on past detections
3. Follow the prompts to configure your detection settings

### Option 2: Using Python Directly

For more advanced usage, you can run the Python script directly:

```bash
# Standard detection
python run_yolov5.py --mode detect --source path/to/image.jpg --weights yolov5s.pt

# Enhanced detection with database
python run_yolov5.py --mode enhanced --source path/to/image.jpg --weights yolov5s.pt --save-db

# Analyze previous detections
python run_yolov5.py --mode analyze
```

## Database Schema

The SQLite database contains two main tables:

1. **detection_runs**
   - `id`: Unique identifier for each detection run
   - `timestamp`: When the detection was performed
   - `model_name`: YOLOv5 model used (e.g., yolov5s.pt)
   - `source_path`: Path to the source images/videos
   - `confidence_threshold`: Confidence threshold used
   - `iou_threshold`: IoU threshold used for NMS
   - `result_path`: Path to stored detection results

2. **detection_results**
   - `id`: Unique identifier for each detection
   - `run_id`: Foreign key to detection_runs
   - `image_path`: Path to the specific image
   - `class_id`: Class ID of the detected object
   - `class_name`: Class name of the detected object
   - `confidence`: Detection confidence score
   - `x_min`, `y_min`, `x_max`, `y_max`: Bounding box coordinates

## Extended Applications

This system can be extended for various applications:

1. **Visual Surveillance Analysis**
   - Automated counting of objects/people
   - Movement pattern analysis
   - Anomaly detection

2. **Research Data Collection**
   - Systematic storage of detection results
   - Experimental condition tracking
   - Statistical validation of hypotheses

3. **Industrial Quality Control**
   - Defect detection with historical tracking
   - Process monitoring and optimization
   - Statistical process control integration

4. **Project Management Analysis**
   - Automated schedule analysis
   - Resource allocation optimization
   - Critical path identification
   - Project progress tracking
   - Automated data extraction from project files

## Performance Considerations

Performance varies significantly based on the model size and hardware:

| Model | Size | Precision | Speed (CPU) | Speed (GPU) |
|-------|------|-----------|------------|------------|
| YOLOv5n | 1.9 MB | 28.0% mAP | ~40 ms | ~2.5 ms |
| YOLOv5s | 14.1 MB | 37.4% mAP | ~98 ms | ~2.6 ms |
| YOLOv5m | 52.3 MB | 45.4% mAP | ~224 ms | ~3.4 ms |
| YOLOv5l | 190.7 MB | 49.0% mAP | ~430 ms | ~4.5 ms |
| YOLOv5x | 344.1 MB | 50.7% mAP | ~766 ms | ~6.9 ms |

*Note: Speed metrics based on 640x640 images; actual performance may vary based on hardware.*
