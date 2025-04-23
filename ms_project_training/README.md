# YOLOv5 Microsoft Project Training Pipeline

This specialized training pipeline enables the creation of custom YOLOv5 models for detecting and extracting data from Microsoft Project interfaces. The system combines computer vision-based detection with OCR to provide a comprehensive solution for project management data extraction.

## Pipeline Architecture

```
ms_project_training/
├── collect_screenshots.py   # Automated screenshot capture utility
├── data_generator.py       # Synthetic data generation for augmentation
├── labeling_tool.py        # Utility for labeling MS Project elements
├── train_model.py          # YOLOv5 model training pipeline
├── ocr_integration.py      # OCR integration for text extraction
├── detector.py             # Integrated detector with OCR
├── data/
│   ├── screenshots/        # Captured MS Project screenshots
│   ├── labeled/            # Annotated training images
│   └── generated/          # Synthetically generated training data
├── models/                 # Trained YOLOv5 models for MS Project
└── ocr/                    # OCR models and utilities
```

## Key Features

### 1. Automated Data Collection

The system provides utilities to automatically capture diverse screenshots from Microsoft Project, including:
- Various view types (Gantt Chart, Calendar, Network Diagram)
- Different project scales (small, medium, large)
- Various UI configurations and themes

### 2. Specialized Element Labeling

The custom labeling tool recognizes project-specific elements:
- Tasks and subtasks
- Milestones
- Critical path segments
- Dependencies (Finish-to-Start, Start-to-Start, etc.)
- Resources and assignments
- Progress indicators
- Timeline elements
- Constraints and deadlines

### 3. Custom YOLOv5 Training

The training pipeline incorporates:
- Transfer learning from pre-trained YOLOv5 models
- Data augmentation strategies specific to UI elements
- Training configuration optimized for interface detection
- Model pruning for faster inference

### 4. Integrated OCR

The OCR integration enables:
- Text extraction from detected elements
- Recognition of dates, durations, and resource names
- Structured data extraction into database format
- Relationship mapping between detected elements

### 5. Database Integration

All extracted data is stored in a structured SQLite database:
- Project metadata
- Task hierarchy and relationships
- Resource allocations
- Temporal data (start dates, end dates, durations)
- Detection confidence metrics

## Usage

This pipeline can be used to:
1. Train custom models for MS Project interface analysis
2. Extract structured project data from screenshots
3. Convert visual project representations to structured database formats
4. Enable AI-powered analysis of project schedules without direct API access

## Requirements

- Python 3.8+
- PyTorch 1.7+
- YOLOv5
- Tesseract OCR
- pytesseract
- OpenCV
- PIL/Pillow
- pyautogui
- labelImg (for manual annotation)
