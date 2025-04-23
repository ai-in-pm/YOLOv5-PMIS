# YOLOv5 Project Management Integration System

This advanced integration extends YOLOv5's object detection capabilities to professional project management software, enabling AI-powered analysis of project schedules, resource allocations, and critical paths directly from Microsoft Project and Primavera P6 Professional interfaces. The system includes OCR (Optical Character Recognition) capabilities to extract text from detected elements, providing comprehensive project analysis.

The system supports multiple input sources including webcams, images, videos, screen captures, and streams, allowing for flexible deployment in various scenarios.

> **Note:** This integration is designed to work independently from the main YOLOv5 repository. All necessary dependencies are specified in the requirements.txt file in the parent directory.

## Integration Architecture

The system follows a modular design with specialized components for each project management application:

```text
integrations/
├── pm_integration.py         # Core integration framework
├── msproject/                # Microsoft Project integration
│   ├── msproject_detector.py # MS Project specialized detector
│   ├── models/               # Specialized models for MS Project
│   └── README.md             # MS Project integration documentation
├── primavera/                # Primavera P6 integration
│   ├── primavera_detector.py # Primavera specialized detector
│   ├── models/               # Specialized models for Primavera
│   └── README.md             # Primavera integration documentation
├── ms_project_training/      # Training pipeline for MS Project models
├── ocr_utils.py              # OCR utilities for text extraction
├── pm_gui.py                 # Graphical user interface
├── multi_source_detector.py  # Multi-source detection module
├── database/                 # SQLite database storage
├── results/                  # Detection results and output files
├── run_msproject_detection.py # Python launcher for MS Project detection
├── run_msproject_detection.bat # Windows batch launcher for MS Project
├── run_primavera_detection.py # Python launcher for Primavera detection
├── run_primavera_detection.bat # Windows batch launcher for Primavera
├── run_multi_source_detection.bat # Windows batch launcher for multi-source detection
├── run_multi_source_detection.sh # Linux/macOS shell launcher for multi-source detection
├── run_pm_gui.bat            # Windows batch launcher for GUI
├── run_pm_gui.sh             # Linux/macOS shell launcher for GUI
└── test_msproject_integration.py # Test script for MS Project integration
```

## Core Features

### 1. AI-Powered Project Element Detection with OCR

The system uses YOLOv5 to recognize and extract structured data from project management interfaces, including:

- **Tasks/Activities**: Identification of work items with their properties
- **Milestones**: Key project events and deliverables
- **Dependencies**: Logical relationships between tasks (FS, FF, SS, SF)
- **Resources**: Labor, equipment and material assignments
- **Critical Path**: Identification of schedule-driving activities
- **WBS Elements**: Project breakdown structure components
- **Constraints**: Schedule limitations such as deadlines

With the new OCR integration, the system can now extract text from detected elements, providing:

- **Task Names**: Actual task descriptions from the schedule
- **Resource Names**: Names of assigned resources
- **Dates**: Start and finish dates for activities
- **Durations**: Task duration information
- **Costs**: Budget and cost information where available

### 2. Multi-Source Detection

The system now supports multiple input sources for detection:

- **Webcam**: Live detection from connected camera devices
- **Images**: Detection from static image files
- **Videos**: Detection from video files with frame-by-frame analysis
- **Screen Captures**: Detection from the current screen display
- **Streams**: Detection from video streams (RTSP, HTTP, etc.)

### 3. Application-Specific Analytics

#### Microsoft Project Integration
- Gantt chart analysis
- Resource utilization detection
- Task hierarchy recognition
- Critical path visualization
- Baseline comparison capabilities

#### Primavera P6 Integration

- Activity network diagram analysis
- WBS structure recognition
- Resource histogram detection
- Relationship logic interpretation
- Schedule constraint identification
- Activity code extraction and analysis

### 3. SQLite Database Integration

All detection results are stored in a comprehensive database schema:

- **pm_applications**: Tracks registered project management tools
- **detection_sessions**: Records analysis sessions with timestamps
- **pm_detections**: Stores individual element detections with metadata

## Usage Examples

### Multi-Source Detection Examples

#### Microsoft Project Examples

```bash
# Run detection from webcam
python multi_source_detector.py --app msproject --source webcam --report

# Run detection from image file
python multi_source_detector.py --app msproject --source image --input msproject_screenshot.jpg

# Run continuous detection from screen capture
python multi_source_detector.py --app msproject --source screen --continuous --interval 2.0 --report

# Run detection from video file
python multi_source_detector.py --app msproject --source video --input msproject_recording.mp4 --max-frames 20

# Run detection from video stream
python multi_source_detector.py --app msproject --source stream --input rtsp://example.com/msproject_stream

# Run real-time detection directly in Microsoft Project
python msproject/msproject_realtime.py --start --report

# Run direct detection with visual feedback in Microsoft Project
python run_msproject_direct_detection.py --start --interval 0.5
```

#### Primavera P6 Examples

```bash
# Run detection from webcam
python multi_source_detector.py --app primavera --source webcam --report

# Run detection from image file
python multi_source_detector.py --app primavera --source image --input primavera_screenshot.jpg

# Run continuous detection from screen capture
python multi_source_detector.py --app primavera --source screen --continuous --interval 2.0 --report

# Run detection from video file
python multi_source_detector.py --app primavera --source video --input primavera_recording.mp4 --max-frames 20

# Run detection from video stream
python multi_source_detector.py --app primavera --source stream --input rtsp://example.com/primavera_stream
```

### Batch File Usage

#### Microsoft Project Batch Examples

```bash
# Run detection from webcam
run_multi_source_detection.bat --app msproject --source webcam --report

# Run detection from image file
run_multi_source_detection.bat --app msproject --source image --input msproject_screenshot.jpg

# Run continuous detection from screen capture
run_multi_source_detection.bat --app msproject --source screen --continuous --interval 2.0 --report

# Run real-time detection directly in Microsoft Project
run_msproject_realtime.bat --start --report

# Run direct detection with visual feedback in Microsoft Project
run_msproject_direct_detection.bat --start --interval 0.5
```

#### Primavera P6 Batch Examples

```bash
# Run detection from webcam
run_multi_source_detection.bat --app primavera --source webcam --report

# Run detection from image file
run_multi_source_detection.bat --app primavera --source image --input primavera_screenshot.jpg

# Run continuous detection from screen capture
run_multi_source_detection.bat --app primavera --source screen --continuous --interval 2.0 --report
```

## Usage Instructions

### Quick Start

#### Microsoft Project Detection

The easiest way to use the Microsoft Project integration is through the provided command-line interface:

```bash
python run_msproject_detection.py --start --report
```

Or use the batch file on Windows:

```bash
run_msproject_detection.bat --start --report
```

#### Command Line Options for MS Project/Primavera Detection

- `--model MODEL`: Path to the YOLOv5 model file (default: "msproject_model.pt")
- `--conf CONF`: Detection confidence threshold (default: 0.25)
- `--iou IOU`: IoU threshold for NMS (default: 0.45)
- `--start`: Start Microsoft Project before detection
- `--report`: Generate and display project report
- `--output OUTPUT`: Path to save detection results (JSON format)
- `--tesseract PATH`: Path to Tesseract OCR executable
- `--no-ocr`: Disable OCR text extraction
- `--debug`: Enable debug logging

#### Command Line Options for Multi-Source Detection

- `--app APP`: Application type (msproject or primavera)
- `--source SOURCE`: Input source (webcam, image, video, screen, stream)
- `--input PATH`: Input file path or URL (for image, video, stream)
- `--camera ID`: Camera device ID (for webcam, default: 0)
- `--model MODEL`: Path to the YOLOv5 model file
- `--conf CONF`: Detection confidence threshold (default: 0.25)
- `--iou IOU`: IoU threshold for NMS (default: 0.45)
- `--tesseract PATH`: Path to Tesseract OCR executable
- `--no-ocr`: Disable OCR text extraction
- `--continuous`: Run continuous detection
- `--interval SEC`: Time interval between detections (default: 1.0)
- `--max-frames N`: Maximum number of frames to process (default: 10)
- `--max-time SEC`: Maximum time to run detection (default: 30)
- `--report`: Generate and display project report
- `--debug`: Enable debug logging

#### Command Line Options for Real-Time Microsoft Project Detection

- `--model MODEL`: Path to the YOLOv5 model file
- `--conf CONF`: Detection confidence threshold (default: 0.25)
- `--iou IOU`: IoU threshold for NMS (default: 0.45)
- `--tesseract PATH`: Path to Tesseract OCR executable
- `--no-ocr`: Disable OCR text extraction
- `--interval SEC`: Time interval between detections (default: 1.0)
- `--no-highlight`: Disable highlighting of detected elements
- `--report`: Generate and display project report periodically
- `--report-interval SEC`: Interval for generating reports (default: 10)
- `--debug`: Enable debug logging
- `--start`: Start Microsoft Project if not already running

### Python API Usage

For programmatic integration, you can use the Python API directly:

```python
# Microsoft Project integration
from msproject.msproject_detector import MSProjectDetector

# Create detector
detector = MSProjectDetector()

# Start Microsoft Project (optional)
detector.start_application("msproject")

# Detect project elements with OCR
results = detector.detect_project_elements(with_ocr=True)

# Generate report
report = detector.generate_project_report()
print(report)
```

### Multi-Source Detection API

For multi-source detection, use the MultiSourceDetector class:

#### Microsoft Project API Example

```python
from multi_source_detector import MultiSourceDetector

# Create detector for Microsoft Project
detector = MultiSourceDetector(
    app_type="msproject",
    model_path="msproject_model.pt",
    conf_thres=0.25,
    use_ocr=True
)

# Detect from webcam
results = detector.detect_from_webcam(camera_id=0)

# Detect from image file
results = detector.detect_from_image("msproject_screenshot.jpg")

# Detect from video file
results = detector.detect_from_video(
    video_path="msproject_recording.mp4",
    interval=1.0,
    max_frames=10
)

# Detect from screen capture
results = detector.detect_from_screen()

# Detect from video stream
results = detector.detect_from_stream("rtsp://example.com/msproject_stream")

# Start continuous detection
detector.start_continuous_detection(
    source_type="screen",
    interval=2.0,
    callback=lambda results: print(f"Detected {results.get('detection_count', 0)} elements")
)

# Stop continuous detection
detector.stop_continuous_detection()
```

#### Real-Time Microsoft Project API Example

```python
from msproject.msproject_realtime import MSProjectRealTimeDetector

# Create real-time detector
detector = MSProjectRealTimeDetector(
    model_path="msproject_model.pt",
    conf_thres=0.25,
    use_ocr=True,
    interval=1.0,
    highlight_elements=True
)

# Define callback function
def detection_callback(results):
    print(f"Detected {results.get('detection_count', 0)} elements")
    # Process results as needed

# Start real-time detection
detector.start_realtime_detection(detection_callback)

# Stop real-time detection
detector.stop_realtime_detection()

# Generate report from last detection
report = detector.generate_report()
print(report)
```

#### Primavera P6 API Example

```python
from multi_source_detector import MultiSourceDetector

# Create detector for Primavera P6
detector = MultiSourceDetector(
    app_type="primavera",
    model_path="primavera_model.pt",
    conf_thres=0.25,
    use_ocr=True
)

# Detect from webcam
results = detector.detect_from_webcam(camera_id=0)

# Detect from image file
results = detector.detect_from_image("primavera_screenshot.jpg")

# Detect from video file
results = detector.detect_from_video(
    video_path="primavera_recording.mp4",
    interval=1.0,
    max_frames=10
)

# Detect from screen capture
results = detector.detect_from_screen()

# Detect from video stream
results = detector.detect_from_stream("rtsp://example.com/primavera_stream")

# Start continuous detection
detector.start_continuous_detection(
    source_type="screen",
    interval=2.0,
    callback=lambda results: print(f"Detected {results.get('detection_count', 0)} elements")
)

# Stop continuous detection
detector.stop_continuous_detection()
```

## Requirements

1. **Python Environment**:
   - Python 3.8+
   - PyTorch 1.7+
   - YOLOv5 dependencies

2. **Project Management Software**:
   - Microsoft Project: Common paths are checked automatically
   - Primavera P6 Professional: Fully supported

3. **Additional Python Libraries**:
   - pyautogui
   - pywin32 (for Windows integration)
   - OpenCV (cv2)
   - numpy
   - pytesseract (for OCR functionality)
   - Tesseract OCR (external dependency for text extraction)

## Advanced Configuration

### Training Custom Detection Models

For improved detection accuracy, you can train specialized models for each project management application:

1. **Capture training data**: Take screenshots of the application interfaces
2. **Label project elements**: Create annotations for tasks, milestones, etc.
3. **Train custom model**: Use YOLOv5 training pipeline with your dataset
4. **Deploy model**: Replace the default model with your custom trained model

### Customizing Element Classes

You can modify the detection classes by editing the `object_classes` lists in:

- `msproject_detector.py` for Microsoft Project
- `primavera_detector.py` for Primavera P6

## Integration Data Flow

### Screen Capture Flow

1. **Application Launch**: System starts the project management software
2. **Screen Capture**: Interface is captured using screenshot functionality
3. **AI Detection**: YOLOv5 processes the image to identify project elements
4. **OCR Processing**: Text is extracted from detected elements using Tesseract OCR
5. **Data Extraction**: Detected elements are processed into structured data
6. **Database Storage**: Results are stored for historical analysis
7. **Report Generation**: Comprehensive reports provide schedule insights

### Multi-Source Flow

1. **Source Selection**: System connects to the selected input source (webcam, image, video, screen, stream)
2. **Frame Acquisition**: Frames are acquired from the source
3. **AI Detection**: YOLOv5 processes each frame to identify project elements
4. **OCR Processing**: Text is extracted from detected elements using Tesseract OCR
5. **Data Extraction**: Detected elements are processed into structured data
6. **Results Storage**: Detection results are saved to JSON files
7. **Report Generation**: Comprehensive reports provide schedule insights

### Real-Time Microsoft Project Flow

1. **Window Detection**: System finds the Microsoft Project application window
2. **Continuous Monitoring**: Screenshots of the window are captured at regular intervals
3. **AI Detection**: YOLOv5 processes each screenshot to identify project elements
4. **OCR Processing**: Text is extracted from detected elements using Tesseract OCR
5. **Visual Feedback**: Detected elements are highlighted in a separate window
6. **Data Extraction**: Detected elements are processed into structured data
7. **Results Storage**: Detection results are saved to JSON files
8. **Report Generation**: Comprehensive reports provide schedule insights

### Direct Detection Flow

1. **Window Detection**: System finds the Microsoft Project application window
2. **Real-Time Monitoring**: Screenshots of the window are captured at high frequency (0.5s intervals)
3. **AI Detection**: YOLOv5 processes each screenshot to identify project elements
4. **Enhanced Visual Feedback**: Detected elements are highlighted with semi-transparent overlays
5. **Legend Display**: A legend shows the types of detected elements with color coding
6. **Live Updates**: Detection results are updated in real-time as the user interacts with Microsoft Project
7. **Immediate Feedback**: Users can see what the AI is detecting as they work

## Graphical User Interface

The system includes a graphical user interface for easier interaction:

1. **Launch the GUI**: Run `run_pm_gui.bat` (Windows) or `run_pm_gui.sh` (Linux/macOS)
2. **Select Application**: Choose between Microsoft Project and Primavera P6
3. **Configure Settings**: Set model path, confidence threshold, and OCR options
4. **Run Detection**: Start the application and detect elements with a single click
5. **View Results**: See detection results and reports in the interface

## Testing

To test the integration without having Microsoft Project installed, you can use the test script:

```bash
python test_msproject_integration.py
```

This will create a mock screenshot and run the detection pipeline on it.

## Extending the Integration

This system can be extended to other project management applications by:

1. Creating a specialized detector class inheriting from `ProjectManagementDetector`
2. Implementing application-specific methods for element detection
3. Training custom models for the new application interface
4. Adding the application to the launcher interface

## Limitations

- Accuracy depends on the visual consistency of the application interface
- Initial detections may require verification for critical project decisions
- Performance varies based on project complexity and screen resolution
- Some project elements may need specialized detection approaches
- OCR accuracy depends on text clarity and Tesseract configuration
- Text extraction may be less reliable for small or low-contrast text
- Video and stream processing may be resource-intensive
- Webcam detection quality depends on camera resolution and lighting conditions
- Real-time detection requires Microsoft Project to be running on Windows
- Window detection may not work with all versions of Microsoft Project
