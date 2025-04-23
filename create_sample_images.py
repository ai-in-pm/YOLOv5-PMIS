#!/usr/bin/env python
"""
Create Sample Images for Testing

This script creates sample images for testing the multi-source detector.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR / "sample_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)

def create_msproject_sample():
    """Create a sample MS Project screenshot"""
    # Create a blank image (1280x720, white background)
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Draw a grid to simulate MS Project table
    # Vertical lines
    for x in range(0, 1280, 100):
        cv2.line(img, (x, 0), (x, 720), (200, 200, 200), 1)
    
    # Horizontal lines
    for y in range(0, 720, 30):
        cv2.line(img, (0, y), (1280, y), (200, 200, 200), 1)
    
    # Draw header
    cv2.rectangle(img, (0, 0), (1280, 30), (230, 230, 230), -1)
    
    # Draw Gantt chart area
    cv2.rectangle(img, (500, 0), (1280, 720), (245, 245, 245), -1)
    
    # Draw some task bars
    # Task 1
    cv2.rectangle(img, (500, 60), (700, 80), (100, 149, 237), -1)  # Blue
    cv2.rectangle(img, (500, 60), (700, 80), (0, 0, 0), 1)
    
    # Task 2
    cv2.rectangle(img, (550, 90), (800, 110), (100, 149, 237), -1)  # Blue
    cv2.rectangle(img, (550, 90), (800, 110), (0, 0, 0), 1)
    
    # Task 3 (critical)
    cv2.rectangle(img, (600, 120), (900, 140), (255, 0, 0), -1)  # Red
    cv2.rectangle(img, (600, 120), (900, 140), (0, 0, 0), 1)
    
    # Milestone
    cv2.drawMarker(img, (950, 170), (0, 0, 0), cv2.MARKER_DIAMOND, 20, 2)
    
    # Summary task
    cv2.rectangle(img, (500, 200), (1000, 210), (0, 0, 0), -1)
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Column headers
    cv2.putText(img, "Task Name", (10, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Duration", (200, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Start", (300, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Finish", (400, 20), font, 0.5, (0, 0, 0), 1)
    
    # Task names
    cv2.putText(img, "Project Planning", (10, 50), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Task 1", (30, 80), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Task 2", (30, 110), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Task 3", (30, 140), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Milestone 1", (30, 170), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Implementation", (10, 200), font, 0.5, (0, 0, 0), 1)
    
    # Durations
    cv2.putText(img, "20 days", (200, 80), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "25 days", (200, 110), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "30 days", (200, 140), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "0 days", (200, 170), font, 0.5, (0, 0, 0), 1)
    
    # Add dependencies (arrows)
    cv2.arrowedLine(img, (700, 70), (550, 90), (0, 0, 0), 1, tipLength=0.02)
    cv2.arrowedLine(img, (800, 100), (600, 120), (0, 0, 0), 1, tipLength=0.02)
    cv2.arrowedLine(img, (900, 130), (950, 170), (0, 0, 0), 1, tipLength=0.02)
    
    # Save the image
    output_path = SAMPLE_DIR / "msproject_screenshot.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"Created MS Project sample image: {output_path}")
    return output_path

def create_primavera_sample():
    """Create a sample Primavera P6 screenshot"""
    # Create a blank image (1280x720, white background)
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Draw a grid to simulate Primavera table
    # Vertical lines
    for x in range(0, 1280, 100):
        cv2.line(img, (x, 0), (x, 720), (200, 200, 200), 1)
    
    # Horizontal lines
    for y in range(0, 720, 30):
        cv2.line(img, (0, y), (1280, y), (200, 200, 200), 1)
    
    # Draw header
    cv2.rectangle(img, (0, 0), (1280, 30), (220, 220, 240), -1)
    
    # Draw Gantt chart area
    cv2.rectangle(img, (500, 0), (1280, 720), (240, 240, 250), -1)
    
    # Draw some activity bars
    # Activity 1
    cv2.rectangle(img, (500, 60), (700, 80), (0, 128, 0), -1)  # Green
    cv2.rectangle(img, (500, 60), (700, 80), (0, 0, 0), 1)
    
    # Activity 2
    cv2.rectangle(img, (550, 90), (800, 110), (0, 128, 0), -1)  # Green
    cv2.rectangle(img, (550, 90), (800, 110), (0, 0, 0), 1)
    
    # Activity 3 (critical)
    cv2.rectangle(img, (600, 120), (900, 140), (255, 0, 0), -1)  # Red
    cv2.rectangle(img, (600, 120), (900, 140), (0, 0, 0), 1)
    
    # Milestone
    cv2.drawMarker(img, (950, 170), (0, 0, 0), cv2.MARKER_DIAMOND, 20, 2)
    
    # WBS
    cv2.rectangle(img, (500, 200), (1000, 210), (0, 0, 0), -1)
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Column headers
    cv2.putText(img, "Activity ID", (10, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Activity Name", (110, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Duration", (300, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Start", (400, 20), font, 0.5, (0, 0, 0), 1)
    
    # Activity IDs and names
    cv2.putText(img, "WBS1000", (10, 50), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "A1010", (10, 80), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "A1020", (10, 110), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "A1030", (10, 140), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "M1000", (10, 170), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "WBS2000", (10, 200), font, 0.5, (0, 0, 0), 1)
    
    cv2.putText(img, "Project Planning", (110, 50), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Activity 1", (110, 80), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Activity 2", (110, 110), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Activity 3", (110, 140), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Milestone 1", (110, 170), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Implementation", (110, 200), font, 0.5, (0, 0, 0), 1)
    
    # Durations
    cv2.putText(img, "20d", (300, 80), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "25d", (300, 110), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "30d", (300, 140), font, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "0d", (300, 170), font, 0.5, (0, 0, 0), 1)
    
    # Add relationships (arrows)
    cv2.arrowedLine(img, (700, 70), (550, 90), (0, 0, 0), 1, tipLength=0.02)
    cv2.arrowedLine(img, (800, 100), (600, 120), (0, 0, 0), 1, tipLength=0.02)
    cv2.arrowedLine(img, (900, 130), (950, 170), (0, 0, 0), 1, tipLength=0.02)
    
    # Save the image
    output_path = SAMPLE_DIR / "primavera_screenshot.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"Created Primavera P6 sample image: {output_path}")
    return output_path

def main():
    """Create all sample images"""
    print("Creating sample images for testing...")
    
    # Create MS Project sample
    msproject_path = create_msproject_sample()
    
    # Create Primavera P6 sample
    primavera_path = create_primavera_sample()
    
    print("\nSample images created successfully!")
    print(f"MS Project sample: {msproject_path}")
    print(f"Primavera P6 sample: {primavera_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
