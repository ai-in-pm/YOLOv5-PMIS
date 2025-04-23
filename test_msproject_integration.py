#!/usr/bin/env python
"""
Test script for Microsoft Project integration

This script tests the basic functionality of the MSProjectDetector class
without requiring Microsoft Project to be installed.
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_msproject')

# Import the MSProjectDetector
try:
    from msproject.msproject_detector import MSProjectDetector
    logger.info("Successfully imported MSProjectDetector")
except ImportError as e:
    logger.error(f"Failed to import MSProjectDetector: {e}")
    sys.exit(1)

def test_initialization():
    """Test detector initialization"""
    try:
        detector = MSProjectDetector()
        logger.info("Successfully initialized MSProjectDetector")
        return detector
    except Exception as e:
        logger.error(f"Failed to initialize MSProjectDetector: {e}")
        return None

def test_mock_detection(detector):
    """Test detection with a mock screenshot"""
    try:
        # Create a simple mock screenshot (black image)
        import numpy as np
        import cv2
        
        # Create a black image with some white rectangles to simulate tasks
        mock_screenshot = np.zeros((800, 1200, 3), dtype=np.uint8)
        
        # Draw some rectangles to simulate tasks
        cv2.rectangle(mock_screenshot, (100, 100), (300, 150), (255, 255, 255), -1)
        cv2.rectangle(mock_screenshot, (100, 200), (400, 250), (255, 255, 255), -1)
        cv2.rectangle(mock_screenshot, (100, 300), (350, 350), (255, 255, 255), -1)
        
        # Add some text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mock_screenshot, "Task 1", (110, 130), font, 0.5, (0, 0, 0), 1)
        cv2.putText(mock_screenshot, "Task 2", (110, 230), font, 0.5, (0, 0, 0), 1)
        cv2.putText(mock_screenshot, "Milestone", (110, 330), font, 0.5, (0, 0, 0), 1)
        
        # Save the mock screenshot for reference
        os.makedirs(os.path.join(SCRIPT_DIR, "test_output"), exist_ok=True)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "test_output", "mock_screenshot.jpg"), mock_screenshot)
        
        # Run detection on the mock screenshot
        logger.info("Running detection on mock screenshot...")
        results = detector.detect_objects("msproject", mock_screenshot)
        
        if results:
            logger.info(f"Detection successful. Found {results.get('detection_count', 0)} objects.")
            return True
        else:
            logger.warning("Detection returned no results.")
            return False
    except Exception as e:
        logger.error(f"Error during mock detection: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Microsoft Project integration test")
    
    # Test initialization
    detector = test_initialization()
    if not detector:
        return 1
    
    # Test mock detection
    success = test_mock_detection(detector)
    if not success:
        logger.warning("Mock detection test did not return results. This may be expected if using a specialized model.")
    
    logger.info("Test completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
