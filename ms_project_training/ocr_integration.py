import os
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse
from pathlib import Path

# Add root path to use shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectOCR:
    """OCR integration for Microsoft Project element text extraction"""
    
    def __init__(self, tesseract_path=None):
        # Set Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try to find Tesseract in common locations
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'/usr/bin/tesseract',
                r'/usr/local/bin/tesseract'
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found Tesseract at: {path}")
                    break
        
        # OCR configuration
        self.config = {
            'task': '--psm 7 --oem 3',  # Single line text
            'milestone': '--psm 7 --oem 3',  # Single line text
            'resource': '--psm 7 --oem 3',  # Single line text
            'date': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/.',  # Dates
            'number': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.%'  # Numbers with percentages
        }
    
    def _preprocess_image(self, image, element_type):
        """Preprocess image for better OCR results based on element type"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply different preprocessing based on element type
        if element_type == 'task' or element_type == 'resource':
            # Text is usually dark on light background
            # Adaptive thresholding works well for text
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
        elif element_type == 'date' or element_type == 'number':
            # Simple binary thresholding may work better for numbers/dates
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:  # Default processing
            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def extract_text(self, image, element_type='task'):
        """Extract text from an image region using OCR"""
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                raise FileNotFoundError(f"Image file not found: {image}")
        
        # Ensure we have a valid image
        if image is None or image.size == 0:
            return ""
        
        # Preprocess the image
        preprocessed = self._preprocess_image(image, element_type)
        
        # Get OCR configuration
        config = self.config.get(element_type, self.config['task'])
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(preprocessed, config=config).strip()
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def extract_from_detection(self, image, bbox, element_type):
        """Extract text from a detection bounding box"""
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                raise FileNotFoundError(f"Image file not found: {image}")
        
        # Get region of interest from bounding box
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Convert relative coordinates to absolute if needed
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:  # Normalized coordinates
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        # Perform OCR on region
        text = self.extract_text(roi, element_type)
        return text
    
    def process_detections(self, image, detections):
        """Process multiple detections in an image"""
        results = []
        
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                raise FileNotFoundError(f"Image file not found: {image}")
        
        for det in detections:
            element_type = det['class']
            bbox = det['bbox']  # Should be [x1, y1, x2, y2] or [xmin, ymin, xmax, ymax]
            
            # Extract text from detection
            text = self.extract_from_detection(image, bbox, element_type)
            
            # Add text to detection result
            det_result = {
                'class': element_type,
                'bbox': bbox,
                'text': text
            }
            
            results.append(det_result)
        
        return results
    
    def extract_project_data(self, image, detections):
        """Extract structured project data from detections"""
        # Process OCR on all detections
        ocr_results = self.process_detections(image, detections)
        
        # Initialize project data structure
        project_data = {
            'tasks': [],
            'milestones': [],
            'resources': [],
            'dependencies': []
        }
        
        # Organize detections by type
        for result in ocr_results:
            element_type = result['class']
            text = result['text']
            bbox = result['bbox']
            
            if element_type == 'task':
                project_data['tasks'].append({
                    'name': text,
                    'bbox': bbox
                })
            elif element_type == 'milestone':
                project_data['milestones'].append({
                    'name': text,
                    'bbox': bbox
                })
            elif element_type == 'resource':
                project_data['resources'].append({
                    'name': text,
                    'bbox': bbox
                })
            elif element_type == 'dependency':
                project_data['dependencies'].append({
                    'text': text,
                    'bbox': bbox
                })
        
        return project_data
    
    def analyze_project_schedule(self, image, detections):
        """Perform detailed analysis of project schedule from detections with OCR"""
        # Extract basic project data
        project_data = self.extract_project_data(image, detections)
        
        # Additional analysis can be performed here...
        # For example:
        # - Detect relationships between tasks based on positions
        # - Calculate schedule metrics
        # - Identify critical path
        
        # Count elements
        task_count = len(project_data['tasks'])
        milestone_count = len(project_data['milestones'])
        resource_count = len(project_data['resources'])
        dependency_count = len(project_data['dependencies'])
        
        print(f"Project Analysis Results:")
        print(f"- Tasks: {task_count}")
        print(f"- Milestones: {milestone_count}")
        print(f"- Resources: {resource_count}")
        print(f"- Dependencies: {dependency_count}")
        
        if task_count > 0:
            print("\nSample Tasks:")
            for i, task in enumerate(project_data['tasks'][:3]):  # Show first 3 tasks
                print(f"  {i+1}. {task['name']}")
        
        if milestone_count > 0:
            print("\nSample Milestones:")
            for i, milestone in enumerate(project_data['milestones'][:3]):  # Show first 3 milestones
                print(f"  {i+1}. {milestone['name']}")
        
        return project_data

def main():
    parser = argparse.ArgumentParser(description="OCR for Microsoft Project elements")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--detections', type=str, help='Path to detections JSON file (optional)')
    parser.add_argument('--tesseract', type=str, help='Path to Tesseract executable')
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = MSProjectOCR(tesseract_path=args.tesseract)
    
    # Load the image
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    image = cv2.imread(args.image)
    
    # Load detections if provided, otherwise run on entire image
    if args.detections and os.path.exists(args.detections):
        import json
        with open(args.detections, 'r') as f:
            detections = json.load(f)
        
        # Process detections
        results = ocr.process_detections(image, detections)
        
        # Print results
        print(f"OCR Results for {len(results)} detections:")
        for i, res in enumerate(results):
            print(f"\nDetection {i+1} ({res['class']}):")
            print(f"  Text: {res['text']}")
    else:
        # Run OCR on the entire image
        text = ocr.extract_text(image)
        print(f"OCR Results for entire image:")
        print(text)

if __name__ == "__main__":
    main()
