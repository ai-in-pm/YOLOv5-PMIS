"""
OCR Utilities for Project Management Integration

This module provides OCR (Optical Character Recognition) functionality for extracting
text from detected elements in project management applications.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger('ocr_utils')

class OCRProcessor:
    """Base class for OCR processing"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR processor
        
        Args:
            tesseract_path: Optional path to Tesseract executable
        """
        self.tesseract_available = False
        self.tesseract_path = tesseract_path
        
        # Try to import pytesseract
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
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
                        logger.info(f"Found Tesseract at: {path}")
                        break
            
            # Test if Tesseract is available
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info(f"Tesseract OCR initialized successfully (version: {pytesseract.get_tesseract_version()})")
            except Exception as e:
                logger.warning(f"Tesseract is installed but not working properly: {e}")
                self.tesseract_available = False
                
        except ImportError:
            logger.warning("pytesseract not installed. OCR functionality will be limited.")
            self.pytesseract = None
    
    def _preprocess_image(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """Preprocess image for better OCR results based on element type
        
        Args:
            image: Input image as numpy array
            element_type: Type of element (task, milestone, etc.)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply different preprocessing based on element type
        if element_type in ['task', 'milestone', 'resource', 'activity']:
            # Text is usually dark on light background
            # Adaptive thresholding works well for text
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
        elif element_type in ['date', 'number', 'duration']:
            # Simple binary thresholding may work better for numbers/dates
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:
            # Default preprocessing
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        
        # Additional preprocessing
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Dilation to make text thicker and more readable
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated
    
    def extract_text(self, image: Union[str, np.ndarray], element_type: str = 'task') -> str:
        """Extract text from an image region using OCR
        
        Args:
            image: Input image as numpy array or path to image file
            element_type: Type of element (task, milestone, etc.)
            
        Returns:
            Extracted text
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. Cannot extract text.")
            return ""
        
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                logger.error(f"Image file not found: {image}")
                return ""
        
        # Ensure we have a valid image
        if image is None or image.size == 0:
            logger.warning("Invalid image for OCR")
            return ""
        
        # Preprocess the image
        preprocessed = self._preprocess_image(image, element_type)
        
        # Get OCR configuration based on element type
        config = self._get_config(element_type)
        
        # Perform OCR
        try:
            text = self.pytesseract.image_to_string(preprocessed, config=config).strip()
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _get_config(self, element_type: str) -> str:
        """Get Tesseract configuration for specific element type
        
        Args:
            element_type: Type of element (task, milestone, etc.)
            
        Returns:
            Tesseract configuration string
        """
        # Default configurations for different element types
        configs = {
            'task': '--psm 7 --oem 3',  # Single line text
            'milestone': '--psm 7 --oem 3',  # Single line text
            'resource': '--psm 7 --oem 3',  # Single line text
            'activity': '--psm 7 --oem 3',  # Single line text
            'date': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/.-',  # Dates
            'number': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.%',  # Numbers with percentages
            'duration': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.dhw ',  # Durations
            'default': '--psm 6 --oem 3'  # Assume sparse text
        }
        
        return configs.get(element_type, configs['default'])
    
    def extract_from_detection(self, image: Union[str, np.ndarray], bbox: List[float], 
                              element_type: str) -> str:
        """Extract text from a detection bounding box
        
        Args:
            image: Input image as numpy array or path to image file
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            element_type: Type of element (task, milestone, etc.)
            
        Returns:
            Extracted text
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. Cannot extract text.")
            return ""
        
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                logger.error(f"Image file not found: {image}")
                return ""
        
        # Get region of interest from bounding box
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Convert relative coordinates to absolute if needed
        if max(x1, y1, x2, y2) <= 1.0:  # Normalized coordinates
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        else:
            # Ensure coordinates are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        # Check if ROI is valid
        if roi.size == 0:
            logger.warning(f"Invalid ROI for OCR: {bbox}")
            return ""
        
        # Perform OCR on region
        text = self.extract_text(roi, element_type)
        return text
    
    def process_detections(self, image: Union[str, np.ndarray], 
                          detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple detections in an image
        
        Args:
            image: Input image as numpy array or path to image file
            detections: List of detection dictionaries
            
        Returns:
            List of detections with extracted text
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. Cannot process detections.")
            return detections
        
        # Check if image is a file path or numpy array
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                logger.error(f"Image file not found: {image}")
                return detections
        
        results = []
        
        for det in detections:
            # Make a copy of the detection
            det_result = det.copy()
            
            # Extract element type and bounding box
            element_type = det.get('class', 'default')
            
            # Handle different bbox formats
            if 'bbox' in det:
                bbox = det['bbox']
            elif all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                bbox = [det['x1'], det['y1'], det['x2'], det['y2']]
            else:
                logger.warning(f"Invalid detection format: {det}")
                results.append(det_result)
                continue
            
            # Extract text from detection
            text = self.extract_from_detection(image, bbox, element_type)
            
            # Add text to detection result
            det_result['text'] = text
            
            results.append(det_result)
        
        return results


class MSProjectOCR(OCRProcessor):
    """Specialized OCR processor for Microsoft Project elements"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize MS Project OCR processor
        
        Args:
            tesseract_path: Optional path to Tesseract executable
        """
        super().__init__(tesseract_path)
        
        # MS Project specific configurations
        self.ms_project_configs = {
            'task': '--psm 7 --oem 3',  # Single line text
            'milestone': '--psm 7 --oem 3',  # Single line text
            'summary': '--psm 7 --oem 3 -c preserve_interword_spaces=1',  # Summary tasks are often bold
            'critical_task': '--psm 7 --oem 3',  # Critical tasks may be in red
            'resource': '--psm 7 --oem 3',  # Resource names
            'date': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/.-',  # Dates
            'duration': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.dhw ',  # Durations
            'cost': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,$€£¥',  # Cost values
            'percent': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.%',  # Percentages
        }
    
    def _get_config(self, element_type: str) -> str:
        """Get MS Project specific Tesseract configuration
        
        Args:
            element_type: Type of element (task, milestone, etc.)
            
        Returns:
            Tesseract configuration string
        """
        return self.ms_project_configs.get(element_type, super()._get_config(element_type))
    
    def extract_project_data(self, image: Union[str, np.ndarray], 
                            detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract structured project data from detections
        
        Args:
            image: Input image as numpy array or path to image file
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of structured project data
        """
        # Process OCR on all detections
        ocr_results = self.process_detections(image, detections)
        
        # Initialize project data structure
        project_data = {
            'tasks': [],
            'milestones': [],
            'resources': [],
            'dependencies': [],
            'critical_path': [],
            'summary': [],
            'constraints': [],
            'deadlines': [],
            'baselines': []
        }
        
        # Organize detections by type
        for result in ocr_results:
            element_type = result.get('class', '')
            text = result.get('text', '')
            
            # Get bbox in consistent format
            if 'bbox' in result:
                bbox = result['bbox']
            elif all(k in result for k in ['x1', 'y1', 'x2', 'y2']):
                bbox = [result['x1'], result['y1'], result['x2'], result['y2']]
            else:
                logger.warning(f"Invalid detection format: {result}")
                continue
            
            # Create element data
            element_data = {
                'name': text,
                'bbox': bbox,
                'confidence': result.get('confidence', 0.0)
            }
            
            # Add to appropriate category
            if element_type == 'task':
                project_data['tasks'].append(element_data)
            elif element_type == 'milestone':
                project_data['milestones'].append(element_data)
            elif element_type == 'resource':
                project_data['resources'].append(element_data)
            elif element_type == 'dependency' or element_type == 'gantt_bar':
                project_data['dependencies'].append(element_data)
            elif element_type == 'critical_task':
                project_data['critical_path'].append(element_data)
            elif element_type == 'summary':
                project_data['summary'].append(element_data)
            elif element_type == 'constraint':
                project_data['constraints'].append(element_data)
            elif element_type == 'deadline':
                project_data['deadlines'].append(element_data)
            elif element_type == 'baseline':
                project_data['baselines'].append(element_data)
        
        return project_data


class PrimaveraOCR(OCRProcessor):
    """Specialized OCR processor for Primavera P6 elements"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize Primavera OCR processor
        
        Args:
            tesseract_path: Optional path to Tesseract executable
        """
        super().__init__(tesseract_path)
        
        # Primavera specific configurations
        self.primavera_configs = {
            'activity': '--psm 7 --oem 3',  # Activity names
            'milestone': '--psm 7 --oem 3',  # Milestone names
            'wbs': '--psm 7 --oem 3 -c preserve_interword_spaces=1',  # WBS elements
            'critical_activity': '--psm 7 --oem 3',  # Critical activities
            'resource': '--psm 7 --oem 3',  # Resource names
            'relationship': '--psm 7 --oem 3',  # Relationship types (FS, SS, FF, SF)
            'constraint': '--psm 7 --oem 3',  # Constraint types
            'progress_bar': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.%',  # Progress values
            'date': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/.-',  # Dates
            'duration': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.dhw ',  # Durations
        }
    
    def _get_config(self, element_type: str) -> str:
        """Get Primavera specific Tesseract configuration
        
        Args:
            element_type: Type of element (activity, milestone, etc.)
            
        Returns:
            Tesseract configuration string
        """
        return self.primavera_configs.get(element_type, super()._get_config(element_type))
    
    def extract_project_data(self, image: Union[str, np.ndarray], 
                            detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract structured project data from detections
        
        Args:
            image: Input image as numpy array or path to image file
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of structured project data
        """
        # Process OCR on all detections
        ocr_results = self.process_detections(image, detections)
        
        # Initialize project data structure
        project_data = {
            'activities': [],
            'milestones': [],
            'resources': [],
            'relationships': [],
            'critical_path': [],
            'wbs_elements': [],
            'constraints': [],
            'progress': [],
            'baselines': [],
            'notebooks': []
        }
        
        # Organize detections by type
        for result in ocr_results:
            element_type = result.get('class', '')
            text = result.get('text', '')
            
            # Get bbox in consistent format
            if 'bbox' in result:
                bbox = result['bbox']
            elif all(k in result for k in ['x1', 'y1', 'x2', 'y2']):
                bbox = [result['x1'], result['y1'], result['x2'], result['y2']]
            else:
                logger.warning(f"Invalid detection format: {result}")
                continue
            
            # Create element data
            element_data = {
                'name': text,
                'bbox': bbox,
                'confidence': result.get('confidence', 0.0)
            }
            
            # Add to appropriate category
            if element_type == 'activity':
                project_data['activities'].append(element_data)
            elif element_type == 'milestone':
                project_data['milestones'].append(element_data)
            elif element_type == 'resource':
                project_data['resources'].append(element_data)
            elif element_type == 'relationship':
                project_data['relationships'].append(element_data)
            elif element_type == 'critical_activity':
                project_data['critical_path'].append(element_data)
            elif element_type == 'wbs':
                project_data['wbs_elements'].append(element_data)
            elif element_type == 'constraint':
                project_data['constraints'].append(element_data)
            elif element_type == 'progress_bar':
                project_data['progress'].append(element_data)
            elif element_type == 'baseline':
                project_data['baselines'].append(element_data)
            elif element_type == 'notebook':
                project_data['notebooks'].append(element_data)
        
        return project_data


# Factory function to get the appropriate OCR processor
def get_ocr_processor(app_type: str, tesseract_path: Optional[str] = None) -> OCRProcessor:
    """Get the appropriate OCR processor for the application type
    
    Args:
        app_type: Application type ('msproject' or 'primavera')
        tesseract_path: Optional path to Tesseract executable
        
    Returns:
        OCR processor instance
    """
    if app_type.lower() == 'msproject':
        return MSProjectOCR(tesseract_path)
    elif app_type.lower() == 'primavera':
        return PrimaveraOCR(tesseract_path)
    else:
        logger.warning(f"Unknown application type: {app_type}. Using generic OCR processor.")
        return OCRProcessor(tesseract_path)


# Test function
def main():
    """Test OCR functionality"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test OCR functionality")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--app', type=str, choices=['msproject', 'primavera'], default='msproject',
                        help='Application type')
    parser.add_argument('--tesseract', type=str, help='Path to Tesseract executable')
    parser.add_argument('--element', type=str, default='task', help='Element type for OCR')
    parser.add_argument('--bbox', type=str, help='Bounding box coordinates (x1,y1,x2,y2)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get OCR processor
    ocr = get_ocr_processor(args.app, args.tesseract)
    
    # Check if Tesseract is available
    if not ocr.tesseract_available:
        logger.error("Tesseract OCR is not available. Please install it and try again.")
        return
    
    # Load the image
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return
    
    image = cv2.imread(args.image)
    
    # Process the image
    if args.bbox:
        # Extract text from specific region
        try:
            x1, y1, x2, y2 = map(int, args.bbox.split(','))
            text = ocr.extract_from_detection(image, [x1, y1, x2, y2], args.element)
            logger.info(f"OCR Result for {args.element} at {args.bbox}:")
            logger.info(f"Text: {text}")
        except ValueError:
            logger.error("Invalid bbox format. Use x1,y1,x2,y2")
    else:
        # Process entire image
        text = ocr.extract_text(image, args.element)
        logger.info(f"OCR Result for entire image as {args.element}:")
        logger.info(f"Text: {text}")


if __name__ == "__main__":
    main()
