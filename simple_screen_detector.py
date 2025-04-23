import os
import sys
import torch
from PIL import ImageGrab
from datetime import datetime

# Add YOLOv5 path to system path
yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(yolo_path)

class ScreenDetector:
    """Simple screen capture and YOLOv5 detection"""
    
    def __init__(self):
        self.model = None
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self, weights=None):
        """Load YOLOv5 model"""
        try:
            if weights is None:
                # Use a pre-trained model
                print("Loading YOLOv5s model...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            else:
                # Load custom weights
                print(f"Loading custom weights from {weights}...")
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
                
            model.conf = 0.25  # confidence threshold
            model.iou = 0.45   # NMS IoU threshold
            self.model = model
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def capture_screen(self):
        """Capture the current screen"""
        try:
            print("Capturing screen...")
            screenshot = ImageGrab.grab()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
            screenshot.save(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            return screenshot_path
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def detect_elements(self, image_path=None):
        """Run YOLOv5 detection on the image"""
        if self.model is None:
            if not self.load_model():
                print("Failed to load YOLOv5 model. Aborting detection.")
                return None
        
        if image_path is None:
            image_path = self.capture_screen()
            if image_path is None:
                return None
        
        try:
            print(f"Running YOLOv5 detection on {image_path}...")
            results = self.model(image_path)
            
            # Save detection results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results.save(save_dir=self.output_dir)
            results.print()
            
            # Process and print detections
            detections = self._process_results(results)
            print(f"Detection completed. Found {len(detections)} objects.")
            return detections
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def _process_results(self, results):
        """Process detection results"""
        try:
            detections = []
            # Get the detection results
            if hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
                # Pandas DataFrame format
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    det = {
                        'class': row['name'],
                        'confidence': row['confidence'],
                        'box': (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    }
                    detections.append(det)
            else:
                # Tensor format
                for det_tensor in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det_tensor.tolist()
                    cls_name = results.names[int(cls)]
                    
                    det = {
                        'class': cls_name,
                        'confidence': conf,
                        'box': (x1, y1, x2, y2)
                    }
                    detections.append(det)
            
            # Print detection results
            self._print_detections(detections)
            
            return detections
        except Exception as e:
            print(f"Error processing results: {e}")
            return []
    
    def _print_detections(self, detections):
        """Print detection results in a readable format"""
        print("\nDetection Results:")
        print("-" * 50)
        print(f"{'Class':<15} {'Confidence':<10} {'Box Coordinates':<30}")
        print("-" * 50)
        
        for det in detections:
            box_str = f"({det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f})"
            print(f"{det['class']:<15} {det['confidence']:.4f}      {box_str:<30}")
    
    def interpret_ms_project_elements(self, detections):
        """Interpret detected elements in the context of Microsoft Project"""
        # Map general YOLOv5 classes to potential MS Project elements
        project_mapping = {
            'person': 'resource',
            'cell phone': 'task', 
            'book': 'document',
            'laptop': 'workstation',
            'monitor': 'screen',
            'mouse': 'pointer',
            'keyboard': 'input device',
            'clock': 'timeline',
            'tv': 'view'  
        }
        
        interpreted = []
        for det in detections:
            interpretation = {
                'original_class': det['class'],
                'project_element': project_mapping.get(det['class'], 'unknown element'),
                'confidence': det['confidence'],
                'box': det['box']
            }
            interpreted.append(interpretation)
        
        # Print interpretations
        print("\nMicrosoft Project Element Interpretations:")
        print("-" * 60)
        print(f"{'Detected As':<15} {'Project Element':<20} {'Confidence':<10}")
        print("-" * 60)
        
        for item in interpreted:
            print(f"{item['original_class']:<15} {item['project_element']:<20} {item['confidence']:.4f}")
        
        # Summarize project elements
        element_counts = {}
        for item in interpreted:
            element_type = item['project_element']
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
            
        print("\nSummary of Project Elements:")
        for element, count in element_counts.items():
            print(f"- {count} {element}(s)")
            
        return interpreted

def main():
    detector = ScreenDetector()
    
    print("YOLOv5 Screen Element Detector")
    print("=============================")
    
    # Capture screen and detect
    detections = detector.detect_elements()
    
    if detections and len(detections) > 0:
        # Interpret in MS Project context
        detector.interpret_ms_project_elements(detections)
    else:
        print("No objects detected or an error occurred.")

if __name__ == "__main__":
    main()
