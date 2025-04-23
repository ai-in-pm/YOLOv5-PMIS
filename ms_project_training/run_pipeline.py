import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add root path for shared utilities
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_path)

class MSProjectTrainingPipeline:
    """Unified interface for the complete MS Project training pipeline"""
    
    def __init__(self):
        self.pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.pipeline_dir, 'data')
        self.models_dir = os.path.join(self.pipeline_dir, 'models')
        self.output_dir = os.path.join(self.pipeline_dir, 'output')
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.models_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def collect_screenshots(self, count=20, views=None):
        """Run screenshot collection module"""
        print("\n=== Step 1: Collecting Microsoft Project Screenshots ===")
        
        collect_script = os.path.join(self.pipeline_dir, 'collect_screenshots.py')
        
        cmd = [
            'python',
            collect_script,
            '--count', str(count)
        ]
        
        if views:
            cmd.extend(['--views'] + views)
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print("Screenshot collection completed successfully!")
                return True
            else:
                print(f"Screenshot collection failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error during screenshot collection: {e}")
            return False
    
    def generate_synthetic_data(self, count=50):
        """Run synthetic data generation module"""
        print("\n=== Step 2: Generating Synthetic Training Data ===")
        
        generate_script = os.path.join(self.pipeline_dir, 'data_generator.py')
        
        cmd = [
            'python',
            generate_script,
            '--count', str(count),
            '--create-yaml'
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print("Synthetic data generation completed successfully!")
                return True
            else:
                print(f"Synthetic data generation failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error during synthetic data generation: {e}")
            return False
    
    def label_screenshots(self):
        """Run the labeling tool for manual annotation"""
        print("\n=== Step 3: Labeling Microsoft Project Elements ===")
        print("This step will open the graphical labeling tool for manual annotation.")
        print("You'll need to label all project elements in the collected screenshots.")
        print("Press Enter to continue or 's' to skip this step...")
        
        choice = input().lower()
        if choice == 's':
            print("Skipping labeling step...")
            return True
        
        label_script = os.path.join(self.pipeline_dir, 'labeling_tool.py')
        
        cmd = [
            'python',
            label_script,
            '--input', os.path.join(self.data_dir, 'screenshots'),
            '--output', os.path.join(self.data_dir, 'labeled')
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        print("The labeling tool window will open now. Please complete the labeling process.")
        
        try:
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("Labeling completed successfully!")
                return True
            else:
                print(f"Labeling failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error during labeling: {e}")
            return False
    
    def train_model(self, model_size='s', epochs=50, batch_size=16):
        """Run model training on the labeled and synthetic data"""
        print("\n=== Step 4: Training YOLOv5 Model for MS Project Elements ===")
        
        train_script = os.path.join(self.pipeline_dir, 'train_model.py')
        data_yaml = os.path.join(self.data_dir, 'ms_project.yaml')
        
        cmd = [
            'python',
            train_script,
            '--data', data_yaml,
            '--output', self.models_dir,
            '--model', model_size,
            '--epochs', str(epochs),
            '--batch', str(batch_size),
            '--export'
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print("Model training completed successfully!")
                return True
            else:
                print(f"Model training failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error during model training: {e}")
            return False
    
    def run_detector(self, image_path=None):
        """Run the detector with the trained model"""
        print("\n=== Step 5: Running MS Project Element Detector with OCR ===")
        
        detector_script = os.path.join(self.pipeline_dir, 'detector.py')
        
        cmd = [
            'python',
            detector_script,
            '--visualize',
            '--analyze',
            '--report'
        ]
        
        if image_path:
            cmd.extend(['--image', image_path])
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print("Detection completed successfully!")
                return True
            else:
                print(f"Detection failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error during detection: {e}")
            return False
    
    def run_full_pipeline(self, skip_steps=None):
        """Run the complete pipeline from screenshot collection to detection"""
        print("\n==============================================")
        print("Microsoft Project Training Pipeline")
        print("==============================================\n")
        print("This pipeline will:")
        print("1. Collect Microsoft Project screenshots")
        print("2. Generate synthetic training data")
        print("3. Label project elements in screenshots")
        print("4. Train a YOLOv5 model on the labeled data")
        print("5. Run detection with OCR on Microsoft Project interface")
        print("\nStarting pipeline at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        if skip_steps is None:
            skip_steps = []
        
        success = True
        
        # Step 1: Collect screenshots
        if 1 not in skip_steps:
            success = self.collect_screenshots(count=20) and success
        else:
            print("\nSkipping Step 1: Screenshot collection")
        
        # Step 2: Generate synthetic data
        if 2 not in skip_steps:
            success = self.generate_synthetic_data(count=50) and success
        else:
            print("\nSkipping Step 2: Synthetic data generation")
        
        # Step 3: Label screenshots
        if 3 not in skip_steps:
            success = self.label_screenshots() and success
        else:
            print("\nSkipping Step 3: Element labeling")
        
        # Step 4: Train model
        if 4 not in skip_steps:
            success = self.train_model(epochs=30) and success
        else:
            print("\nSkipping Step 4: Model training")
        
        # Step 5: Run detector
        if 5 not in skip_steps:
            success = self.run_detector() and success
        else:
            print("\nSkipping Step 5: Element detection")
        
        print("\n===============================================")
        if success:
            print("Pipeline completed successfully!")
            print("You now have a trained YOLOv5 model for detecting Microsoft Project elements")
            print("and extracting text from them using OCR.")
        else:
            print("Pipeline completed with some errors. Please check the logs above.")
        
        print("\nModels are saved in:", self.models_dir)
        print("Detection results are saved in:", self.output_dir)
        print("===============================================")

def main():
    parser = argparse.ArgumentParser(description="Microsoft Project Training Pipeline")
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5], help='Run a specific step of the pipeline')
    parser.add_argument('--skip', type=int, nargs='+', choices=[1, 2, 3, 4, 5], help='Skip specific steps of the pipeline')
    parser.add_argument('--image', type=str, help='Path to image for detection (when running only step 5)')
    parser.add_argument('--screenshot-count', type=int, default=20, help='Number of screenshots to collect (step 1)')
    parser.add_argument('--synthetic-count', type=int, default=50, help='Number of synthetic images to generate (step 2)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (step 4)')
    parser.add_argument('--model-size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv5 model size (step 4)')
    
    args = parser.parse_args()
    
    pipeline = MSProjectTrainingPipeline()
    
    if args.step:
        # Run a specific step
        if args.step == 1:
            pipeline.collect_screenshots(count=args.screenshot_count)
        elif args.step == 2:
            pipeline.generate_synthetic_data(count=args.synthetic_count)
        elif args.step == 3:
            pipeline.label_screenshots()
        elif args.step == 4:
            pipeline.train_model(model_size=args.model_size, epochs=args.epochs)
        elif args.step == 5:
            pipeline.run_detector(image_path=args.image)
    else:
        # Run the complete pipeline
        pipeline.run_full_pipeline(skip_steps=args.skip)

if __name__ == "__main__":
    main()
