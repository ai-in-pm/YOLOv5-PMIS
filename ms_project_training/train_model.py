import os
import sys
import argparse
import yaml
import shutil
import torch
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'train_model.log'))
    ]
)
logger = logging.getLogger('train_model')

# Add root path to use YOLOv5
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

# Try to import OCR utilities
try:
    sys.path.append(str(SCRIPT_DIR.parent))
    from ocr_utils import MSProjectOCR
    HAS_OCR = True
    logger.info("OCR utilities imported successfully")
except ImportError:
    HAS_OCR = False
    logger.warning("OCR utilities not available. Text extraction will be disabled.")

class MSProjectModelTrainer:
    """YOLOv5 model trainer for Microsoft Project elements"""

    def __init__(self, data_yaml=None, output_dir=None, tesseract_path=None):
        self.data_yaml = data_yaml or os.path.join(SCRIPT_DIR, 'data', 'ms_project.yaml')
        self.output_dir = output_dir or os.path.join(SCRIPT_DIR, 'models')
        self.yolov5_dir = ROOT_DIR
        self.tesseract_path = tesseract_path

        # Initialize OCR if available
        if HAS_OCR:
            self.ocr = MSProjectOCR(tesseract_path=tesseract_path)
            logger.info("OCR processor initialized")
        else:
            self.ocr = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Verify YOLOv5 directory exists
        if not os.path.exists(os.path.join(self.yolov5_dir, 'train.py')):
            raise FileNotFoundError(f"YOLOv5 directory not found at {self.yolov5_dir}")

    def validate_dataset(self):
        """Validate the dataset configuration and check if files exist"""
        logger.info("Validating dataset configuration...")

        if not os.path.exists(self.data_yaml):
            logger.error(f"Dataset configuration not found at {self.data_yaml}")
            raise FileNotFoundError(f"Dataset configuration not found at {self.data_yaml}")

        # Load YAML configuration
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Extract paths
        dataset_path = data_config.get('path')
        train_path = os.path.join(dataset_path, data_config.get('train'))
        val_path = os.path.join(dataset_path, data_config.get('val'))

        # Verify paths exist
        if not os.path.exists(train_path):
            logger.error(f"Training data directory not found at {train_path}")
            raise FileNotFoundError(f"Training data directory not found at {train_path}")

        if not os.path.exists(val_path):
            logger.error(f"Validation data directory not found at {val_path}")
            raise FileNotFoundError(f"Validation data directory not found at {val_path}")

        # Count images and labels
        train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        train_labels_dir = os.path.join(dataset_path, 'labeled', 'labels')

        if os.path.exists(train_labels_dir):
            train_labels = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        else:
            train_labels = []

        logger.info(f"Found {len(train_images)} training images")
        logger.info(f"Found {len(train_labels)} training labels")

        if len(train_images) == 0:
            logger.error("No training images found")
            raise ValueError("No training images found")

        if len(train_labels) == 0:
            logger.error("No training labels found")
            raise ValueError("No training labels found")

        logger.info("Dataset validation successful!")
        return True

    def train(self, model_size='s', batch_size=16, epochs=50, image_size=640):
        """Train YOLOv5 model on MS Project dataset"""
        logger.info(f"Starting training with the following parameters:")
        logger.info(f"  - Model: YOLOv5{model_size}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Image size: {image_size}")

        # Validate dataset first
        try:
            self.validate_dataset()
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False

        # Set up training command
        train_script = os.path.join(self.yolov5_dir, 'train.py')
        weights_path = os.path.join(self.yolov5_dir, f"yolov5{model_size}.pt")

        # If weights file doesn't exist, download it
        if not os.path.exists(weights_path):
            logger.info(f"Pre-trained weights not found at {weights_path}")
            logger.info("Downloading pre-trained weights...")
            try:
                result = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
                logger.info("Downloaded pre-trained weights successfully")
            except Exception as e:
                logger.error(f"Failed to download pre-trained weights: {e}")
                return False

        # Prepare the command for training
        cmd = [
            'python',
            train_script,
            '--img', str(image_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', self.data_yaml,
            '--weights', weights_path,
            '--project', self.output_dir,
            '--name', f'ms_project_yolov5{model_size}',
            '--cache'
        ]

        # Run the training
        logger.info("Starting YOLOv5 training process...")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                logger.debug(line.strip())

            # Wait for completion
            process.wait()

            if process.returncode == 0:
                logger.info("Training completed successfully!")

                # Copy the best model to the msproject directory
                best_model = os.path.join(self.output_dir, f'ms_project_yolov5{model_size}', 'weights', 'best.pt')
                if os.path.exists(best_model):
                    msproject_model = os.path.join(SCRIPT_DIR.parent, 'msproject', 'models', 'msproject_model.pt')
                    os.makedirs(os.path.dirname(msproject_model), exist_ok=True)
                    shutil.copy(best_model, msproject_model)
                    logger.info(f"Best model copied to {msproject_model}")

                return True
            else:
                logger.error(f"Training failed with return code {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def export_model(self, model_path=None, format='onnx'):
        """Export trained model to desired format"""
        if model_path is None:
            # Find the best.pt file in the output directory
            model_dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]

            if not model_dirs:
                logger.error("No trained models found in output directory")
                return False

            # Sort by modification time (newest first)
            model_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(self.output_dir, d)), reverse=True)
            best_model = os.path.join(self.output_dir, model_dirs[0], 'weights', 'best.pt')

            if os.path.exists(best_model):
                model_path = best_model
            else:
                logger.error(f"Best model weights not found at {best_model}")
                return False

        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return False

        logger.info(f"Exporting model from {model_path} to {format} format...")

        # Set up export command
        export_script = os.path.join(self.yolov5_dir, 'export.py')

        cmd = [
            'python',
            export_script,
            '--weights', model_path,
            '--include', format,
            '--imgsz', '640'
        ]

        try:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                logger.debug(line.strip())

            # Wait for completion
            process.wait()

            if process.returncode == 0:
                # Get the output file path
                output_file = model_path.replace('.pt', f'.{format}')
                logger.info(f"Model exported successfully to {output_file}")
                return True
            else:
                logger.error(f"Model export failed with return code {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"Error during model export: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5 model for MS Project element detection")
    parser.add_argument('--data', type=str, default=None, help='Path to dataset.yaml configuration')
    parser.add_argument('--output', type=str, default=None, help='Output directory for trained models')
    parser.add_argument('--model', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv5 model size (n, s, m, l, x)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tflite', 'coreml'], help='Export format')
    parser.add_argument('--tesseract', type=str, default=None, help='Path to Tesseract OCR executable')

    args = parser.parse_args()

    trainer = MSProjectModelTrainer(data_yaml=args.data, output_dir=args.output, tesseract_path=args.tesseract)

    try:
        # Train the model
        success = trainer.train(
            model_size=args.model,
            batch_size=args.batch,
            epochs=args.epochs,
            image_size=args.img
        )

        # Export if requested and training was successful
        if success and args.export:
            trainer.export_model(format=args.format)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
